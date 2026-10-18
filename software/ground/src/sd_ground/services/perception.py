"""Perception reports from the vehicle: stored detections, finds, thumbnails and events.

Reports wait in a short queue per vehicle and are handled by their own task, so the
link's inbound loop (acks, events, snapshots) never waits for the database or the disk;
when the queue is full the oldest report is dropped and counted.

Every detection in a ``PerceptionReport`` is stored (append-only) if its class is in the
class register; anything else is dropped and counted. The table decides what is new: a
resent detection is neither counted nor published again. Detections of the same class
within ``FIND_RADIUS_M`` of an existing find join it, otherwise they start a new find
(the first detection's id, stored on every row), so the panel and the AI commander see
one find per object instead of one per still. A find's position is the mean of its
detections, its thumbnail the best-scoring one with a thumbnail, and its source a real
detector's when one saw it (``oracle`` only when nothing else did). After a restart the
finds are rebuilt from the table, grouped by the stored find id.

A target's thumbnail is saved in the mission folder (``crops/<detection_id>.jpg``,
registered as a mission file) after checking that it is a small JPEG, once the row is
stored, and never over an existing file.

Events on the bus (topic ``detections``): ``detection.new`` for every stored detection,
``find.new`` for a new find and ``find.updated`` when a find gets another detection.
The commander API passes ``find.new`` on to the AI as ``event.detection``. Nothing here
steers the vehicle, and a find is never confirmed here: only a person confirms.
"""

from __future__ import annotations

import asyncio
import base64
import binascii
import hashlib
import math
from collections import deque
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any

from sd_detect.classes import default_registry
from sd_ground.logging import get_logger
from sd_schemas.geo import tangent_radii
from sd_schemas.link import PerceptionReport

if TYPE_CHECKING:
    from sd_ground.services.missions import MissionFlow

__all__ = [
    "CROP_MAX_BYTES",
    "FIND_RADIUS_M",
    "MAX_CACHED_MISSIONS",
    "REPORT_QUEUE_MAX",
    "Find",
    "Finds",
    "ReportQueue",
    "enqueue",
    "finds_for",
    "on_report",
]

log = get_logger("sd_ground.perception")

FIND_RADIUS_M = 6.0  # detections of one class this close belong to the same find
CROP_MAX_BYTES = 9_000
MAX_CACHED_MISSIONS = 4  # missions whose finds stay in memory (the open one and a few read)
REPORT_QUEUE_MAX = 32  # reports waiting per vehicle; the oldest goes first when full
_JPEG_MAGIC = b"\xff\xd8\xff"
_ORACLE = "oracle"


@dataclass
class Find:
    find_id: str
    label: str
    name_sv: str
    kind: str
    source: str
    lat: float
    lon: float
    sigma95_m: float
    score: float
    count: int = 1
    thumb_detection_id: str | None = None
    synthetic: bool = False
    thumb_score: float = -1.0  # the score of the thumbnail's detection (not published)

    def as_dict(self) -> dict[str, Any]:
        return {
            "find_id": self.find_id,
            "label": self.label,
            "name_sv": self.name_sv,
            "kind": self.kind,
            "source": self.source,
            "lat": round(self.lat, 7),
            "lon": round(self.lon, 7),
            "sigma95_m": round(self.sigma95_m, 1),
            "score": round(self.score, 3),
            "count": self.count,
            "thumb_detection_id": self.thumb_detection_id,
            "synthetic": self.synthetic,
        }


def _distance_m(lat1: float, lon1: float, lat2: float, lon2: float) -> float:
    east_r, north_r = tangent_radii((lat1 + lat2) / 2)
    return math.hypot(math.radians(lon2 - lon1) * east_r, math.radians(lat2 - lat1) * north_r)


@dataclass
class Finds:
    """The finds of one mission (kept in memory, rebuilt from the table after a restart)."""

    items: list[Find] = field(default_factory=list[Find])

    def add(self, row: dict[str, Any]) -> tuple[Find, bool]:
        """Put a detection row into its find; True when the find is new. Sets
        ``row["find_id"]``."""
        best: Find | None = None
        best_d = FIND_RADIUS_M
        for f in self.items:
            if f.label != row["label"]:
                continue
            d = _distance_m(f.lat, f.lon, row["lat"], row["lon"])
            if d <= best_d:
                best, best_d = f, d
        has_thumb = bool(row.get("crop_path"))
        if best is None:
            spec = default_registry().get(row["label"])
            best = Find(
                find_id=str(row["id"]),
                label=row["label"],
                name_sv=spec.name_sv,
                kind=spec.kind,
                source=row["source"],
                lat=row["lat"],
                lon=row["lon"],
                sigma95_m=row["sigma95_m"],
                score=row["score"],
                thumb_detection_id=str(row["id"]) if has_thumb else None,
                synthetic=bool(row["synthetic"]),
                thumb_score=row["score"] if has_thumb else -1.0,
            )
            self.items.append(best)
            row["find_id"] = best.find_id
            return best, True
        n = best.count
        best.lat = (best.lat * n + row["lat"]) / (n + 1)
        best.lon = (best.lon * n + row["lon"]) / (n + 1)
        best.sigma95_m = min(best.sigma95_m, row["sigma95_m"])
        best.score = max(best.score, row["score"])
        if has_thumb and row["score"] > best.thumb_score:
            best.thumb_detection_id, best.thumb_score = str(row["id"]), row["score"]
        if best.source == _ORACLE and row["source"] != _ORACLE:
            best.source = row["source"]
        best.count = n + 1
        row["find_id"] = best.find_id
        return best, False

    @classmethod
    def from_groups(cls, groups: list[dict[str, Any]]) -> Finds:
        """The finds as the table holds them (``FlowRepo.find_groups``)."""
        reg = default_registry()
        items: list[Find] = []
        for g in groups:
            if g["label"] not in reg:
                continue
            spec = reg.get(g["label"])
            items.append(
                Find(
                    find_id=g["find_id"],
                    label=g["label"],
                    name_sv=spec.name_sv,
                    kind=spec.kind,
                    source=g["source"],
                    lat=float(g["lat"]),
                    lon=float(g["lon"]),
                    sigma95_m=float(g["sigma95_m"]),
                    score=float(g["score"]),
                    count=int(g["count"]),
                    thumb_detection_id=g["thumb_detection_id"],
                    synthetic=bool(g["synthetic"]),
                    thumb_score=-1.0 if g["thumb_score"] is None else float(g["thumb_score"]),
                )
            )
        return cls(items)


def _crop(b64: str | None) -> bytes | None:
    if b64 is None:
        return None
    try:
        data = base64.b64decode(b64, validate=True)
    except (binascii.Error, ValueError):
        return None
    if len(data) > CROP_MAX_BYTES or not data.startswith(_JPEG_MAGIC):
        return None
    return data


async def finds_for(flow: MissionFlow, mission_id: str) -> Finds:
    cache: dict[str, Finds] = flow.finds
    finds = cache.get(mission_id)
    if finds is None:
        finds = Finds.from_groups(await flow.repo.find_groups(mission_id))
        while len(cache) >= MAX_CACHED_MISSIONS:
            cache.pop(next(iter(cache)))  # the oldest cached mission
        cache[mission_id] = finds
    return finds


# --- the queue between the link and the database ----------------------------------------


@dataclass
class ReportQueue:
    items: deque[PerceptionReport] = field(
        default_factory=lambda: deque[PerceptionReport](maxlen=REPORT_QUEUE_MAX)
    )
    running: bool = False
    dropped: int = 0
    handled: int = 0  # reports taken off the queue (stored, or failed and logged)


def enqueue(flow: MissionFlow, vehicle_id: str, report: PerceptionReport) -> None:
    """Called from the link's inbound loop: queue the report and return at once."""
    q: ReportQueue = flow.reports.setdefault(vehicle_id, ReportQueue())
    if len(q.items) == q.items.maxlen:
        q.dropped += 1
        log.warning("perception.report_dropped", vehicle_id=vehicle_id, dropped=q.dropped)
    q.items.append(report)
    if not q.running:
        q.running = True
        flow.spawn(_drain(flow, vehicle_id, q), f"perception-{vehicle_id}")


async def _drain(flow: MissionFlow, vehicle_id: str, q: ReportQueue) -> None:
    try:
        while q.items:
            report = q.items.popleft()
            try:
                await on_report(flow, vehicle_id, report)
            except asyncio.CancelledError:
                raise
            except Exception as exc:
                log.error(
                    "perception.report_failed",
                    vehicle_id=vehicle_id,
                    still_id=str(report.still_id),
                    error=type(exc).__name__,
                    detail=str(exc)[:300],
                )
            q.handled += 1
    finally:
        q.running = False


def _save_crop(flow: MissionFlow, mid: str, rel: str, data: bytes) -> bool:
    folder = flow.mission_dir(mid) / "crops"
    folder.mkdir(parents=True, exist_ok=True)
    try:
        with (flow.mission_dir(mid) / rel).open("xb") as f:  # never over an existing file
            f.write(data)
    except FileExistsError:
        return False
    return True


async def on_report(flow: MissionFlow, vehicle_id: str, report: PerceptionReport) -> None:
    mission = await flow.repo.open_mission(vehicle_id)
    if mission is None:
        return  # nothing to search for: no mission is open
    mid = str(mission.id)
    reg = default_registry()
    finds = await finds_for(flow, mid)
    stored = await flow.repo.stored_detection_ids([str(d.detection_id) for d in report.detections])
    rows: list[dict[str, Any]] = []
    crops: dict[str, bytes] = {}
    dropped = 0
    for d in report.detections:
        did = str(d.detection_id)
        if did in stored:
            continue  # a resent report: already stored and counted
        if d.label not in reg:
            dropped += 1
            continue
        data = _crop(d.crop_jpeg_b64)
        if data is not None:
            crops[did] = data
        rows.append(
            {
                "id": did,
                "mission_id": mid,
                "vehicle_id": vehicle_id,
                "find_id": "",
                "still_id": str(report.still_id),
                "captured_at": report.captured_utc,
                "label": d.label,
                "kind": reg.get(d.label).kind,
                "source": d.source,
                "score": d.score,
                "bbox": list(d.bbox),
                "lat": d.lat,
                "lon": d.lon,
                "sigma95_m": d.sigma95_m,
                "size_m": d.size_m,
                "synthetic": report.synthetic,
                "crop_path": f"crops/{did}.jpg" if did in crops else None,
            }
        )
    changes = [finds.add(r) for r in rows]
    try:
        inserted = await flow.repo.add_detections(rows)
    except BaseException:
        flow.finds.pop(mid, None)  # the cache ran ahead of the table: rebuild it next time
        raise
    if len(inserted) != len(rows):
        flow.finds.pop(mid, None)  # someone stored some of them first: rebuild from the table
    if dropped:
        flow.g.add_event(
            "perception.unknown_class",
            severity="warning",
            mission_id=mid,
            vehicle_id=vehicle_id,
            payload={"dropped": dropped, "still_id": str(report.still_id)},
        )
    for row, (find, new) in zip(rows, changes, strict=True):
        if row["id"] not in inserted:
            continue
        rel = row["crop_path"]
        if rel is not None:
            data = crops[row["id"]]
            if _save_crop(flow, mid, rel, data):
                await flow.repo.add_file(
                    mission_id=mid,
                    kind="crop",
                    rel_path=rel,
                    size=len(data),
                    sha256=hashlib.sha256(data).hexdigest(),
                    started_at=report.captured_utc,
                    ended_at=report.captured_utc,
                    notes=None,
                )
        flow.publish(
            "detections",
            "detection.new",
            {
                k: (v.isoformat() if k == "captured_at" else v)
                for k, v in row.items()
                if k not in ("mission_id", "crop_path")
            }
            | {"has_crop": rel is not None},
            vehicle_id=vehicle_id,
            mission_id=mid,
        )
        flow.publish(
            "detections",
            "find.new" if new else "find.updated",
            find.as_dict(),
            vehicle_id=vehicle_id,
            mission_id=mid,
        )
        if new:
            flow.g.add_event(
                "perception.find",
                severity="info",
                mission_id=mid,
                vehicle_id=vehicle_id,
                payload=find.as_dict(),
            )
