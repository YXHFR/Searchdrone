"""Perception reports from the vehicle: stored detections, finds, thumbnails and events.

Every detection in a ``PerceptionReport`` is stored (append-only) if its class is in the
class register; anything else is dropped and counted. Detections of the same class
within ``FIND_RADIUS_M`` of an existing find join it, otherwise they start a new find
(the first detection's id), so the panel and the AI commander see one find per object
instead of one per still. A find's position is the mean of its detections.

A target's thumbnail is saved in the mission folder (``crops/<detection_id>.jpg``,
registered as a mission file) after checking that it is a small JPEG.

Events on the bus (topic ``detections``): ``detection.new`` for every stored detection,
``find.new`` for a new find and ``find.updated`` when a find gets another detection.
The commander API passes ``find.new`` on to the AI as ``event.detection``. Nothing here
steers the vehicle, and a find is never confirmed here: only a person confirms.
"""

from __future__ import annotations

import base64
import binascii
import hashlib
import math
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any

from sd_detect.classes import default_registry
from sd_schemas.geo import tangent_radii
from sd_schemas.link import PerceptionReport

if TYPE_CHECKING:
    from sd_ground.services.missions import MissionFlow

__all__ = ["CROP_MAX_BYTES", "FIND_RADIUS_M", "Find", "Finds", "on_report"]

FIND_RADIUS_M = 6.0  # detections of one class this close belong to the same find
CROP_MAX_BYTES = 9_000
_JPEG_MAGIC = b"\xff\xd8\xff"


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
    seen: set[str] = field(default_factory=set[str])  # detection ids already counted

    def add(self, row: dict[str, Any]) -> tuple[Find, bool]:
        """Put a detection row into its find; True when the find is new. Sets
        ``row["find_id"]``."""
        self.seen.add(str(row["id"]))
        best: Find | None = None
        best_d = FIND_RADIUS_M
        for f in self.items:
            if f.label != row["label"]:
                continue
            d = _distance_m(f.lat, f.lon, row["lat"], row["lon"])
            if d <= best_d:
                best, best_d = f, d
        has_thumb = bool(row.get("crop_path") or row.get("has_crop"))
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
            )
            self.items.append(best)
            row["find_id"] = best.find_id
            return best, True
        n = best.count
        best.lat = (best.lat * n + row["lat"]) / (n + 1)
        best.lon = (best.lon * n + row["lon"]) / (n + 1)
        best.sigma95_m = min(best.sigma95_m, row["sigma95_m"])
        if row["score"] > best.score or (best.thumb_detection_id is None and has_thumb):
            best.score = max(best.score, row["score"])
            if has_thumb:
                best.thumb_detection_id = str(row["id"])
        best.count = n + 1
        row["find_id"] = best.find_id
        return best, False


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
        finds = Finds()
        for row in reversed(await flow.repo.detections(mission_id)):
            finds.add(dict(row, id=row["id"]))
        cache[mission_id] = finds
    return finds


async def on_report(flow: MissionFlow, vehicle_id: str, report: PerceptionReport) -> None:
    mission = await flow.repo.open_mission(vehicle_id)
    if mission is None:
        return  # nothing to search for: no mission is open
    mid = str(mission.id)
    reg = default_registry()
    finds = await finds_for(flow, mid)
    rows: list[dict[str, Any]] = []
    dropped = 0
    folder = flow.mission_dir(mid) / "crops"
    for d in report.detections:
        if str(d.detection_id) in finds.seen:
            continue  # a resent report: already stored and counted
        if d.label not in reg:
            dropped += 1
            continue
        spec = reg.get(d.label)
        crop_path: str | None = None
        data = _crop(d.crop_jpeg_b64)
        if data is not None:
            folder.mkdir(parents=True, exist_ok=True)
            rel = f"crops/{d.detection_id}.jpg"
            (flow.mission_dir(mid) / rel).write_bytes(data)
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
            crop_path = rel
        rows.append(
            {
                "id": str(d.detection_id),
                "mission_id": mid,
                "vehicle_id": vehicle_id,
                "find_id": "",
                "still_id": str(report.still_id),
                "captured_at": report.captured_utc,
                "label": d.label,
                "kind": spec.kind,
                "source": d.source,
                "score": d.score,
                "bbox": list(d.bbox),
                "lat": d.lat,
                "lon": d.lon,
                "sigma95_m": d.sigma95_m,
                "size_m": d.size_m,
                "synthetic": report.synthetic,
                "crop_path": crop_path,
            }
        )
    changes = [finds.add(r) for r in rows]
    await flow.repo.add_detections(rows)
    if dropped:
        flow.g.add_event(
            "perception.unknown_class",
            severity="warning",
            mission_id=mid,
            vehicle_id=vehicle_id,
            payload={"dropped": dropped, "still_id": str(report.still_id)},
        )
    for row, (find, new) in zip(rows, changes, strict=True):
        flow.publish(
            "detections",
            "detection.new",
            {
                k: (v.isoformat() if k == "captured_at" else v)
                for k, v in row.items()
                if k not in ("mission_id", "crop_path")
            }
            | {"has_crop": row["crop_path"] is not None},
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
