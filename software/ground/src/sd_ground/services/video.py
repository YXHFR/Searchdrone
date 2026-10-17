"""Video: the state of MediaMTX and the recording that belongs to each mission.

MediaMTX (compose service ``mediamtx``) records every ``drone/<vehicle_id>/main`` path in
60 s fMP4 segments under ``<data_dir>/video/``. When a mission closes, the segments that
overlap ``[activated_at, landed_at]`` are taken into ``missions/<id>/video/``:

* a **finished** segment (a later one exists, or the file has not changed for 5 s) is
  hard-linked (copied if the link fails, with the warning event ``video.copied``);
* the segment still being written is **copied**, exactly the bytes there were when it
  was measured, and its row gets ``notes = 'partial'``.

Each file gets a ``mission_files`` row (``kind = 'video'``) with the SHA-256 of the file
in the mission folder, and ``video/index.json`` lists the segments. MediaMTX is only ever
reached on ``127.0.0.1`` (its API on :9997), never through a proxy from the environment.
"""

from __future__ import annotations

import asyncio
import datetime as dt
import json
import os
import shutil
from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING, Any

import httpx

from sd_ground.logging import get_logger
from sd_ground.sources import PartState
from sd_schemas.api import Mission

if TYPE_CHECKING:
    from sd_ground.context import Ground
    from sd_ground.services.missions import MissionFlow

__all__ = [
    "FINISHED_AFTER_S",
    "NAME_FORMAT",
    "Segment",
    "VideoMonitor",
    "close_mission_video",
    "path_name",
    "plan_segments",
    "segment_file",
]

log = get_logger("sd_ground.video")
NAME_FORMAT = "%Y-%m-%d_%H-%M-%S-%f"  # recordPath in config/mediamtx.yml
FINISHED_AFTER_S = 5.0
MATCH_WITHIN_S = 1.0
POLL_S = 2.0
_TIMEOUT = httpx.Timeout(3.0)


def path_name(vehicle_id: str) -> str:
    return f"drone/{vehicle_id}/main"


def client() -> httpx.AsyncClient:
    """A client for MediaMTX on loopback; ``trust_env=False`` keeps any proxy out."""
    return httpx.AsyncClient(timeout=_TIMEOUT, trust_env=False)


@dataclass(frozen=True)
class Segment:
    start: dt.datetime
    path: Path
    finished: bool
    size: int  # bytes at the moment it was measured


def _parse_start(value: str) -> dt.datetime:
    start = dt.datetime.fromisoformat(value.replace("Z", "+00:00"))
    return start if start.tzinfo else start.replace(tzinfo=dt.UTC)


def segment_file(folder: Path, start: dt.datetime) -> Path | None:
    """The file MediaMTX wrote for a segment starting at ``start``.

    The exact name from ``recordPath``, else the closest name within ±1 s.
    """
    utc = start.astimezone(dt.UTC)
    exact = folder / f"{utc.strftime(NAME_FORMAT)}.mp4"
    if exact.is_file():
        return exact
    best: tuple[float, Path] | None = None
    for item in folder.glob("*.mp4") if folder.is_dir() else ():
        try:
            when = dt.datetime.strptime(item.stem, NAME_FORMAT).replace(tzinfo=dt.UTC)
        except ValueError:
            continue
        diff = abs((when - utc).total_seconds())
        if diff <= MATCH_WITHIN_S and (best is None or diff < best[0]):
            best = (diff, item)
    return None if best is None else best[1]


def plan_segments(
    starts: Sequence[dt.datetime],
    folder: Path,
    begin: dt.datetime,
    end: dt.datetime,
    now: dt.datetime,
) -> tuple[list[Segment], list[dt.datetime]]:
    """The segments overlapping ``[begin, end]`` and the starts whose file is missing."""
    ordered = sorted(starts)
    chosen: list[Segment] = []
    missing: list[dt.datetime] = []
    for i, start in enumerate(ordered):
        if start >= end:
            continue
        later = ordered[i + 1] if i + 1 < len(ordered) else None
        path = segment_file(folder, start)
        if path is None:
            if later is None or later > begin:
                missing.append(start)
            continue
        st = path.stat()
        mtime = dt.datetime.fromtimestamp(st.st_mtime, dt.UTC)
        seg_end = later if later is not None else max(mtime, start)
        if seg_end <= begin:
            continue
        finished = later is not None or (now - mtime).total_seconds() > FINISHED_AFTER_S
        chosen.append(Segment(start=start, path=path, finished=finished, size=st.st_size))
    return chosen, missing


def _copy_prefix(src: Path, dst: Path, size: int) -> None:
    left = size
    with src.open("rb") as fin, dst.open("wb") as fout:
        while left > 0:
            chunk = fin.read(min(left, 1 << 20))
            if not chunk:
                break
            fout.write(chunk)
            left -= len(chunk)


def _take(seg: Segment, target: Path) -> str:
    """Link or copy one segment into the mission folder: 'linked', 'copied' or 'partial'."""
    target.parent.mkdir(parents=True, exist_ok=True)
    if target.exists():
        target.unlink()
    if not seg.finished:
        _copy_prefix(seg.path, target, seg.size)
        return "partial"
    try:
        os.link(seg.path, target)
        return "linked"
    except OSError:
        shutil.copyfile(seg.path, target)
        return "copied"


async def recording_starts(api: str, vehicle_id: str) -> list[dt.datetime]:
    async with client() as http:
        res = await http.get(f"{api}/v3/recordings/get/{path_name(vehicle_id)}")
    if res.status_code == 404:
        return []
    res.raise_for_status()
    body: dict[str, Any] = res.json()
    return [_parse_start(str(s["start"])) for s in body.get("segments") or []]


async def close_mission_video(flow: MissionFlow, mission: Mission) -> None:
    """Take the mission's recording into its folder and record each file with its hash."""
    from sd_ground.services.files import sha256_of

    g = flow.g
    if not g.config.video.enabled or mission.activated_at is None:
        return
    mid = str(mission.id)
    begin = mission.activated_at
    end = mission.landed_at or flow.now()
    try:
        starts = await recording_starts(g.config.video.mediamtx_api, mission.vehicle_id)
    except (httpx.HTTPError, ValueError, KeyError) as exc:
        log.warning("video.list_failed", mission_id=mid, error=type(exc).__name__)
        g.add_event(
            "video.unavailable", severity="warning", mission_id=mid, payload={"error": "api"}
        )
        return
    source = g.loaded.data_dir / "video" / path_name(mission.vehicle_id)
    segments, missing = await asyncio.to_thread(
        plan_segments, starts, source, begin, end, flow.now()
    )
    for start in missing:
        g.add_event(
            "video.segment_missing",
            severity="warning",
            mission_id=mid,
            payload={"start": start.isoformat()},
        )
    folder = flow.mission_dir(mid) / "video"
    index: list[dict[str, Any]] = []
    for seg in segments:
        rel = f"video/{seg.path.name}"
        try:
            how = await asyncio.to_thread(_take, seg, folder / seg.path.name)
            size, digest = await asyncio.to_thread(sha256_of, folder / seg.path.name)
        except OSError as exc:
            log.error("video.take_failed", mission_id=mid, error=type(exc).__name__)
            g.add_event(
                "video.segment_missing",
                severity="warning",
                mission_id=mid,
                payload={"start": seg.start.isoformat(), "error": type(exc).__name__},
            )
            continue
        if how == "copied":
            g.add_event("video.copied", severity="warning", mission_id=mid, payload={"file": rel})
        notes = "partial" if how == "partial" else None
        await flow.repo.add_file(
            mission_id=mid,
            kind="video",
            rel_path=rel,
            size=size,
            sha256=digest,
            started_at=seg.start,
            ended_at=None,
            notes=notes,
        )
        index.append({"file": seg.path.name, "start": seg.start.isoformat(), "notes": notes})
    if index:
        text = json.dumps({"vehicle_id": mission.vehicle_id, "segments": index}, indent=2)
        await asyncio.to_thread(_write_index, folder / "index.json", text)


def _write_index(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text, encoding="utf-8")


class VideoMonitor:
    """Keeps ``g.video_state`` current: ok = the vehicle's stream is ready in MediaMTX,
    degraded = MediaMTX answers but there is no stream, down = MediaMTX does not answer."""

    def __init__(self, g: Ground) -> None:
        self.g = g

    async def state(self, http: httpx.AsyncClient) -> PartState:
        cfg = self.g.config.video
        url = f"{cfg.mediamtx_api}/v3/paths/get/{path_name(self.g.config.default_vehicle_id)}"
        try:
            res = await http.get(url)
        except httpx.HTTPError:
            return "down"
        if res.status_code == 200 and res.json().get("ready") is True:
            return "ok"
        return "degraded" if res.status_code in (200, 404) else "down"

    async def run(self) -> None:
        if not self.g.config.video.enabled:
            return
        async with client() as http:
            while True:
                new = await self.state(http)
                if new != self.g.video_state:
                    log.info("video.state", state=new)
                    self.g.video_state = new
                await self.g.clock.sleep(POLL_S)
