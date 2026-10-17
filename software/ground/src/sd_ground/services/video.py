"""The video server MediaMTX seen from the ground station: health and recordings.

* :class:`VideoMonitor` asks MediaMTX's API (``[video] mediamtx_api``, loopback only)
  every few seconds whether the vehicle's stream ``drone/<vehicle_id>/main`` is live and
  sets the part ``mediamtx``: ``ok`` (stream ready), ``degraded`` (server up, no
  stream) or ``down`` (no answer, or video switched off with ``[video] enabled``).
* :func:`close_video` runs when a mission closes. It asks MediaMTX for the recorded
  segments (``GET /v3/recordings/get/drone/<vehicle_id>/main``), picks those that overlap
  the mission (activation to landing), and puts them into ``missions/<id>/video/``:
  finished segments as hard links (a copy, with the event ``video.copied``, when the
  link fails), the segment still being written as a copy of exactly the bytes there at
  that moment (``notes = 'partial'``). Every file gets a ``mission_files`` row with its
  SHA-256, and ``video/index.json`` lists them.

A segment's file name comes from its start time in MediaMTX's ``recordPath``
(``%Y-%m-%d_%H-%M-%S-%f`` in UTC, the container runs with ``TZ=UTC``); when no file
has exactly that name, the one whose name is closest within 1 s is used.
"""

from __future__ import annotations

import asyncio
import datetime as dt
import json
import os
import re
import shutil
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING, Any

import httpx

from sd_ground.logging import get_logger
from sd_ground.sources import PartState

if TYPE_CHECKING:
    from sd_ground.context import Ground
    from sd_ground.services.missions import MissionFlow
    from sd_schemas.api import Mission

__all__ = [
    "FINISHED_AFTER_S",
    "MediaMtx",
    "Planned",
    "Segment",
    "VideoMonitor",
    "close_video",
    "plan_segments",
    "recordings_index",
    "segment_file",
    "select_segments",
    "stream_path",
]

log = get_logger("sd_ground.video")
POLL_S = 2.0
TIMEOUT_S = 3.0
FINISHED_AFTER_S = 5.0  # a segment file untouched this long is finished
NAME_SLACK_S = 1.0
NAME_FORMAT = "%Y-%m-%d_%H-%M-%S-%f"
_NAME = re.compile(r"^(\d{4}-\d{2}-\d{2}_\d{2}-\d{2}-\d{2}-\d{6})\.mp4$")
_COPY_CHUNK = 1 << 20
API_USER = "ground"
API_PASSWORD_KEY = "SD_GROUND__VIDEO_API_PASSWORD"  # noqa: S105 - a variable name


def stream_path(vehicle_id: str) -> str:
    return f"drone/{vehicle_id}/main"


@dataclass(frozen=True)
class Segment:
    start: dt.datetime
    end: dt.datetime | None  # the next segment's start; None for the last one


class MediaMtx:
    """MediaMTX's control API on loopback (never through an HTTP proxy).

    The API needs the user ``ground`` with ``SD_GROUND__VIDEO_API_PASSWORD``: MediaMTX's
    API can change its configuration at runtime, so no password-less user may reach it.
    """

    def __init__(
        self,
        api_url: str,
        password: str = "",
        transport: httpx.AsyncBaseTransport | None = None,
    ) -> None:
        self.api_url = api_url.rstrip("/")
        self._auth = httpx.BasicAuth(API_USER, password)
        self._transport = transport

    @classmethod
    def of(cls, g: Ground) -> MediaMtx:
        return cls(g.config.video.mediamtx_api, g.loaded.secret(API_PASSWORD_KEY))

    def _client(self) -> httpx.AsyncClient:
        return httpx.AsyncClient(
            base_url=self.api_url,
            timeout=TIMEOUT_S,
            trust_env=False,
            auth=self._auth,
            transport=self._transport,
        )

    async def stream_state(self, vehicle_id: str) -> PartState:
        try:
            async with self._client() as client:
                res = await client.get(f"/v3/paths/get/{stream_path(vehicle_id)}")
            body = res.json() if res.status_code == 200 else {}
        except (httpx.HTTPError, ValueError):
            return "down"
        if res.status_code == 200 and isinstance(body, dict) and body.get("ready") is True:
            return "ok"
        return "degraded" if res.status_code in (200, 404) else "down"

    async def segment_starts(self, vehicle_id: str) -> list[dt.datetime]:
        """Start times (UTC) of the recorded segments, oldest first; [] when none.

        Raises ``httpx.HTTPError`` or ``ValueError`` when the answer is not usable."""
        async with self._client() as client:
            res = await client.get(f"/v3/recordings/get/{stream_path(vehicle_id)}")
        if res.status_code == 404:
            return []
        res.raise_for_status()
        body = res.json()
        segments = body.get("segments") if isinstance(body, dict) else None
        if not isinstance(segments, list):
            raise ValueError("oväntat svar från MediaMTX")
        starts: list[dt.datetime] = []
        for seg in segments:
            if not isinstance(seg, dict) or "start" not in seg:
                raise ValueError("oväntat segment från MediaMTX")
            starts.append(dt.datetime.fromisoformat(str(seg["start"])).astimezone(dt.UTC))
        return sorted(starts)


def select_segments(
    starts: list[dt.datetime], start_at: dt.datetime, end_at: dt.datetime
) -> list[Segment]:
    """The segments that begin before ``end_at`` and overlap ``[start_at, end_at]``."""
    out: list[Segment] = []
    for i, start in enumerate(starts):
        end = starts[i + 1] if i + 1 < len(starts) else None
        if start < end_at and (end is None or end > start_at):
            out.append(Segment(start, end))
    return out


def recordings_index(folder: Path) -> dict[dt.datetime, Path]:
    """Every recording in ``folder`` by the start time in its name (read once per close)."""
    found: dict[dt.datetime, Path] = {}
    if not folder.is_dir():
        return found
    for path in folder.iterdir():
        match = _NAME.match(path.name)
        if match is not None:
            when = dt.datetime.strptime(match.group(1), NAME_FORMAT).replace(tzinfo=dt.UTC)
            found[when] = path
    return found


def segment_file(index: dict[dt.datetime, Path], start: dt.datetime) -> Path | None:
    """The recording that starts at ``start``: exact name, else the closest within 1 s."""
    start = start.astimezone(dt.UTC)
    if start in index:
        return index[start]
    best: tuple[float, Path] | None = None
    for when, path in index.items():
        gap = abs((when - start).total_seconds())
        if gap <= NAME_SLACK_S and (best is None or gap < best[0]):
            best = (gap, path)
    return None if best is None else best[1]


@dataclass(frozen=True)
class Planned:
    segment: Segment
    source: Path
    finished: bool


def plan_segments(
    folder: Path, segments: list[Segment], start_at: dt.datetime, now: dt.datetime
) -> tuple[list[Planned], list[Segment]]:
    """Find each segment's file (runs in a worker thread: it reads the folder).

    The last segment has no next start, so its real end is the file's last write: a
    stream that stopped before the mission began is not part of it. Returns the files
    to place and the segments whose file is missing."""
    index = recordings_index(folder)
    planned: list[Planned] = []
    missing: list[Segment] = []
    for seg in segments:
        src = segment_file(index, seg.start)
        if src is None:
            missing.append(seg)
            continue
        written = dt.datetime.fromtimestamp(src.stat().st_mtime, tz=dt.UTC)
        if seg.end is None and written < start_at:
            continue
        finished = seg.end is not None or (now - written).total_seconds() > FINISHED_AFTER_S
        planned.append(Planned(seg, src, finished))
    return planned, missing


def _copy_bytes(src: Path, dst: Path, size: int) -> None:
    """Copy exactly ``size`` bytes (the file may still grow while this runs)."""
    left = size
    with src.open("rb") as fin, dst.open("wb") as fout:
        while left > 0:
            chunk = fin.read(min(_COPY_CHUNK, left))
            if not chunk:
                break
            fout.write(chunk)
            left -= len(chunk)


def _place(src: Path, dst: Path, *, finished: bool) -> tuple[str | None, str | None]:
    """Link or copy one segment. Returns (notes, warning event type or None)."""
    dst.parent.mkdir(parents=True, exist_ok=True)
    if dst.exists():
        dst.unlink()
    if not finished:
        _copy_bytes(src, dst, src.stat().st_size)
        return "partial", None
    try:
        os.link(src, dst)
        return None, None
    except OSError:
        shutil.copyfile(src, dst)
        return None, "video.copied"


class VideoMonitor:
    """Keeps ``g.video_state`` up to date (the part ``mediamtx`` in ``/health``)."""

    def __init__(self, g: Ground, mtx: MediaMtx | None = None) -> None:
        self.g = g
        self.mtx = mtx or MediaMtx.of(g)

    async def tick(self) -> PartState:
        g = self.g
        state: PartState = "down"
        if g.config.video.enabled:
            state = await self.mtx.stream_state(g.config.default_vehicle_id)
        if state != g.video_state:
            log.info("video.state", state=state)
        g.video_state = state
        return state

    async def run(self) -> None:
        while True:
            try:
                await self.tick()
            except Exception as exc:  # the monitor never stops; the state says down
                log.warning("video.monitor_failed", error=type(exc).__name__)
                self.g.video_state = "down"
            await self.g.clock.sleep(POLL_S)


async def close_video(flow: MissionFlow, mission: Mission, mtx: MediaMtx | None = None) -> None:
    """Put the mission's video segments into its folder and record them.

    Never raises: a failure here (a full disk, an odd answer from MediaMTX) is logged
    and recorded as ``video.archive_failed``, and the mission still closes."""
    try:
        await _close_video(flow, mission, mtx)
    except Exception as exc:
        log.error("video.archive_failed", mission_id=str(mission.id), error=type(exc).__name__)
        flow.g.add_event(
            "video.archive_failed",
            severity="critical",
            mission_id=str(mission.id),
            vehicle_id=mission.vehicle_id,
            payload={"error": type(exc).__name__},
        )


async def _close_video(flow: MissionFlow, mission: Mission, mtx: MediaMtx | None) -> None:
    from sd_ground.services.files import sha256_of

    g = flow.g
    if not g.config.video.enabled:
        return
    mid, vid = str(mission.id), mission.vehicle_id
    start_at = mission.activated_at or mission.created_at
    end_at = mission.landed_at or flow.now()
    mtx = mtx or MediaMtx.of(g)
    try:
        starts = await mtx.segment_starts(vid)
    except (httpx.HTTPError, ValueError) as exc:
        log.warning("video.recordings_unavailable", mission_id=mid, error=type(exc).__name__)
        g.add_event("video.unavailable", severity="warning", mission_id=mid, vehicle_id=vid)
        return
    source = g.loaded.data_dir / "video" / stream_path(vid)
    target = flow.mission_dir(mid) / "video"
    segments = select_segments(starts, start_at, end_at)
    planned, missing = await asyncio.to_thread(
        plan_segments, source, segments, start_at, flow.now()
    )
    for seg in missing:
        g.add_event(
            "video.segment_missing",
            severity="warning",
            mission_id=mid,
            vehicle_id=vid,
            payload={"start": seg.start.isoformat()},
        )
    index: list[dict[str, Any]] = []
    for item in planned:
        seg, src = item.segment, item.source
        dst = target / src.name
        try:
            notes, warning = await asyncio.to_thread(_place, src, dst, finished=item.finished)
        except OSError as exc:
            log.error("video.place_failed", mission_id=mid, error=type(exc).__name__)
            g.add_event("files.write_failed", severity="critical", mission_id=mid)
            continue
        if warning is not None:
            g.add_event(warning, severity="warning", mission_id=mid, payload={"file": src.name})
        size, digest = await asyncio.to_thread(sha256_of, dst)
        await flow.repo.add_file(
            mission_id=mid,
            kind="video",
            rel_path=f"video/{src.name}",
            size=size,
            sha256=digest,
            started_at=seg.start,
            ended_at=seg.end or end_at,
            notes=notes,
        )
        index.append(
            {
                "file": src.name,
                "start": seg.start.isoformat(),
                "end": None if seg.end is None else seg.end.isoformat(),
                "partial": notes == "partial",
                "sha256": digest,
            }
        )
    if not index:
        return
    body = json.dumps({"vehicle_id": vid, "segments": index}, indent=2)
    path = target / "index.json"
    await asyncio.to_thread(path.write_text, body, "utf-8")
    size, digest = await asyncio.to_thread(sha256_of, path)
    await flow.repo.add_file(
        mission_id=mid,
        kind="video",
        rel_path="video/index.json",
        size=size,
        sha256=digest,
        started_at=start_at,
        ended_at=end_at,
        notes="index",
    )
