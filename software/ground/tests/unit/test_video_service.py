"""Video health and the per-mission recordings (04 §15.2; B1-09 point 3 without MediaMTX).

MediaMTX's API is an ``httpx.MockTransport``; the recordings are small files in a
temporary data root, named the way MediaMTX names them.
"""

from __future__ import annotations

import datetime as dt
import json
import os
from collections.abc import Callable
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import httpx
import pytest

from sd_ground.context import Ground
from sd_ground.services import video
from sd_schemas.api import Mission
from sd_schemas.base import new_id
from sd_testing import FakeClock

T0 = dt.datetime(2026, 10, 17, 12, 0, 0, 250000, tzinfo=dt.UTC)


def at(seconds: float) -> dt.datetime:
    return T0 + dt.timedelta(seconds=seconds)


def api(segments: list[dt.datetime] | None, *, ready: bool | None = True) -> httpx.MockTransport:
    def handler(request: httpx.Request) -> httpx.Response:
        if request.url.path == "/v3/recordings/get/drone/sim-01/main":
            if segments is None:
                return httpx.Response(404, json={"error": "not found"})
            body = {
                "name": "drone/sim-01/main",
                "segments": [{"start": s.isoformat()} for s in segments],
            }
            return httpx.Response(200, json=body)
        if request.url.path == "/v3/paths/get/drone/sim-01/main":
            if ready is None:
                return httpx.Response(404, json={"error": "path not found"})
            return httpx.Response(200, json={"ready": ready})
        return httpx.Response(500)

    return httpx.MockTransport(handler)


def down() -> httpx.MockTransport:
    def handler(request: httpx.Request) -> httpx.Response:
        raise httpx.ConnectError("refused", request=request)

    return httpx.MockTransport(handler)


def test_select_segments() -> None:
    starts = [at(-120), at(-60), at(0), at(60), at(120)]
    picked = video.select_segments(starts, at(10), at(70))
    assert [s.start for s in picked] == [at(0), at(60)]
    assert picked[0].end == at(60)
    # The last segment has no end and is picked while it starts before the end.
    last = video.select_segments(starts, at(130), at(140))
    assert [(s.start, s.end) for s in last] == [(at(120), None)]
    assert video.select_segments(starts, at(200), at(210)) == [video.Segment(at(120), None)]
    assert video.select_segments(starts, at(-200), at(-130)) == []


def test_segment_file_exact_and_nearest(tmp_path: Path) -> None:
    exact = tmp_path / "2026-10-17_12-00-00-250000.mp4"
    exact.write_bytes(b"x")
    assert video.segment_file(tmp_path, at(0)) == exact
    near = tmp_path / "2026-10-17_12-01-00-900000.mp4"
    near.write_bytes(b"x")
    assert video.segment_file(tmp_path, at(60)) == near  # 0.65 s off
    assert video.segment_file(tmp_path, at(120)) is None
    assert video.segment_file(tmp_path / "missing", at(0)) is None


@pytest.mark.parametrize(
    ("transport", "state"),
    [
        (api([], ready=True), "ok"),
        (api([], ready=False), "degraded"),
        (api([], ready=None), "degraded"),
        (down(), "down"),
    ],
)
async def test_monitor(ground: Ground, transport: httpx.MockTransport, state: str) -> None:
    monitor = video.VideoMonitor(ground, video.MediaMtx("http://127.0.0.1:9997", transport))
    assert await monitor.tick() == state
    assert ground.video_state == state


async def test_monitor_disabled(make_ground: Callable[..., Ground]) -> None:
    g = make_ground(video={"enabled": False})
    monitor = video.VideoMonitor(g, video.MediaMtx("http://127.0.0.1:9997", api([])))
    assert await monitor.tick() == "down"


async def test_health_reports_the_video_state(ground: Ground) -> None:
    from sd_ground.services import health

    ground.video_state = "ok"
    assert health.parts(ground)["mediamtx"] == "ok"


@dataclass
class FakeRepo:
    files: list[dict[str, Any]] = field(default_factory=list[dict[str, Any]])

    async def add_file(self, **row: Any) -> None:
        self.files.append(row)


@dataclass
class FakeFlow:
    g: Ground
    clock: FakeClock
    repo: FakeRepo = field(default_factory=FakeRepo)

    def now(self) -> dt.datetime:
        return self.clock.utc_now()

    def mission_dir(self, mission_id: str) -> Path:
        return self.g.loaded.data_dir / "missions" / mission_id


def mission(activated: dt.datetime, landed: dt.datetime | None) -> Mission:
    from sd_schemas.mission import MissionState
    from sd_schemas.safety import SafetyLimits

    return Mission(
        id=new_id(),
        vehicle_id="sim-01",
        title="test",
        status=MissionState.LANDED,
        paused_by=None,
        returning_cause=None,
        state_seq=1,
        limits=SafetyLimits(),
        active_charter_id=None,
        autonomy_mode=None,
        ai_enabled=True,
        created_by="operator:demo",
        created_at=activated - dt.timedelta(seconds=30),
        activated_at=activated,
        landed_at=landed,
        closed_at=None,
        closed_reason=None,
        notes=None,
    )


def recordings(g: Ground, starts: list[dt.datetime], size: int = 1000) -> Path:
    folder = g.loaded.data_dir / "video" / "drone" / "sim-01" / "main"
    folder.mkdir(parents=True)
    for i, start in enumerate(starts):
        path = folder / f"{start.strftime(video.NAME_FORMAT)}.mp4"
        path.write_bytes(bytes([i]) * size)
    return folder


async def test_close_links_finished_and_copies_the_open_segment(ground: Ground) -> None:
    clock = FakeClock(start_utc=at(130))
    starts = [at(-60), at(0), at(60), at(120)]
    folder = recordings(ground, starts)
    live = folder / f"{at(120).strftime(video.NAME_FORMAT)}.mp4"
    flow: Any = FakeFlow(ground, clock)
    m = mission(at(10), at(125))
    os.utime(live, (at(129).timestamp(), at(129).timestamp()))  # still being written
    await video.close_video(flow, m, video.MediaMtx("http://127.0.0.1:9997", api(starts)))

    target = flow.mission_dir(str(m.id)) / "video"
    rows = {r["rel_path"]: r for r in flow.repo.files}
    names = [f"{s.strftime(video.NAME_FORMAT)}.mp4" for s in (at(0), at(60), at(120))]
    assert set(rows) == {f"video/{n}" for n in names} | {"video/index.json"}
    for name in names[:2]:  # finished: hard links, same inode
        assert (target / name).stat().st_ino == (folder / name).stat().st_ino
        assert rows[f"video/{name}"]["notes"] is None
    assert (target / names[2]).stat().st_ino != live.stat().st_ino  # open: a copy
    assert rows[f"video/{names[2]}"]["notes"] == "partial"
    assert rows[f"video/{names[0]}"]["started_at"] == at(0)
    assert rows[f"video/{names[0]}"]["ended_at"] == at(60)
    assert all(r["kind"] == "video" and len(r["sha256"]) == 64 for r in rows.values())
    index = json.loads((target / "index.json").read_text())
    assert [s["file"] for s in index["segments"]] == names
    assert index["segments"][2]["partial"] is True


async def test_partial_copy_takes_the_size_seen_once(ground: Ground, tmp_path: Path) -> None:
    src = tmp_path / "a.mp4"
    src.write_bytes(b"0123456789")
    dst = tmp_path / "b.mp4"
    video._copy_bytes(src, dst, 4)  # pyright: ignore[reportPrivateUsage]
    assert dst.read_bytes() == b"0123"


async def test_old_last_segment_is_finished(ground: Ground) -> None:
    clock = FakeClock(start_utc=at(200))
    starts = [at(0)]
    folder = recordings(ground, starts)
    os.utime(
        folder / f"{at(0).strftime(video.NAME_FORMAT)}.mp4",
        (at(100).timestamp(), at(100).timestamp()),
    )
    flow: Any = FakeFlow(ground, clock)
    await video.close_video(
        flow, mission(at(10), at(90)), video.MediaMtx("http://x:1", api(starts))
    )
    (row,) = [r for r in flow.repo.files if r["rel_path"].endswith(".mp4")]
    assert row["notes"] is None


async def test_missing_file_and_no_server(ground: Ground) -> None:
    clock = FakeClock(start_utc=at(100))
    flow: Any = FakeFlow(ground, clock)
    await video.close_video(
        flow, mission(at(10), at(90)), video.MediaMtx("http://x:1", api([at(0)]))
    )
    assert flow.repo.files == []
    assert [e["type"] for e in ground.queued_events()] == ["video.segment_missing"]
    await video.close_video(flow, mission(at(10), at(90)), video.MediaMtx("http://x:1", down()))
    assert ground.queued_events()[-1]["type"] == "video.unavailable"
    await video.close_video(flow, mission(at(10), at(90)), video.MediaMtx("http://x:1", api(None)))
    assert flow.repo.files == []


async def test_close_does_nothing_when_disabled(
    make_ground: Callable[..., Ground], clock: FakeClock
) -> None:
    g = make_ground(video={"enabled": False})
    flow: Any = FakeFlow(g, clock)
    await video.close_video(flow, mission(at(10), at(90)), video.MediaMtx("http://x:1", down()))
    assert g.queued_events() == []
