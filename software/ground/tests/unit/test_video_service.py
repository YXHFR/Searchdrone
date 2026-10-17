"""Video health and the per-mission recordings (B1-09, without MediaMTX).

MediaMTX's API is an ``httpx.MockTransport``; the recordings are small files in a
temporary data root, named the way MediaMTX names them.
"""

from __future__ import annotations

import asyncio
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
    near = tmp_path / "2026-10-17_12-01-00-900000.mp4"
    near.write_bytes(b"x")
    (tmp_path / "notes.txt").write_text("x")
    index = video.recordings_index(tmp_path)
    assert len(index) == 2
    assert video.segment_file(index, at(0)) == exact
    assert video.segment_file(index, at(60)) == near  # 0.65 s off
    assert video.segment_file(index, at(120)) is None
    assert video.recordings_index(tmp_path / "missing") == {}


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
    monitor = video.VideoMonitor(ground, video.MediaMtx("http://127.0.0.1:9997", "pw", transport))
    assert await monitor.tick() == state
    assert ground.video_state == state


async def test_monitor_disabled(make_ground: Callable[..., Ground]) -> None:
    g = make_ground(video={"enabled": False})
    monitor = video.VideoMonitor(g, video.MediaMtx("http://127.0.0.1:9997", "pw", api([])))
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
    await video.close_video(flow, m, video.MediaMtx("http://127.0.0.1:9997", "pw", api(starts)))

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
        flow, mission(at(10), at(90)), video.MediaMtx("http://x:1", "pw", api(starts))
    )
    (row,) = [r for r in flow.repo.files if r["rel_path"].endswith(".mp4")]
    assert row["notes"] is None


async def test_missing_file_and_no_server(ground: Ground) -> None:
    clock = FakeClock(start_utc=at(100))
    flow: Any = FakeFlow(ground, clock)
    await video.close_video(
        flow, mission(at(10), at(90)), video.MediaMtx("http://x:1", "pw", api([at(0)]))
    )
    assert flow.repo.files == []
    assert [e["type"] for e in ground.queued_events()] == ["video.segment_missing"]
    await video.close_video(
        flow, mission(at(10), at(90)), video.MediaMtx("http://x:1", "pw", down())
    )
    assert ground.queued_events()[-1]["type"] == "video.unavailable"
    await video.close_video(
        flow, mission(at(10), at(90)), video.MediaMtx("http://x:1", "pw", api(None))
    )
    assert flow.repo.files == []


async def test_close_does_nothing_when_disabled(
    make_ground: Callable[..., Ground], clock: FakeClock
) -> None:
    g = make_ground(video={"enabled": False})
    flow: Any = FakeFlow(g, clock)
    await video.close_video(
        flow, mission(at(10), at(90)), video.MediaMtx("http://x:1", "pw", down())
    )
    assert g.queued_events() == []


async def test_a_stream_that_stopped_before_the_mission_is_not_part_of_it(
    ground: Ground,
) -> None:
    """The last segment has no next start; its file's last write bounds it."""
    clock = FakeClock(start_utc=at(400))
    starts = [at(0)]
    folder = recordings(ground, starts)
    old = folder / f"{at(0).strftime(video.NAME_FORMAT)}.mp4"
    os.utime(old, (at(50).timestamp(), at(50).timestamp()))  # the stream stopped at 50 s
    flow: Any = FakeFlow(ground, clock)
    await video.close_video(
        flow, mission(at(200), at(300)), video.MediaMtx("http://x:1", "pw", api(starts))
    )
    assert flow.repo.files == []
    # A mission that overlaps the recording still gets it.
    await video.close_video(
        flow, mission(at(10), at(40)), video.MediaMtx("http://x:1", "pw", api(starts))
    )
    assert flow.repo.files[0]["rel_path"].endswith(".mp4")


async def test_api_uses_its_own_user() -> None:
    seen: list[httpx.Request] = []

    def handler(request: httpx.Request) -> httpx.Response:
        seen.append(request)
        return httpx.Response(200, json={"ready": True})

    mtx = video.MediaMtx("http://127.0.0.1:9997", "api-pw", httpx.MockTransport(handler))
    assert await mtx.stream_state("sim-01") == "ok"
    assert seen[0].headers["authorization"] == httpx.BasicAuth("ground", "api-pw")._auth_header  # pyright: ignore[reportPrivateUsage]


@pytest.mark.parametrize(
    "response",
    [
        httpx.Response(200, content=b"not json"),
        httpx.Response(200, json=["x"]),
        httpx.Response(401),
    ],
)
async def test_odd_answers_mean_down_or_degraded(ground: Ground, response: httpx.Response) -> None:
    mtx = video.MediaMtx("http://127.0.0.1:9997", "pw", httpx.MockTransport(lambda r: response))
    assert await mtx.stream_state("sim-01") in ("down", "degraded")
    assert await mtx.stream_state("sim-01") != "ok"


@pytest.mark.parametrize(
    "body", [{"segments": "x"}, {"segments": [{"no": 1}]}, ["x"], {"segments": [{"start": "bad"}]}]
)
async def test_odd_recordings_answer_is_unavailable(ground: Ground, body: Any) -> None:
    flow: Any = FakeFlow(ground, FakeClock(start_utc=at(100)))
    transport = httpx.MockTransport(lambda r: httpx.Response(200, json=body))
    await video.close_video(
        flow, mission(at(10), at(90)), video.MediaMtx("http://x:1", "pw", transport)
    )
    assert flow.repo.files == []
    assert ground.queued_events()[-1]["type"] == "video.unavailable"


async def test_monitor_keeps_running_after_an_error(
    ground: Ground, clock: FakeClock, monkeypatch: pytest.MonkeyPatch
) -> None:
    class Boom(video.MediaMtx):
        async def stream_state(self, vehicle_id: str) -> Any:
            raise RuntimeError("boom")

    sleeps: list[float] = []

    async def sleep(seconds: float) -> None:
        sleeps.append(seconds)
        if len(sleeps) == 2:
            raise asyncio.CancelledError

    monkeypatch.setattr(clock, "sleep", sleep)
    ground.video_state = "ok"
    with pytest.raises(asyncio.CancelledError):
        await video.VideoMonitor(ground, Boom("http://x:1")).run()
    assert len(sleeps) == 2  # it went round again after the error
    assert ground.video_state == "down"


async def test_a_failing_archive_never_stops_the_close(
    ground: Ground, monkeypatch: pytest.MonkeyPatch
) -> None:
    async def broken(*_: Any) -> None:
        raise OSError(28, "No space left on device")

    monkeypatch.setattr(video, "_close_video", broken)
    flow: Any = FakeFlow(ground, FakeClock(start_utc=at(100)))
    await video.close_video(flow, mission(at(10), at(90)))  # does not raise
    event = ground.queued_events()[-1]
    assert event["type"] == "video.archive_failed"
    assert event["severity"] == "critical"


async def test_a_gap_in_the_stream_is_not_part_of_a_mission(ground: Ground) -> None:
    """Stream 0-50 s, then nothing until a new stream at 500 s: a mission at 200-300 s
    gets neither segment, and a mission at 10-40 s gets the first, ending at 50 s."""
    clock = FakeClock(start_utc=at(600))
    starts = [at(0), at(500)]
    folder = recordings(ground, starts)
    first = folder / f"{at(0).strftime(video.NAME_FORMAT)}.mp4"
    os.utime(first, (at(50).timestamp(), at(50).timestamp()))
    second = folder / f"{at(500).strftime(video.NAME_FORMAT)}.mp4"
    os.utime(second, (at(590).timestamp(), at(590).timestamp()))
    flow: Any = FakeFlow(ground, clock)
    mtx = video.MediaMtx("http://x:1", "pw", api(starts))
    await video.close_video(flow, mission(at(200), at(300)), mtx)
    assert flow.repo.files == []
    await video.close_video(flow, mission(at(10), at(40)), mtx)
    (row,) = [r for r in flow.repo.files if r["rel_path"].endswith(".mp4")]
    assert row["started_at"] == at(0)
    assert row["ended_at"] == at(50)
