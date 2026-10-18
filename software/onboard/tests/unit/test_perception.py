"""Perception onboard: stills from a folder, the pipeline on rendered pictures, the link.

The stills are the detector's rendered test pictures (Gazebo, SK-RED objects), written
to a folder the way the simulator's camera process writes them. The telemetry is built
from each picture's camera pose, so the georeferenced blocks can be checked against the
ground truth.
"""

from __future__ import annotations

import asyncio
import base64
import io
import json
import math
from dataclasses import replace
from pathlib import Path
from typing import Any

import pytest
from PIL import Image

from sd_geo import PinholeCamera
from sd_mav.vehicle import FcTelemetry
from sd_obs.clock import MonotonicClock
from sd_onboard.link_client import OnboardLink
from sd_onboard.perception import loop as loop_module
from sd_onboard.perception.loop import MAX_POSE_AGE_MS, PerceptionLoop, pose_from
from sd_onboard.perception.sources import DirStillSource, read_ppm
from sd_safety.targets import Plane
from sd_schemas.geo import plane_offset
from sd_testing import FakeClock

STILLS = Path(__file__).resolve().parents[3] / "detector" / "tests" / "testdata" / "stills"
META: dict[str, Any] = json.loads((STILLS / "stills.json").read_text(encoding="utf-8"))
HOME = (-35.363261, 149.165230)  # the public SITL site, as in the rendered world
LABELS = {"red_block": 1, "person": 2, "car": 10, "building": 11, "road": 12}
CAM = PinholeCamera(1280, 720, 66.0)
SIM = {"SD_SIM": "1"}


def shot(name: str) -> dict[str, Any]:
    return next(s for s in META["shots"] if s["name"] == name)


def write_still(folder: Path, name: str, stem: str | None = None) -> None:
    """A rendered picture in the camera folder's format: PPM plus JSON with the boxes."""
    folder.mkdir(parents=True, exist_ok=True)
    stem = stem or name
    with Image.open(STILLS / f"{name}.png") as im:
        rgb = im.convert("RGB")
        (folder / f"{stem}.ppm").write_bytes(b"P6 %d %d 255\n" % rgb.size + rgb.tobytes())
    boxes = [
        {
            "label": LABELS[b["cls"]],
            "x0": b["x"],
            "y0": b["y"],
            "x1": b["x"] + b["w"],
            "y1": b["y"] + b["h"],
        }
        for b in shot(name)["boxes"]
    ]
    (folder / f"{stem}.json").write_text(
        json.dumps({"name": stem, "boxes": boxes}), encoding="utf-8"
    )


def telemetry(name: str, now_ms: int) -> FcTelemetry:
    s = shot(name)
    lat, lon = plane_offset(*HOME, s["east_m"], s["north_m"])
    a = s["attitude"]
    return FcTelemetry(
        position_mono_ms=now_ms,
        attitude_mono_ms=now_ms,
        lat=lat,
        lon=lon,
        alt_rel_home_m=s["height_m"],
        alt_amsl_m=584.0 + s["height_m"],
        roll_deg=a["roll_deg"],
        pitch_deg=a["pitch_deg"],
        yaw_deg=a["yaw_deg"],
    )


def loop(
    folder: Path, clock: FakeClock, env: dict[str, str], keep: Path | None = None
) -> PerceptionLoop:
    return PerceptionLoop(
        DirStillSource(folder, synthetic=True),
        lambda: FcTelemetry(),
        lambda _r: None,
        clock,
        CAM,
        keep_dir=keep,
        env=env,
    )


def truth_error_m(lat: float, lon: float) -> float:
    plane = Plane(*HOME)
    e, n = plane.xy(lat, lon)
    blocks = [o for o in META["objects"] if o["cls"] == "red_block"]
    return min(math.hypot(e - b["east_m"], n - b["north_m"]) for b in blocks)


def test_read_ppm() -> None:
    f = read_ppm(b"P6\n2 1\n255\n" + bytes(6))
    assert (f.width, f.height, len(f.rgb)) == (2, 1, 6)
    for bad in (
        b"P5 2 1 255\n" + bytes(2),
        b"P6 2 1 65535\n" + bytes(12),
        b"P6 2",
        b"P6 2 1 255\n" + bytes(5),  # truncated pixel data
    ):
        with pytest.raises(ValueError):
            read_ppm(bad)
    with pytest.raises(ValueError, match="out of range"):
        read_ppm(b"P6 4 1 255\n" + bytes(12), max_size=(2, 2))


def test_a_still_larger_than_the_camera_is_not_read(tmp_path: Path) -> None:
    write_still(tmp_path, "block0_h30")  # 1280 x 720
    src = DirStillSource(tmp_path, synthetic=True, max_size=(640, 360))
    assert src.poll() is None
    assert list(tmp_path.iterdir()) == []


def test_source_takes_the_newest_still_and_drops_older_ones(tmp_path: Path) -> None:
    write_still(tmp_path, "empty_lake_h40", "000001")
    write_still(tmp_path, "block0_h30", "000002")
    src = DirStillSource(tmp_path, synthetic=True)
    got = src.poll()
    assert got is not None and got.name == "000002" and got.synthetic
    assert got.still.sim_boxes is not None and got.still.sim_boxes[0].gz_label == 1
    assert src.skipped == 1
    assert list(tmp_path.iterdir()) == []
    assert src.poll() is None
    assert DirStillSource(tmp_path / "missing", synthetic=True).poll() is None


def test_unreadable_still_is_skipped(tmp_path: Path) -> None:
    (tmp_path / "000001.ppm").write_bytes(b"garbage")
    (tmp_path / "000001.json").write_text("{}", encoding="utf-8")
    assert DirStillSource(tmp_path, synthetic=True).poll() is None
    assert list(tmp_path.iterdir()) == []


@pytest.mark.parametrize("name", ["block0_h20", "block1_h30", "block2_h40", "block0_tilted"])
def test_report_places_the_block_on_the_truth(tmp_path: Path, name: str) -> None:
    clock = FakeClock()
    write_still(tmp_path, name)
    lp = loop(tmp_path, clock, {})
    captured = lp.source.poll()
    assert captured is not None
    report = lp.handle(captured, telemetry(name, clock.monotonic_ms()), clock.monotonic_ms())
    assert report is not None and report.synthetic
    blocks = [d for d in report.detections if d.label == "red_block"]
    assert blocks, report
    for d in blocks:
        assert d.source == "color"
        assert truth_error_m(d.lat, d.lon) <= 0.5
        assert 0.4 <= d.size_m <= 1.2
        assert d.sigma95_m > 0
        assert d.crop_jpeg_b64 is not None
        with Image.open(io.BytesIO(base64.b64decode(d.crop_jpeg_b64))) as thumb:
            assert thumb.format == "JPEG" and max(thumb.size) <= 128
            assert not thumb.getexif()
    assert report.sources == ["color"] and report.skipped_sources == {
        "model": "inte byggd än",
        "oracle": "bara i simulatorn",
    }
    assert len(report.footprint) == 4
    assert len(report.model_dump_json()) < 64 * 1024  # inside the link's message limit


def test_oracle_context_in_the_simulator(tmp_path: Path) -> None:
    clock = FakeClock()
    write_still(tmp_path, "block0_h40")
    lp = loop(tmp_path, clock, SIM, keep=tmp_path / "kept")
    captured = lp.source.poll()
    assert captured is not None
    report = lp.handle(captured, telemetry("block0_h40", 0), 0)
    assert report is not None
    got = {(d.label, d.source) for d in report.detections}
    assert ("red_block", "color") in got and ("car", "oracle") in got and ("road", "oracle") in got
    assert report.detections[0].label == "red_block"  # targets first
    assert all(d.crop_jpeg_b64 is None for d in report.detections if d.source == "oracle")
    kept = sorted(p.suffix for p in (tmp_path / "kept").iterdir())
    assert kept == [".jpg", ".json"]
    assert lp.stats.by_label["red_block"] == 1


def test_red_car_is_not_reported_as_a_block(tmp_path: Path) -> None:
    write_still(tmp_path, "red_car_0_h30")
    lp = loop(tmp_path, FakeClock(), {})
    captured = lp.source.poll()
    assert captured is not None
    report = lp.handle(captured, telemetry("red_car_0_h30", 0), 0)
    assert report is not None and report.detections == []


def test_stale_pose_or_ground_gives_no_report(tmp_path: Path) -> None:
    t = telemetry("block0_h30", 0)
    assert pose_from(t, MAX_POSE_AGE_MS + 1) is None
    assert pose_from(replace(t, attitude_mono_ms=None), 0) is None
    assert pose_from(t, 0) is not None
    write_still(tmp_path, "block0_h30")
    lp = loop(tmp_path, FakeClock(), {})
    captured = lp.source.poll()
    assert captured is not None
    assert lp.handle(captured, replace(t, alt_rel_home_m=0.5), 0) is None
    assert lp.stats.skipped_on_ground == 1


class _Session:
    def __init__(self) -> None:
        self.sent: list[str] = []

    async def send(self, body: Any, **_: Any) -> None:
        self.sent.append(type(body).__name__)


async def test_reports_go_after_events_on_the_link(tmp_path: Path) -> None:
    write_still(tmp_path, "block0_h30")
    lp = loop(tmp_path, FakeClock(), {})
    captured = lp.source.poll()
    assert captured is not None
    report = lp.handle(captured, telemetry("block0_h30", 0), 0)
    assert report is not None
    link = OnboardLink(
        url="ws://127.0.0.1:1/x", token="t" * 40, vehicle_id="sim-01", clock=MonotonicClock()
    )
    link.perception(report)
    link.perception(report)
    link.event("executor.state_changed", "info", {"to": "HOLD"})
    session = _Session()
    flush = asyncio.ensure_future(link._flush(session))  # type: ignore[arg-type]  # pyright: ignore[reportPrivateUsage]
    try:
        for _ in range(100):
            if session.sent.count("PerceptionReport") == 2:
                break
            await asyncio.sleep(0.01)
    finally:
        flush.cancel()
        await asyncio.gather(flush, return_exceptions=True)
    assert session.sent == ["OnboardEvent", "PerceptionReport", "PerceptionReport"]


def test_queue_keeps_the_newest_reports(tmp_path: Path) -> None:
    write_still(tmp_path, "block0_h30")
    lp = loop(tmp_path, FakeClock(), {})
    captured = lp.source.poll()
    assert captured is not None
    report = lp.handle(captured, telemetry("block0_h30", 0), 0)
    assert report is not None
    link = OnboardLink(
        url="ws://127.0.0.1:1/x", token="t" * 40, vehicle_id="sim-01", clock=MonotonicClock()
    )
    for _ in range(12):
        link.perception(report)
    assert link.dropped_perception == 2


def test_kept_stills_are_capped(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(loop_module, "KEEP_MAX", 2)
    lp = loop(tmp_path / "in", FakeClock(), {}, keep=tmp_path / "kept")
    ids = []
    for _ in range(3):
        write_still(tmp_path / "in", "block0_h30")
        captured = lp.source.poll()
        assert captured is not None
        report = lp.handle(captured, telemetry("block0_h30", 0), 0)
        assert report is not None and report.detections
        ids.append(str(report.still_id))
    kept = sorted(p.name for p in (tmp_path / "kept").iterdir())
    assert kept == sorted(f"{i}{ext}" for i in ids[1:] for ext in (".jpg", ".json"))


class _FlakySource:
    """Raises on the first poll, then gives one real still, then nothing."""

    def __init__(self, folder: Path) -> None:
        self.inner = DirStillSource(folder, synthetic=True)
        self.polls = 0
        self.threads: set[str] = set()

    def poll(self) -> Any:
        import threading

        self.threads.add(threading.current_thread().name)
        self.polls += 1
        if self.polls == 1:
            raise RuntimeError("camera folder gone")
        return self.inner.poll()


async def test_run_skips_a_failing_still_sends_reports_and_stops(tmp_path: Path) -> None:
    import threading

    write_still(tmp_path, "block0_h30")
    clock = MonotonicClock()
    sent: list[Any] = []
    source = _FlakySource(tmp_path)
    lp = PerceptionLoop(
        source,
        lambda: telemetry("block0_h30", clock.monotonic_ms()),  # read when the still arrives
        sent.append,
        clock,
        CAM,
        period_s=0.01,
        env={},
    )
    stop = asyncio.Event()
    task = asyncio.ensure_future(lp.run(stop))
    for _ in range(500):
        if sent:
            break
        await asyncio.sleep(0.01)
    assert not task.done()  # the failing poll did not end the loop
    stop.set()
    await asyncio.wait_for(task, 2)
    assert lp.stats.failed == 1
    assert len(sent) == 1 and sent[0].detections[0].label == "red_block"
    assert threading.current_thread().name not in source.threads  # never on the event loop


class _TooLarge:
    def __init__(self) -> None:
        self.sent: list[str] = []

    async def send(self, body: Any, **_: Any) -> None:
        from sd_link.envelope import MessageTooLargeError

        if type(body).__name__ == "PerceptionReport":
            raise MessageTooLargeError("too large")
        self.sent.append(type(body).__name__)


async def test_a_report_too_large_to_send_never_holds_up_events(tmp_path: Path) -> None:
    write_still(tmp_path, "block0_h30")
    lp = loop(tmp_path, FakeClock(), {})
    captured = lp.source.poll()
    assert captured is not None
    report = lp.handle(captured, telemetry("block0_h30", 0), 0)
    assert report is not None
    link = OnboardLink(
        url="ws://127.0.0.1:1/x", token="t" * 40, vehicle_id="sim-01", clock=MonotonicClock()
    )
    link.perception(report)
    session = _TooLarge()
    flush = asyncio.ensure_future(link._flush(session))  # type: ignore[arg-type]  # pyright: ignore[reportPrivateUsage]
    try:
        for _ in range(100):
            if link.dropped_perception:
                break
            await asyncio.sleep(0.01)
        link.event("executor.state_changed", "info", {"to": "HOLD"})
        for _ in range(100):
            if session.sent:
                break
            await asyncio.sleep(0.01)
        assert not flush.done()
    finally:
        flush.cancel()
        await asyncio.gather(flush, return_exceptions=True)
    assert link.dropped_perception == 1
    assert session.sent == ["OnboardEvent"]
