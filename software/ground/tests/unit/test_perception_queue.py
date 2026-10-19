"""The queue between the link's inbound loop and the perception storage: the loop only
queues, one task per vehicle stores in order, a failing report is logged and skipped, a
full queue drops the oldest report, and a report waits a short while for its crops."""

from __future__ import annotations

import asyncio
import datetime as dt
from collections.abc import Awaitable
from typing import Any

import pytest

from sd_ground.services import perception
from sd_schemas.base import LatLon, new_id
from sd_schemas.link import (
    BBox,
    CropData,
    DetectorInfo,
    GeoDetectionMsg,
    PerceptionReport,
    Pose,
)

VID = "sd-test-1"


JPEG = b"\xff\xd8\xff\xe0" + b"0" * 60


def _detection(crop: bool) -> GeoDetectionMsg:
    return GeoDetectionMsg(
        detection_id=new_id(),
        bbox=BBox(x=10, y=10, w=20, h=20),
        score=0.8,
        lat=0.0,
        lon=0.0,
        sigma95_m=3.0,
        px_size=20,
        crop_inline=crop,
        label="red_block",
        source="color",
    )


def _report(*detections: GeoDetectionMsg) -> PerceptionReport:
    return PerceptionReport(
        still_id=new_id(),
        command_id=None,
        captured_utc=dt.datetime.now(dt.UTC),
        sensor="rgb",
        pose=Pose(
            lat=0, lon=0, alt_amsl_m=0, agl_m=40, roll_deg=0, pitch_deg=0, yaw_deg=0, source_ts_ms=0
        ),
        footprint=[LatLon(lat=0, lon=0)] * 4,
        px_per_m=24.0,
        synthetic=True,
        detector=DetectorInfo(
            name="sd_detect:color",
            license="Apache-2.0",
            version="0",
            placement="onboard",
            ok=True,
            tiles=1,
            latency_ms=1,
            rate_hz=1,
        ),
        detections=list(detections),
    )


def _crop(report: PerceptionReport, d: GeoDetectionMsg) -> CropData:
    return CropData(
        ref_id=d.detection_id,
        purpose="detection",
        still_id=report.still_id,
        bbox_in_still=BBox(x=0, y=0, w=60, h=60),
        width=60,
        height=60,
        jpeg=JPEG,
    )


class FakeFlow:
    def __init__(self) -> None:
        self.reports: dict[str, Any] = {}
        self.tasks: list[asyncio.Task[Any]] = []

    def spawn(self, work: Awaitable[Any], name: str) -> None:
        self.tasks.append(asyncio.ensure_future(work))


async def test_reports_are_stored_in_order_by_their_own_task(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    done: list[str] = []
    calls: list[int] = []
    gate = asyncio.Event()

    async def on_report(flow: Any, vehicle_id: str, report: PerceptionReport, crops: Any) -> None:
        await gate.wait()  # a slow database
        calls.append(1)
        if len(calls) == 2:
            raise RuntimeError("database gone")
        done.append(str(report.still_id))

    monkeypatch.setattr(perception, "on_report", on_report)
    flow: Any = FakeFlow()
    reps = [_report() for _ in range(3)]
    for rep in reps:
        perception.enqueue(flow, VID, rep)  # returns at once
    assert len(flow.tasks) == 1  # one worker per vehicle
    q = flow.reports[VID]
    assert q.running and len(q.items) == 3
    gate.set()
    await asyncio.gather(*flow.tasks)
    assert done == [str(reps[0].still_id), str(reps[2].still_id)]  # the failed one skipped
    assert q.handled == 3 and not q.running

    perception.enqueue(flow, VID, _report())  # a new worker after the queue ran dry
    await asyncio.gather(*flow.tasks)
    assert len(flow.tasks) == 2 and q.handled == 4


async def test_a_full_queue_drops_the_oldest(monkeypatch: pytest.MonkeyPatch) -> None:
    stored: list[str] = []

    async def on_report(flow: Any, vehicle_id: str, report: PerceptionReport, crops: Any) -> None:
        stored.append(str(report.still_id))

    monkeypatch.setattr(perception, "on_report", on_report)
    flow: Any = FakeFlow()
    reps = [_report() for _ in range(perception.REPORT_QUEUE_MAX + 3)]
    for rep in reps:
        perception.enqueue(flow, VID, rep)
    q = flow.reports[VID]
    assert q.dropped == 3
    await asyncio.gather(*flow.tasks)
    assert stored == [str(r.still_id) for r in reps[3:]]


async def test_a_report_waits_for_its_crops_and_takes_late_ones(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    got: list[set[str]] = []

    async def on_report(flow: Any, vehicle_id: str, report: PerceptionReport, crops: Any) -> None:
        got.append(set(crops))

    monkeypatch.setattr(perception, "on_report", on_report)
    flow: Any = FakeFlow()
    a, b, ctx = _detection(True), _detection(True), _detection(False)
    rep = _report(a, b, ctx)
    perception.add_crop(flow, VID, _crop(rep, a))  # a crop may come first
    perception.enqueue(flow, VID, rep)
    await asyncio.sleep(0.05)
    assert got == []  # still waiting for b's crop
    perception.add_crop(flow, VID, _crop(rep, b))
    await asyncio.gather(*flow.tasks)
    assert got == [{str(a.detection_id), str(b.detection_id)}]
    assert flow.reports[VID].crops == {}  # taken by the report


async def test_a_lost_crop_does_not_hold_the_report_longer_than_the_wait(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    got: list[set[str]] = []

    async def on_report(flow: Any, vehicle_id: str, report: PerceptionReport, crops: Any) -> None:
        got.append(set(crops))

    monkeypatch.setattr(perception, "on_report", on_report)
    monkeypatch.setattr(perception, "CROP_WAIT_S", 0.1)
    flow: Any = FakeFlow()
    rep = _report(_detection(True))
    perception.enqueue(flow, VID, rep)
    await asyncio.wait_for(asyncio.gather(*flow.tasks), 2)
    assert got == [set()]  # stored without the crop


def test_waiting_crops_are_capped_and_views_are_ignored() -> None:
    flow: Any = FakeFlow()
    rep = _report()
    for _ in range(perception.CROPS_KEPT_MAX + 5):
        perception.add_crop(flow, VID, _crop(rep, _detection(True)))
    view = _crop(rep, _detection(True)).model_copy(update={"purpose": "view"})
    perception.add_crop(flow, VID, view)
    crops = flow.reports[VID].crops
    assert len(crops) == perception.CROPS_KEPT_MAX
    assert str(view.ref_id) not in crops
