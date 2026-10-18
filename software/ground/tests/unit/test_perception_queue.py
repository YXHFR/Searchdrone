"""The queue between the link's inbound loop and the perception storage: the loop only
queues, one task per vehicle stores in order, a failing report is logged and skipped, and
a full queue drops the oldest report."""

from __future__ import annotations

import asyncio
import datetime as dt
from collections.abc import Awaitable
from typing import Any

import pytest

from sd_ground.services import perception
from sd_schemas.base import new_id
from sd_schemas.link import PerceptionReport, StillPose

VID = "sd-test-1"


def _report() -> PerceptionReport:
    return PerceptionReport(
        still_id=new_id(),
        captured_utc=dt.datetime.now(dt.UTC),
        pose=StillPose(lat=0, lon=0, alt_amsl_m=0, agl_m=40, roll_deg=0, pitch_deg=0, yaw_deg=0),
        footprint=[],
        px_per_m=24.0,
        synthetic=True,
        sources=["color"],
        latency_ms=1,
        detections=[],
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

    async def on_report(flow: Any, vehicle_id: str, report: PerceptionReport) -> None:
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

    async def on_report(flow: Any, vehicle_id: str, report: PerceptionReport) -> None:
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
