"""``VehicleActions`` and ``OperatorPresence`` on a stub MAVLink thread.

The STOP / FLY HOME flows on a fake flight controller are in ``test_stop_home.py``;
these tests cover the emergency path's own bookkeeping: which vehicles count as
reachable, the answer caches, refusals and how the main loop's part survives errors.
"""

from __future__ import annotations

import asyncio
from dataclasses import dataclass, field
from typing import Any

import pytest
import structlog.testing
from stop_helpers import OBSERVER, PILOT, VID, RecordingSink, StubGround

from sd_ground.mav.thread import ActionReport, MavSnapshot, ModeAction
from sd_ground.services.latches import Latches
from sd_ground.services.vehicle_actions import (
    OUTCOMES_MAX,
    ActionRefusedError,
    NoNoticeSink,
    NoticeResult,
    OperatorPresence,
    VehicleActions,
)
from sd_ground.sources import EmergencyUnavailableError
from sd_mav.vehicle import FcTelemetry
from sd_schemas.base import SdModel, new_id
from sd_schemas.link import CommanderStatus, OperatorOverrideNotice, StickAxes
from sd_testing import FakeClock

OTHER = "sim-02"


@dataclass
class StubMav:
    """The MAVLink thread as ``VehicleActions`` sees it: one vehicle, connected or not."""

    vehicle_id: str = VID
    connected: bool = True
    submitted: list[ModeAction] = field(default_factory=list[ModeAction])
    calls: list[str] = field(default_factory=list[str])

    def submit(self, action: ModeAction) -> bool:
        self.submitted.append(action)
        return True

    def takeover(self, takeover_id: str, action_seq: int, wait_s: float = 0.0) -> bool:
        self.calls.append("takeover" if wait_s <= 0 else "takeover_after_notice")
        return True

    def takeover_acknowledged(self, takeover_id: str) -> bool:
        self.calls.append("takeover_acknowledged")
        return True

    def handback(self) -> bool:
        self.calls.append("handback")
        return True

    def sticks(self, axes: StickAxes) -> bool:
        return True

    def snapshot(self) -> MavSnapshot | None:
        return MavSnapshot(
            vehicle_id=self.vehicle_id,
            connected=self.connected,
            telemetry=FcTelemetry(),
            now_mono_ms=0,
            in_flight=False,
            onboard_alive=True,
            airborne_since=None,
            manual=False,
            watchdog_phase="ok",
        )


@dataclass
class PerVehicleSink:
    """SD-link that is up only for some vehicles."""

    up_for: set[str] = field(default_factory=set[str])

    def up(self, vehicle_id: str) -> bool:
        return vehicle_id in self.up_for

    async def send(self, vehicle_id: str, message: SdModel) -> bool:
        return vehicle_id in self.up_for

    async def send_acked(self, vehicle_id: str, message: SdModel, timeout_s: float) -> NoticeResult:
        return "acked" if vehicle_id in self.up_for else "down"


class BrokenSink(RecordingSink):
    async def send(self, vehicle_id: str, message: SdModel) -> bool:
        raise ConnectionResetError("link went away mid-send")

    async def send_acked(self, vehicle_id: str, message: SdModel, timeout_s: float) -> NoticeResult:
        raise ConnectionResetError("link went away mid-send")


def _actions(
    clock: FakeClock,
    *,
    mav: StubMav | None = None,
    notices: Any = None,
    ids: list[str] | None = None,
) -> VehicleActions:
    actions = VehicleActions(clock, Latches(clock), vehicle_ids=ids or [VID], notices=notices)
    actions.mav = mav
    return actions


def _stop(actions: VehicleActions, client_action_id: str | None = None) -> Any:
    return actions.dispatch(
        "stop", VID, client_action_id=client_action_id or str(new_id()), via="button", by=PILOT
    )


def _published(g: StubGround) -> list[tuple[str, dict[str, Any]]]:
    return [(e.type, e.data) for e in g.bus.replay(g.bus.server_epoch, 0) or []]


# --- reachability and refusals ---------------------------------------------------------


def test_vehicles_counts_mavlink_or_sdlink() -> None:
    clock = FakeClock()
    mav = StubMav(vehicle_id=VID, connected=True)
    sink = PerVehicleSink(up_for={OTHER})
    actions = _actions(clock, mav=mav, notices=sink, ids=[VID, OTHER, "sim-03"])
    assert actions.vehicles() == [VID, OTHER]  # VID by MAVLink, OTHER by SD-link
    mav.connected = False
    assert actions.vehicles() == [OTHER]
    sink.up_for = set()
    assert actions.vehicles() == []
    actions.mav = None
    sink.up_for = {VID}
    assert actions.vehicles() == [VID]


async def test_without_a_sink_nothing_reaches_onboard() -> None:
    sink = NoNoticeSink()
    notice = OperatorOverrideNotice(
        override_id=new_id(), kind="stop", action_seq=1, issuer_kind="system", operator_id="x"
    )
    assert sink.up(VID) is False
    assert await sink.send(VID, notice) is False


def test_system_action_without_any_path_sets_no_latch() -> None:
    clock = FakeClock()
    actions = _actions(clock, mav=StubMav(connected=False))
    assert actions.system(VID, "battery_rtl") is None
    assert actions.latches.get(VID) is None
    assert actions.post.empty() and actions.latest_via(VID) is None


def test_system_action_with_a_path_sets_its_latch() -> None:
    clock = FakeClock()
    mav = StubMav()
    actions = _actions(clock, mav=mav)
    reply = actions.system(VID, "watchdog_stop")
    assert reply is not None and reply.paths["mavlink"] == "queued"
    latch = actions.latches.get(VID)
    assert latch is not None and latch.action == "watchdog_stop"
    assert [a.kind for a in mav.submitted] == ["watchdog_stop"]
    assert actions.latest_via(VID) == "system"


def test_pilot_action_on_an_unknown_vehicle_is_404() -> None:
    actions = _actions(FakeClock(), mav=StubMav())
    for call in (actions.resume, actions.takeover, actions.handback):
        with pytest.raises(ActionRefusedError) as exc:
            call("sim-99", client_action_id=str(new_id()), by=PILOT)
        assert (exc.value.status, exc.value.code) == (404, "VEHICLE_NOT_FOUND")
    assert actions.post.empty()


def test_chat_lines_count_only_for_the_pilot() -> None:
    actions = _actions(FakeClock(), mav=StubMav())
    with pytest.raises(EmergencyUnavailableError) as exc:
        actions.chat_intent("stop", vehicle_id=VID, message_id=str(new_id()), by=OBSERVER)
    assert (exc.value.status, exc.value.code) == (403, "FORBIDDEN_ROLE")
    assert actions.latches.get(VID) is None and actions.post.empty()


# --- the answer caches -----------------------------------------------------------------


def test_repeated_pilot_action_gets_the_same_answer() -> None:
    mav = StubMav()
    actions = _actions(FakeClock(), mav=mav)
    first = actions.takeover(VID, client_action_id="tap-1", by=PILOT)
    again = actions.takeover(VID, client_action_id="tap-1", by=PILOT)
    assert again == first
    assert mav.calls == ["takeover"]  # the MAVLink thread was asked once
    assert actions.post.qsize() == 1


def test_old_answers_are_forgotten_after_outcomes_max() -> None:
    """Both caches keep the newest ``OUTCOMES_MAX`` entries: a client id from before
    that is a new action, and its outcome is no longer found."""
    actions = _actions(FakeClock(), mav=StubMav())
    first = _stop(actions, "click-0")
    replies = [_stop(actions, f"click-{i}") for i in range(1, OUTCOMES_MAX + 1)]
    newest = replies[-1]
    assert _stop(actions, f"click-{OUTCOMES_MAX}") == newest  # still remembered
    assert actions.outcome(VID, str(newest.action_id)) is not None
    assert actions.outcome(VID, str(first.action_id)) is None
    again = _stop(actions, "click-0")
    assert again.action_id != first.action_id and again.action_seq > newest.action_seq


# --- the main loop's part --------------------------------------------------------------


async def test_notice_that_fails_to_send_marks_sdlink_failed() -> None:
    clock = FakeClock()
    actions = _actions(clock, mav=StubMav(), notices=BrokenSink())
    reply = _stop(actions)
    g = StubGround(clock)
    await actions.drain(g)
    outcome = actions.outcome(VID, str(reply.action_id))
    assert outcome is not None and outcome.paths["sdlink"] == "failed"
    assert outcome.paths["mavlink"] == "queued"
    assert "vehicle.stop" in g.types()  # the event is still written
    assert [t for t, _ in _published(g) if t == "vehicle.action"] == ["vehicle.action"]


async def _drain_all(actions: VehicleActions, g: StubGround) -> None:
    await actions.drain(g)
    tasks = list(actions._tasks)  # pyright: ignore[reportPrivateUsage]
    await asyncio.gather(*tasks)


async def test_takeover_goes_to_the_mavlink_thread_at_the_click() -> None:
    """With SD-link up the take-over is handed over at once, with its deadline; the
    notice's ack only brings it forward."""
    clock = FakeClock()
    mav = StubMav()
    actions = _actions(clock, mav=mav, notices=PerVehicleSink(up_for={VID}))
    reply = actions.takeover(VID, client_action_id=str(new_id()), by=PILOT)
    assert mav.calls == ["takeover_after_notice"]  # before any pump
    g = StubGround(clock)
    await _drain_all(actions, g)
    assert mav.calls == ["takeover_after_notice", "takeover_acknowledged"]
    outcome = actions.outcome(VID, str(reply.action_id))
    assert outcome is not None and outcome.paths["sdlink"] == "sent"


async def test_takeover_whose_notice_fails_still_goes_and_marks_sdlink_failed() -> None:
    clock = FakeClock()
    mav = StubMav()
    actions = _actions(clock, mav=mav, notices=BrokenSink())
    reply = actions.takeover(VID, client_action_id=str(new_id()), by=PILOT)
    g = StubGround(clock)
    await _drain_all(actions, g)
    assert mav.calls == ["takeover_after_notice"]  # LOITER comes at the deadline
    outcome = actions.outcome(VID, str(reply.action_id))
    assert outcome is not None and outcome.paths["sdlink"] == "failed"
    assert "vehicle.takeover" in g.types()


async def test_takeover_without_sdlink_goes_at_once() -> None:
    clock = FakeClock()
    mav = StubMav()
    actions = _actions(clock, mav=mav, notices=NoNoticeSink())
    actions.takeover(VID, client_action_id=str(new_id()), by=PILOT)
    assert mav.calls == ["takeover"]


async def test_a_failing_takeover_task_is_logged_not_raised() -> None:
    clock = FakeClock()
    actions = _actions(clock, mav=StubMav(), notices=PerVehicleSink(up_for={VID}))

    def broken(_g: object, _o: object) -> None:
        raise RuntimeError("the event store went away")

    actions._posted = broken  # type: ignore[method-assign]  # pyright: ignore[reportPrivateUsage]
    actions.takeover(VID, client_action_id=str(new_id()), by=PILOT)
    g = StubGround(clock)
    with structlog.testing.capture_logs() as logs:
        await actions.drain(g)
        tasks = list(actions._tasks)  # pyright: ignore[reportPrivateUsage]
        await asyncio.gather(*tasks, return_exceptions=True)
        await asyncio.sleep(0)
    assert actions._tasks == set()  # pyright: ignore[reportPrivateUsage]
    assert [e["event"] for e in logs] == ["actions.task_failed"]


async def test_stopping_the_pump_cancels_a_waiting_takeover_notice() -> None:
    clock = FakeClock()
    sink = RecordingSink()
    sink.hangs = True
    actions = _actions(clock, mav=StubMav(), notices=sink)
    pump = asyncio.ensure_future(actions.pump(StubGround(clock)))
    actions.takeover(VID, client_action_id=str(new_id()), by=PILOT)
    await clock.run_for(0.1)
    tasks = list(actions._tasks)  # pyright: ignore[reportPrivateUsage]
    assert tasks
    pump.cancel()
    await asyncio.gather(pump, *tasks, return_exceptions=True)
    assert all(t.cancelled() for t in tasks)


async def test_without_sdlink_nothing_is_acknowledged() -> None:
    message = CommanderStatus(connected=True, state="ready")
    assert await NoNoticeSink().send_acked(VID, message, 0.3) == "down"


async def test_mavlink_events_are_published_only_for_alarms_and_vehicle() -> None:
    clock = FakeClock()
    actions = _actions(clock, mav=StubMav())
    actions.on_event("mav.forbidden_mavlink_blocked", "warning", {"vehicle_id": VID})
    actions.on_event("alarm.STOP_PATH_FAILED", "critical", {"vehicle_id": VID})
    actions.on_event("vehicle.override_reasserted", "warning", {"vehicle_id": 7})
    g = StubGround(clock)
    await actions.drain(g)
    assert g.types() == [
        "mav.forbidden_mavlink_blocked",
        "alarm.STOP_PATH_FAILED",
        "vehicle.override_reasserted",
    ]
    assert [t for t, _ in _published(g)] == [
        "alarm.STOP_PATH_FAILED",
        "vehicle.override_reasserted",
    ]


async def test_report_for_an_unknown_action_is_ignored() -> None:
    clock = FakeClock()
    actions = _actions(clock, mav=StubMav())
    actions.on_report(ActionReport("not-ours", VID, "stop", 1, "acked", "BRAKE", "BRAKE", None))
    g = StubGround(clock)
    await actions.drain(g)
    assert g.events == [] and _published(g) == []


async def test_resume_without_a_latch_publishes_a_bare_release() -> None:
    clock = FakeClock()
    actions = _actions(clock, mav=StubMav(), notices=RecordingSink())
    reply = actions.resume(VID, client_action_id=str(new_id()), by=PILOT)
    g = StubGround(clock)
    await actions.drain(g)
    released = [p for t, _s, p in g.events if t == "latch.released"]
    assert released == [{"action_seq": reply.action_seq}]  # nothing was latched


async def test_pump_keeps_going_after_a_failing_round() -> None:
    clock = FakeClock()

    class FlakyGround(StubGround):
        fail = True

        def add_event(self, type_: str, **kw: Any) -> None:
            if self.fail:
                self.fail = False
                raise RuntimeError("database hiccup")
            super().add_event(type_, **kw)

    g = FlakyGround(clock)
    actions = _actions(clock, mav=StubMav())
    pump = asyncio.ensure_future(actions.pump(g))
    try:
        first = _stop(actions)  # writing its latch event fails
        await clock.run_for(0.2)
        second = _stop(actions)
        await clock.run_for(0.2)
        assert not pump.done()
        latched = [p["action_seq"] for t, _s, p in g.events if t == "latch.set"]
        assert latched == [second.action_seq]  # only the failed write is lost
        stops = [p["action_seq"] for t, _s, p in g.events if t == "vehicle.stop"]
        assert stops == [first.action_seq, second.action_seq]
    finally:
        pump.cancel()
        await asyncio.gather(pump, return_exceptions=True)


async def test_presence_watch_keeps_running_after_an_error() -> None:
    clock = FakeClock()
    actions = _actions(clock, mav=StubMav())
    calls = {"n": 0}

    def active(vehicle_id: str) -> bool:
        calls["n"] += 1
        if calls["n"] == 1:
            raise RuntimeError("mission table not ready")
        return True

    presence = OperatorPresence(actions, absent_s=1.0, mission_active=active)
    g = StubGround(clock, operator_seen_mono=clock.monotonic())
    run = asyncio.ensure_future(presence.run(g, every_s=0.1))
    try:
        await clock.run_for(1.5)
        assert not run.done()
        assert g.types().count("vehicle.operator_absent") == 1
        latch = actions.latches.get(VID)
        assert latch is not None and latch.action == "watchdog_stop"
    finally:
        run.cancel()
        await asyncio.gather(run, return_exceptions=True)
