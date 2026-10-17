"""``sd_ground.mav.gcs`` on its own: the flight phase, the disarm step and the watch.

The functions run straight on a ground ``FakeVehicle`` (250/190) and a
``FakeFlightController`` on a ``FakeClock``, without the MAVLink thread around them.
"""

from __future__ import annotations

import asyncio
from collections.abc import AsyncIterator, Awaitable
from contextlib import asynccontextmanager
from dataclasses import dataclass, field
from typing import Any, TypeVar

from pymavlink.dialects.v20 import ardupilotmega as mavlink
from stop_helpers import ground_disarms, ground_modes

from sd_ground.mav.gcs import guard_override, phase_of, run_override
from sd_mav.connection import Role
from sd_mav.fake import FakeFlightController, FakeLink, FakeVehicle
from sd_mav.modes import HOME_STEPS, STOP_STEPS, CopterMode, OverridePhase
from sd_testing import FakeClock

T = TypeVar("T")


class ScriptedFc(FakeFlightController):
    """A fake flight controller that can keep some telemetry back (``quiet``) and answer
    a disarm its own way (``on_disarm``: the result to send; ``None``: as usual)."""

    def __init__(self, clock: FakeClock) -> None:
        super().__init__(clock)
        self.quiet: set[str] = set()
        self.on_disarm: Any = None

    def _send(self, msg: Any, link: FakeLink | None = None) -> None:
        if msg.get_type() in self.quiet:
            return
        super()._send(msg, link)

    def _command(self, link: FakeLink, msg: Any) -> int:
        disarm = int(msg.command) == mavlink.MAV_CMD_COMPONENT_ARM_DISARM and int(msg.param1) == 0
        if disarm and self.on_disarm is not None:
            return int(self.on_disarm())
        return super()._command(link, msg)


@dataclass
class Bench:
    clock: FakeClock
    fc: ScriptedFc
    v: FakeVehicle
    events: list[tuple[str, dict[str, Any]]] = field(
        default_factory=list[tuple[str, dict[str, Any]]]
    )

    async def run(self, work: Awaitable[T], limit_s: float = 15.0) -> T:
        task = asyncio.ensure_future(work)
        await self.clock.run_until(task, limit_s)
        return task.result()

    def armed_on_ground(self) -> None:
        self.fc.armed = True
        self.fc.mode = CopterMode.GUIDED
        self.fc.alt_rel = 0.0
        self.fc.target = None

    def lift_off(self, alt: float = 3.0) -> None:
        self.fc.alt_rel = alt
        self.fc.target = (self.fc.lat, self.fc.lon, alt)


@asynccontextmanager
async def bench() -> AsyncIterator[Bench]:
    clock = FakeClock()
    fc = ScriptedFc(clock)
    events: list[tuple[str, dict[str, Any]]] = []
    v = FakeVehicle(
        clock, fc=fc, role=Role.GROUND, on_event=lambda kind, d: events.append((kind, d))
    )
    await v.start()
    b = Bench(clock, fc, v, events)
    await clock.run_for(2.0)  # heartbeat, position and landed state are fresh
    try:
        yield b
    finally:
        await v.close()


# --- the flight phase ------------------------------------------------------------------


async def test_stale_position_with_a_disarmed_heartbeat_is_on_the_ground() -> None:
    """Landed state and position are old, but a fresh HEARTBEAT says disarmed: nothing
    to stop, so STOP sends nothing. Armed with the same stale picture counts as airborne."""
    async with bench() as b:
        b.fc.quiet = {"EXTENDED_SYS_STATE", "GLOBAL_POSITION_INT"}
        await b.clock.run_for(2.0)
        assert phase_of(b.v) is OverridePhase.ON_GROUND_DISARMED
        before = len(b.fc.received)
        result = await b.run(run_override(b.v, "stop"))
        assert (result.result, result.mode_requested, result.mode_confirmed) == (
            "already_active",
            None,
            "DISARMED",
        )
        assert ground_modes(b.fc, before) == [] and ground_disarms(b.fc, before) == 0
        b.fc.armed = True
        await b.clock.run_for(0.5)
        assert phase_of(b.v) is OverridePhase.AIRBORNE


# --- the disarm step -------------------------------------------------------------------


async def test_armed_on_the_ground_reports_the_disarm_step() -> None:
    async with bench() as b:
        b.armed_on_ground()
        await b.clock.run_for(0.6)
        assert phase_of(b.v) is OverridePhase.ON_GROUND_ARMED
        steps: list[str] = []
        result = await b.run(run_override(b.v, "return_home", on_step=steps.append))
        assert steps == ["DISARM"]
        assert (result.result, result.mode_requested, result.mode_confirmed) == (
            "acked",
            "DISARM",
            "DISARMED",
        )
        assert not b.fc.armed


async def test_disarm_aborted_by_a_newer_action_sends_nothing() -> None:
    async with bench() as b:
        b.armed_on_ground()
        await b.clock.run_for(0.6)
        before = len(b.fc.received)
        result = await b.run(run_override(b.v, "stop", aborted=lambda: True))
        assert (result.result, result.mode_requested, result.mode_confirmed, result.alarm) == (
            "aborted",
            "DISARM",
            None,
            None,
        )
        assert ground_disarms(b.fc, before) == 0 and b.fc.armed


async def test_disarm_accepted_but_still_armed_is_resent_then_the_stop_steps() -> None:
    """The flight controller says yes but stays armed: the disarm goes again every
    resend interval until ``STEP_CONFIRM_S`` has passed, then BRAKE at once."""
    async with bench() as b:
        b.armed_on_ground()
        b.fc.on_disarm = lambda: mavlink.MAV_RESULT_ACCEPTED  # and does nothing
        await b.clock.run_for(0.6)
        before = len(b.fc.received)
        steps: list[str] = []
        result = await b.run(run_override(b.v, "stop", on_step=steps.append))
        assert 4 <= ground_disarms(b.fc, before) <= 6  # 1 s at one every 0.2 s
        assert steps == ["DISARM", "BRAKE"]
        assert ground_modes(b.fc, before) == [CopterMode.BRAKE]
        assert (result.result, result.mode_requested, result.mode_confirmed) == (
            "acked",
            "BRAKE",
            "BRAKE",
        )
        assert result.alarm is None


async def test_lifting_off_during_the_disarm_takes_the_mode_steps() -> None:
    """A disarm that is answered while the vehicle rises: the next round sees it airborne
    and sends no second disarm, only the mode steps."""
    async with bench() as b:
        b.armed_on_ground()

        def rise() -> int:
            b.lift_off()
            return mavlink.MAV_RESULT_ACCEPTED

        b.fc.on_disarm = rise
        await b.clock.run_for(0.6)
        before = len(b.fc.received)
        result = await b.run(run_override(b.v, "return_home"))
        assert ground_disarms(b.fc, before) == 1
        assert ground_modes(b.fc, before)[:1] == [HOME_STEPS[0]]
        assert result.result == "acked" and result.mode_requested == HOME_STEPS[0].name
        assert b.fc.armed


async def test_disarm_refused_by_the_allowlist_after_lift_off_takes_the_mode_steps() -> None:
    """Two actions disarm at once: the first one's disarm is lost on the way and holds
    the command; meanwhile the vehicle lifts off. The second disarm is then refused by
    the ground's allowlist (nothing sent) and that action goes straight to its steps."""
    async with bench() as b:
        b.armed_on_ground()
        await b.clock.run_for(0.6)
        before = len(b.fc.received)
        b.fc.drop_commands = 1
        first = asyncio.ensure_future(run_override(b.v, "stop"))
        await asyncio.sleep(0)
        second = asyncio.ensure_future(run_override(b.v, "return_home"))
        await b.clock.run_for(0.05)
        b.lift_off()
        both = asyncio.gather(first, second)
        await b.clock.run_until(both, 15.0)
        stop, home = both.result()
        assert ground_disarms(b.fc, before) == 1  # only the lost one ever went out
        blocked = [d for kind, d in b.events if kind == "mav.forbidden_mavlink_blocked"]
        assert [d["reason"] for d in blocked] == ["disarm only when standing on the ground"]
        assert home.mode_requested in {m.name for m in HOME_STEPS}
        assert stop.mode_requested in {m.name for m in STOP_STEPS}
        assert {CopterMode.BRAKE, HOME_STEPS[0]} <= set(ground_modes(b.fc, before))


# --- the watch after the click ---------------------------------------------------------


async def test_watch_without_a_callback_still_undoes_a_late_order() -> None:
    async with bench() as b:
        b.fc.armed, b.fc.alt_rel, b.fc.mode = True, 20.0, CopterMode.BRAKE
        await b.clock.run_for(0.6)
        before = len(b.fc.received)
        watch = asyncio.ensure_future(
            guard_override(
                b.v,
                "stop",
                CopterMode.BRAKE,
                clicked_mono=b.clock.monotonic(),
                rc_at_click=None,
            )
        )
        await b.clock.run_for(0.5)
        b.fc.set_mode_external(CopterMode.GUIDED)  # a late order from onboard
        await b.clock.run_for(1.0)
        assert b.fc.mode == CopterMode.BRAKE
        assert ground_modes(b.fc, before) == [CopterMode.BRAKE]
        b.fc.armed, b.fc.alt_rel = False, 0.0  # landed and disarmed: the watch ends
        await b.clock.run_until(watch, 2.0)
        assert watch.result() == 1
