"""Compare-and-set and telling a person from a failsafe."""

from __future__ import annotations

import asyncio

import pytest
from hypothesis import HealthCheck, given, settings
from hypothesis import strategies as st
from onboard_helpers import Rig, at, rig

from sd_mav.modes import CopterMode
from sd_mav.vehicle import FcTelemetry, SendOrigin
from sd_onboard.executor.authority import Authority, LateMode
from sd_schemas.base import new_id
from sd_schemas.executor import ExecutorState as S
from sd_schemas.link import OperatorRelease


async def _in_goto(r: Rig) -> None:
    await r.takeoff()
    lat, lon = at(150, 0)
    await r.send(r.order("goto", {"lat": lat, "lon": lon, "alt_m": 40, "speed_ms": 8}))
    await r.run(3)
    assert r.ex.state == S.TRANSIT


async def test_compare_and_set() -> None:
    """A mode change from outside in the middle of a goto: HELD, the next target never goes."""
    async with rig() as r:
        await _in_goto(r)
        r.fc.set_mode_external(CopterMode.LOITER)
        await r.run(0.3)
        before = len(r.vehicle.sent)
        await r.run(10)
        assert r.ex.state == S.HELD and r.ex.held_by == "external"
        assert r.steering_sent(before) == []
        assert r.ex.over.stop_latch  # an unrequested LOITER means stop
        assert r.ex.resume_point is not None


async def test_classification_window() -> None:
    async with rig() as r:
        await _in_goto(r)
        r.fc.set_mode_external(CopterMode.RTL)
        await r.run(1.5)
        r.fc.statustext("Battery 1 is low 15.1V used 7600 mAh")
        await r.run(1.0)  # t = 2.5 s after the change
        assert r.ex.state == S.FAILSAFE
    async with rig() as r:
        await _in_goto(r)
        r.fc.set_mode_external(CopterMode.RTL)
        await r.run(2.5)
        assert r.ex.state == S.HELD and r.ex.held_by == "external"
        assert not r.ex.control_suspended  # the window is over; HELD stays
        r.fc.statustext("GCS Failsafe")  # too late to count
        await r.run(1.0)
        assert r.ex.state == S.HELD
    async with rig() as r:
        await _in_goto(r)
        r.fc.statustext("Fence breach")
        await r.run(1.0)
        r.fc.set_mode_external(CopterMode.SMART_RTL)  # 1 s after the signal
        await r.run(1.5)
        assert r.ex.state == S.FAILSAFE


async def test_notice_names_who_changed_the_mode() -> None:
    async with rig() as r:
        await _in_goto(r)
        r.fc.set_mode_external(CopterMode.BRAKE)  # the ground's STOP arrives first as MAVLink
        await r.run(0.3)
        r.ex.on_override(r.notice("stop"))
        assert r.ex.state == S.HELD and r.ex.held_by == "operator_stop"


async def test_external_mode_change_held() -> None:
    """Orders after a takeover are all refused, and nothing that steers goes out."""
    async with rig() as r:
        await _in_goto(r)
        r.fc.set_mode_external(CopterMode.LOITER)
        await r.run(0.5)
        before = len(r.vehicle.sent)
        lat, lon = at(50, 50)
        for _ in range(20):
            for skill, params in (
                ("goto", {"lat": lat, "lon": lon, "alt_m": 40}),
                ("resume", {}),
                ("hold", {}),
            ):
                cmd = await r.send(r.order(skill, params), 0.2)
                assert r.up.codes(cmd.command_id) == ("ONB_CONTROL_HELD",)
        assert r.steering_sent(before) == []


async def test_fc_silent_goes_held() -> None:
    async with rig() as r:
        await _in_goto(r)
        r.fc.silent = True
        await r.run(4)
        assert r.ex.state == S.HELD and r.ex.held_by == "fc_silent"
        assert "executor.fc_silent" in r.up.kinds()


async def test_inbound_command_ignored() -> None:
    """A MAVLink command to the onboard computer changes nothing."""
    from pymavlink.dialects.v20 import ardupilotmega as mavlink

    async with rig() as r:
        await _in_goto(r)
        r.fc.set_mode_external(CopterMode.LOITER)
        await r.run(0.5)
        for sysid in (255, 253):
            msg = mavlink.MAVLink_command_long_message(
                1, 191, mavlink.MAV_CMD_USER_1, 0, 0, 0, 0, 0, 0, 0, 0
            )
            enc = mavlink.MAVLink(None, srcSystem=sysid, srcComponent=190)
            r.vehicle.link.receive(bytes(msg.pack(enc)))
        await r.run(0.5)
        assert r.ex.state == S.HELD
        assert r.up.kinds().count("executor.inbound_command_ignored") == 2


_OPS = st.lists(
    st.sampled_from(
        [
            "goto",
            "resume",
            "hold",
            "survey",
            "takeoff",
            "home",
            "mode",
            "failsafe",
            "stop",
            "rtl_notice",
            "watchdog",
            "battery",
            "cancel",
            "wait",
        ]
    ),
    min_size=1,
    max_size=12,
)


async def _restart_rig_case(ops: list[str]) -> None:
    async with rig(start=False) as r:
        # The vehicle flies (armed, 40 m, LOITER) when sd-onboard starts: HELD.
        r.fc.armed, r.fc.alt_rel, r.fc.mode = True, 40.0, CopterMode.LOITER
        r.fc.target = (r.fc.lat, r.fc.lon, 40.0)
        task = asyncio.ensure_future(r.ex.run(r.stop))
        r.background.append(task)
        await r.until(lambda: r.ex._started, 10)
        r.ex.on_link(True, r.clock.monotonic_ms())
        await r.install()
        assert r.ex.state == S.HELD
        before = len(r.vehicle.sent)
        lat, lon = at(60, 0)
        polygon = {
            "type": "Polygon",
            "coordinates": [[[lon, lat], [lon + 1e-4, lat], [lon + 1e-4, lat + 1e-4], [lon, lat]]],
        }
        for op in ops:
            match op:
                case "goto":
                    await r.ex.on_skill_command(
                        r.order("goto", {"lat": lat, "lon": lon, "alt_m": 40}), deadline_ms=None
                    )
                case "resume" | "hold" | "takeoff":
                    await r.ex.on_skill_command(r.order(op, {}), deadline_ms=None)
                case "home":
                    await r.ex.on_skill_command(r.order("return_home", {}), deadline_ms=None)
                case "survey":
                    await r.ex.on_skill_command(
                        r.order("survey_sector", {"polygon": polygon}), deadline_ms=None
                    )
                case "mode":
                    r.fc.set_mode_external(CopterMode.GUIDED)
                case "failsafe":
                    r.fc.statustext("GCS Failsafe")
                case "stop":
                    r.ex.on_override(r.notice("stop"))
                case "rtl_notice":
                    r.ex.on_override(r.notice("return_home"))
                case "watchdog":
                    r.ex.on_override(r.notice("watchdog_stop"))
                case "battery":
                    r.ex.on_override(r.notice("battery_rtl"))
                case "cancel":
                    r.ex.on_cancel(new_id())
                case _:
                    await r.run(0.5)
            await r.run(0.1)
        await r.run(1)
        assert r.ex.state in (S.HELD, S.FAILSAFE)
        # Only the emergency fallback's mode steps (the ground's MAVLink is not part of
        # this rig, so the flight controller never shows the target mode by itself).
        assert r.steering_sent(before, allow_fallback=True) == []
        # Only Resume hands back; then steering is possible again (when the mode allows).
        r.fc.set_mode_external(CopterMode.LOITER)
        await r.run(0.5)
        r.ex.on_release(OperatorRelease(action_seq=r._seq, operator_id="operator:pilot"), new_id())


@settings(max_examples=2000, deadline=None, suppress_health_check=[HealthCheck.too_slow])
@given(ops=_OPS)
def test_no_control_when_held(ops: list[str]) -> None:
    asyncio.run(_restart_rig_case(ops))


# --- a mode of ours, sent and cancelled before a HEARTBEAT showed it -------------------


def _shows(mode: CopterMode, at_ms: int, rc: int | None = None) -> FcTelemetry:
    return FcTelemetry(fc_heartbeat_mono_ms=at_ms, fc_custom_mode=int(mode), rc_mode_pwm=rc)


def _late_auto(rc: int | None = None) -> Authority:
    """GUIDED expected; a switch to AUTO accepted and cancelled, its deadline at 3000."""
    a = Authority()
    a.confirmed(CopterMode.GUIDED, accepted_ms=None, seen_ms=None)
    a.sent_and_cancelled(CopterMode.AUTO, 3000, rc)
    return a


# Modes a person or a failsafe picks: never ours while a late AUTO may still show.
_OTHERS = (
    CopterMode.ALT_HOLD,
    CopterMode.LOITER,
    CopterMode.BRAKE,
    CopterMode.RTL,
    CopterMode.SMART_RTL,
    CopterMode.LAND,
)


def test_a_cancelled_auto_is_ours_until_its_own_confirm_deadline() -> None:
    """Accepted and cancelled: a HEARTBEAT showing AUTO up to that switch's own deadline is
    ours (it becomes the expected mode); after it, it is a change from outside."""
    a = _late_auto()
    t = _shows(CopterMode.AUTO, 3000)
    assert a.unexpected_mode(t, 3000) is None
    assert a.mode_problem(t, 3000) is None
    assert a.adopt_late(t, 3000) == CopterMode.AUTO
    assert a.expected_mode == CopterMode.AUTO and a.late is None
    a = _late_auto()
    t = _shows(CopterMode.AUTO, 3001)
    assert a.adopt_late(t, 3001) is None
    assert a.unexpected_mode(t, 3001) is not None and a.late is None


@pytest.mark.parametrize("other", _OTHERS)
def test_another_mode_while_a_cancelled_auto_may_show_is_not_ours(other: CopterMode) -> None:
    a = _late_auto()
    t = _shows(other, 1100)
    assert a.adopt_late(t, 1100) is None
    assert a.unexpected_mode(t, 1100) is not None
    assert a.mode_problem(t, 1100) == f"mode_{other.name}"


@pytest.mark.parametrize(
    "mode", [CopterMode.LOITER, CopterMode.GUIDED, CopterMode.RTL, CopterMode.SMART_RTL]
)
def test_only_a_cancelled_auto_is_kept(mode: CopterMode) -> None:
    """A cancelled LOITER (or any mode but AUTO) that shows later is a change from outside:
    the executor never steers by holding a mode, and a person's LOITER must stop it."""
    a = Authority()
    a.confirmed(CopterMode.AUTO, accepted_ms=None, seen_ms=None)
    a.sent_and_cancelled(mode, 3000, None)
    assert a.late is None
    t = _shows(mode, 1100)
    assert a.adopt_late(t, 1100) is None and a.unexpected_mode(t, 1100) is not None


def test_a_notice_a_change_from_outside_or_a_hold_ends_a_cancelled_auto() -> None:
    """After an OperatorOverrideNotice (STOP, FLY HOME, a takeover), a change from
    outside, or a hold the executor enters itself, AUTO showing late is not ours."""
    t = _shows(CopterMode.AUTO, 1100)
    a = _late_auto()
    a.notice(1050)
    assert a.adopt_late(t, 1100) is None and a.unexpected_mode(t, 1100) is not None
    a = _late_auto()
    change = a.unexpected_mode(_shows(CopterMode.ALT_HOLD, 1050), 1050)
    assert change is not None
    a.start_classification(change)
    assert a.unexpected_mode(t, 1100) is not None
    a = _late_auto()
    a.pending_mode = CopterMode.LOITER
    a.drop_ours()
    assert a.late is None and a.pending_mode is None
    assert a.unexpected_mode(t, 1100) is not None


def test_any_heartbeat_after_a_newer_accept_ends_a_cancelled_auto() -> None:
    """The flight controller handles mode commands in order: after a newer mode command of
    ours was accepted, the cancelled AUTO can no longer show. Any HEARTBEAT received after
    that ACCEPTED ends the record, whatever mode it shows; AUTO in it is from outside."""
    t = _shows(CopterMode.AUTO, 1100)
    # #161 round-4 review: GUIDED confirmed from the telemetry before its accept.
    a = _late_auto()
    a.confirmed(CopterMode.GUIDED, accepted_ms=1050, seen_ms=1000)
    assert a.late is not None
    assert a.unexpected_mode(_shows(CopterMode.AUTO, 1040), 1045) is None  # before the accept
    # Received in the accept's own millisecond: it may have left before the command.
    assert a.unexpected_mode(_shows(CopterMode.AUTO, 1050), 1050) is None
    assert a.unexpected_mode(t, 1100) is not None and a.late is None
    # Recorded at the ACCEPTED, with no confirm (yet): the same.
    a = _late_auto()
    a.accepted(CopterMode.LOITER, 1050)
    assert a.unexpected_mode(_shows(CopterMode.AUTO, 1040), 1045) is None
    assert a.unexpected_mode(t, 1100) is not None and a.late is None
    # Confirmed from a HEARTBEAT after the accept: ended at once.
    a = _late_auto()
    a.confirmed(CopterMode.LOITER, accepted_ms=1050, seen_ms=1060)
    assert a.late is None and a.unexpected_mode(t, 1100) is not None
    # The earliest newer accept counts.
    a = _late_auto()
    a.accepted(CopterMode.LOITER, 1050)
    a.accepted(CopterMode.GUIDED, 1200)
    assert a.unexpected_mode(t, 1100) is not None
    # A new switch to AUTO itself: it is pending now, the old record ends.
    a = _late_auto()
    a.accepted(CopterMode.AUTO, 1050)
    assert a.late is None


def test_a_pilots_rc_auto_seen_before_the_rc_sample_is_a_takeover() -> None:
    """A late AUTO taken as ours, then RC_CHANNELS shows the mode switch moved: the pilot
    chose AUTO, a takeover. The switch is watched to the deadline, and at least
    ``RC_RECHECK_MS`` after the adopt; noise on the channel is not a move."""
    a = _late_auto(rc=1100)
    assert a.adopt_late(_shows(CopterMode.AUTO, 1100, rc=1100), 1100) == CopterMode.AUTO
    assert a.unexpected_mode(_shows(CopterMode.AUTO, 1200, rc=1150), 1200) is None
    change = a.unexpected_mode(_shows(CopterMode.AUTO, 1300, rc=1700), 1300)
    assert change is not None and change.mode == "AUTO"
    # Adopted near the deadline: watched RC_RECHECK_MS past it.
    a = _late_auto(rc=1100)
    assert a.adopt_late(_shows(CopterMode.AUTO, 2900, rc=1100), 2900) == CopterMode.AUTO
    assert a.unexpected_mode(_shows(CopterMode.AUTO, 3800, rc=1700), 3800) is not None
    # After the watch: the switch is the pilot's own business again (no mode change).
    a = _late_auto(rc=1100)
    assert a.adopt_late(_shows(CopterMode.AUTO, 1100, rc=1100), 1100) == CopterMode.AUTO
    assert a.unexpected_mode(_shows(CopterMode.AUTO, 3001, rc=1100), 3001) is None
    assert a.adopted is None
    # A notice ends the watch.
    a = _late_auto(rc=1100)
    a.adopt_late(_shows(CopterMode.AUTO, 1100, rc=1100), 1100)
    a.notice(1150)
    assert a.adopted is None


def test_the_rc_mode_switch_moving_ends_a_cancelled_auto() -> None:
    """A person who moved the RC mode switch since the switch was sent chose the mode, AUTO
    included; noise on the channel is not a move."""
    a = _late_auto(rc=1300)
    assert a.unexpected_mode(_shows(CopterMode.AUTO, 1100, rc=1350), 1100) is None
    assert a.unexpected_mode(_shows(CopterMode.AUTO, 1200, rc=1700), 1200) is not None
    assert a.late is None
    a = _late_auto(rc=None)  # no RC receiver heard when it was sent: one heard now moved
    assert a.unexpected_mode(_shows(CopterMode.AUTO, 1100, rc=1300), 1100) is not None


async def _late_auto_live(r: Rig) -> None:
    """In a goto (GUIDED), the executor's set_mode(AUTO) is cancelled after the flight
    controller accepted and took it, while its HEARTBEAT is still held back."""
    await _in_goto(r)
    r.fc.heartbeat_held = True
    task = asyncio.ensure_future(r.ex.set_mode(CopterMode.AUTO, SendOrigin.SKILL))
    await r.until(lambda: r.fc.mode == CopterMode.AUTO, 2)
    await r.run(0.05)
    task.cancel()
    await r.run(0.05)
    assert task.cancelled() and r.ex.auth.pending_mode is None
    assert r.ex.auth.late is not None


async def test_a_switch_cancelled_after_the_flight_controller_took_it_is_not_a_takeover() -> None:
    """The executor's set_mode(AUTO) is cancelled after the flight controller took AUTO but
    before a HEARTBEAT showed it: the HEARTBEAT that then shows AUTO is not a takeover,
    and the executor steers on with AUTO expected."""
    async with rig() as r:
        await _late_auto_live(r)
        r.fc.release_heartbeat()
        await r.run(0.5)
        assert r.ex.state == S.TRANSIT, (r.ex.state, r.ex.held_by)
        assert r.ex.auth.expected_mode == CopterMode.AUTO
        assert "executor.mode_late" in r.up.kinds()
        assert "executor.control_taken" not in r.up.kinds()


def _taken(r: Rig) -> None:
    assert r.ex.state == S.HELD and r.ex.held_by == "external", (
        r.ex.state,
        r.ex.held_by,
        r.fc.mode,
    )
    assert "executor.mode_late" not in r.up.kinds()


@pytest.mark.parametrize("mode", [*_OTHERS, CopterMode.AUTO])
async def test_rc_while_a_cancelled_auto_may_show_is_a_takeover(mode: CopterMode) -> None:
    """The pilot flips the RC mode switch while a cancelled AUTO may still show: HELD,
    whatever the mode, AUTO included."""
    async with rig() as r:
        await _late_auto_live(r)
        r.fc.rc_mode_switch(1300, mode)
        r.fc.release_heartbeat()
        await r.run(1.0)
        _taken(r)


async def test_a_cancelled_own_loiter_then_rc_loiter_is_a_takeover() -> None:
    """#161 round-3 review: an own LOITER cancelled after the flight controller took it,
    then GUIDED sent and confirmed; 0.3 s later the pilot picks LOITER on the RC switch.
    HELD, and the next goto is refused (on main as well)."""
    async with rig() as r:
        await _in_goto(r)
        exp = r.ex.auth.expected_mode
        assert exp is not None
        r.fc.heartbeat_held = True
        task = asyncio.ensure_future(r.ex.set_mode(CopterMode.LOITER, SendOrigin.SKILL))
        await r.until(lambda: r.fc.mode == CopterMode.LOITER, 2)
        task.cancel()
        await r.run(0.05)
        t2 = asyncio.ensure_future(r.ex.set_mode(exp, SendOrigin.SKILL))
        await r.until(lambda: r.fc.mode == exp, 2)
        r.fc.release_heartbeat()
        await r.until(t2.done, 3)
        await r.run(0.3)
        r.fc.rc_mode_switch(1300, CopterMode.LOITER)
        await r.run(1.0)
        _taken(r)
        assert r.ex.over.stop_latch
        lat, lon = at(50, 50)
        cmd = await r.send(r.order("goto", {"lat": lat, "lon": lon, "alt_m": 40}), 0.5)
        assert r.up.codes(cmd.command_id) == ("ONB_CONTROL_HELD",)


async def test_auto_from_outside_after_a_newer_own_mode_is_a_takeover() -> None:
    """A cancelled AUTO, then the executor's own GUIDED is accepted and a HEARTBEAT after
    that shows it: the cancelled AUTO can no longer show, so AUTO from outside (a ground
    station) is a takeover."""
    async with rig() as r:
        await _late_auto_live(r)
        t2 = asyncio.ensure_future(r.ex.set_mode(CopterMode.GUIDED, SendOrigin.SKILL))
        await r.until(lambda: r.fc.mode == CopterMode.GUIDED, 2)
        r.fc.release_heartbeat()
        await r.until(t2.done, 3)
        await r.run(0.3)
        assert t2.result() and r.ex.auth.late is None
        r.fc.set_mode_external(CopterMode.AUTO)
        await r.run(1.0)
        _taken(r)


async def test_auto_from_outside_after_a_stale_confirm_is_a_takeover() -> None:
    """#161 round-4 review: a cancelled AUTO; the executor's own GUIDED is confirmed from
    the telemetry before its accept (the HEARTBEAT is still held); then AUTO from outside
    before that HEARTBEAT comes. The HEARTBEAT after the accept showing AUTO is a takeover."""
    async with rig() as r:
        await _late_auto_live(r)
        t2 = asyncio.ensure_future(r.ex.set_mode(CopterMode.GUIDED, SendOrigin.SKILL))
        await r.until(t2.done, 3)
        assert t2.result() and r.ex.auth.late is not None
        r.fc.set_mode_external(CopterMode.AUTO)
        r.fc.release_heartbeat()
        await r.run(1.0)
        _taken(r)


async def test_auto_from_outside_after_a_newer_accept_not_confirmed_is_a_takeover() -> None:
    """A cancelled AUTO; the executor's own LOITER is accepted and still waits for its
    confirm; AUTO from outside comes first. The HEARTBEAT after that ACCEPTED showing AUTO
    is a takeover: the record of the newer switch is made at its ACCEPTED."""
    async with rig() as r:
        await _late_auto_live(r)
        t2 = asyncio.ensure_future(r.ex.set_mode(CopterMode.LOITER, SendOrigin.SKILL))
        await r.until(lambda: r.fc.mode == CopterMode.LOITER, 2)
        await r.run(0.05)
        assert not t2.done()
        late = r.ex.auth.late
        assert late is not None and late.newer_ms is not None
        r.fc.set_mode_external(CopterMode.AUTO)
        r.fc.release_heartbeat()
        await r.run(1.0)
        _taken(r)
        await r.until(t2.done, 3)


async def test_a_pilots_rc_auto_ahead_of_its_rc_sample_is_a_takeover() -> None:
    """A cancelled AUTO; the pilot picks AUTO on the RC switch and the HEARTBEAT showing it
    comes before the RC_CHANNELS sample that shows the switch moved. AUTO is first taken as
    ours; the sample then makes it a takeover."""
    async with rig() as r:
        r.fc.rc_mode_pwm = 1100
        await _late_auto_live(r)
        late = r.ex.auth.late
        assert late is not None and late.rc_mode_pwm == 1100
        r.fc.release_heartbeat()
        await r.until(lambda: r.ex.auth.expected_mode == CopterMode.AUTO, 1)
        assert r.ex.state == S.TRANSIT
        r.fc.rc_mode_pwm = 1700  # the switch, seen in the next RC_CHANNELS
        await r.run(1.0)
        assert r.ex.state == S.HELD and r.ex.held_by == "external", (r.ex.state, r.ex.held_by)
        assert "executor.control_taken" in r.up.kinds()


async def test_a_loiter_recorded_as_cancelled_does_not_hide_rc_loiter() -> None:
    """Even told directly that a LOITER was cancelled, a LOITER on the RC switch is a
    takeover, and the next order is refused."""
    async with rig() as r:
        await _in_goto(r)
        r.ex.auth.sent_and_cancelled(CopterMode.LOITER, r.ex.now_ms() + 2000, None)
        r.fc.rc_mode_switch(1300, CopterMode.LOITER)
        await r.run(1.0)
        _taken(r)
        lat, lon = at(50, 50)
        cmd = await r.send(r.order("goto", {"lat": lat, "lon": lon, "alt_m": 40}), 0.5)
        assert r.up.codes(cmd.command_id) == ("ONB_CONTROL_HELD",)


async def test_a_switch_cancelled_before_it_was_accepted_is_not_kept() -> None:
    """Cancelled while it waited for the flight controller's answer: not kept, so AUTO
    showing later is classified as any change (the safe side)."""
    async with rig() as r:
        await _in_goto(r)
        r.fc.heartbeat_held = True
        r.fc.drop_acks = 1  # the switch acts, its answer is lost
        task = asyncio.ensure_future(r.ex.set_mode(CopterMode.AUTO, SendOrigin.SKILL))
        await r.until(lambda: r.fc.mode == CopterMode.AUTO, 2)
        task.cancel()
        await r.run(0.05)
        assert r.ex.auth.late is None
        r.fc.release_heartbeat()
        await r.run(1.0)
        _taken(r)


async def test_a_hold_of_the_executors_own_ends_a_cancelled_auto() -> None:
    """The flight controller goes silent with a cancelled AUTO on record: HELD (fc_silent),
    and nothing of ours is on the way any more. A late AUTO is never taken as ours while
    the executor does not steer."""
    async with rig() as r:
        await _in_goto(r)
        # A deadline past the 3 s to fc_silent, so only the hold can end it.
        r.ex.auth.sent_and_cancelled(CopterMode.AUTO, r.ex.now_ms() + 10_000, None)
        r.ex.auth.pending_mode = CopterMode.LOITER
        r.fc.silent = True
        await r.until(lambda: r.ex.state == S.HELD, 10)
        assert r.ex.held_by == "fc_silent"
        assert r.ex.auth.late is None and r.ex.auth.pending_mode is None
        r.ex.auth.late = LateMode(CopterMode.AUTO, r.ex.now_ms() + 10_000, None)
        r.fc.mode = CopterMode.AUTO
        r.fc.silent = False
        await r.run(1.0)
        assert r.ex.auth.expected_mode == CopterMode.GUIDED
        assert "executor.mode_late" not in r.up.kinds()
