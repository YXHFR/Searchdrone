"""Compare-and-set and telling a person from a failsafe."""

from __future__ import annotations

import asyncio

from heartbeat_hold import HeartbeatHold
from hypothesis import HealthCheck, given, settings
from hypothesis import strategies as st
from onboard_helpers import Rig, at, rig

from sd_mav.modes import CopterMode
from sd_mav.vehicle import FcTelemetry, SendOrigin
from sd_onboard.executor.authority import Authority
from sd_safety.constants import MODE_CONFIRM_TIMEOUT_S
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


def _shows(mode: CopterMode, at_ms: int) -> FcTelemetry:
    return FcTelemetry(fc_heartbeat_mono_ms=at_ms, fc_custom_mode=int(mode))


def test_a_cancelled_mode_of_ours_is_ours_until_its_confirm_timeout() -> None:
    """Sent and cancelled: a HEARTBEAT showing it within MODE_CONFIRM_TIMEOUT_S is ours (it
    becomes the expected mode); after that, or another mode, is a change from outside."""
    late_ms = int(MODE_CONFIRM_TIMEOUT_S * 1000)
    a = Authority()
    a.confirmed(CopterMode.LOITER)
    a.sent_and_cancelled(CopterMode.AUTO, 1000)
    t = _shows(CopterMode.AUTO, 1000 + late_ms)
    assert a.unexpected_mode(t, 1000 + late_ms) is None
    assert a.mode_problem(t, 1000 + late_ms) is None
    assert a.adopt_late(t, 1000 + late_ms) == CopterMode.AUTO
    assert a.expected_mode == CopterMode.AUTO and not a.cancelled
    # Too late: a person's (or a failsafe's).
    a = Authority()
    a.confirmed(CopterMode.LOITER)
    a.sent_and_cancelled(CopterMode.AUTO, 1000)
    t = _shows(CopterMode.AUTO, 1001 + late_ms)
    assert a.adopt_late(t, 1001 + late_ms) is None
    assert a.unexpected_mode(t, 1001 + late_ms) is not None
    # Another mode in the window is not ours either.
    a = Authority()
    a.confirmed(CopterMode.LOITER)
    a.sent_and_cancelled(CopterMode.AUTO, 1000)
    t = _shows(CopterMode.ALT_HOLD, 1100)
    assert a.adopt_late(t, 1100) is None
    assert a.unexpected_mode(t, 1100) is not None
    assert a.mode_problem(t, 1100) == "mode_ALT_HOLD"


def test_a_notice_or_a_change_from_outside_ends_a_cancelled_mode_being_ours() -> None:
    """After an OperatorOverrideNotice (STOP, FLY HOME, a takeover) or a change from
    outside, a late HEARTBEAT showing the cancelled mode is not taken as ours."""
    a = Authority()
    a.confirmed(CopterMode.LOITER)
    a.sent_and_cancelled(CopterMode.AUTO, 1000)
    a.notice(1050)
    t = _shows(CopterMode.AUTO, 1100)
    assert a.adopt_late(t, 1100) is None and a.unexpected_mode(t, 1100) is not None
    a = Authority()
    a.confirmed(CopterMode.LOITER)
    a.sent_and_cancelled(CopterMode.AUTO, 1000)
    change = a.unexpected_mode(_shows(CopterMode.ALT_HOLD, 1050), 1050)
    assert change is not None
    a.start_classification(change)
    assert a.unexpected_mode(_shows(CopterMode.AUTO, 1100), 1100) is not None


async def test_a_switch_cancelled_after_the_flight_controller_took_it_is_not_a_takeover() -> None:
    """The executor's set_mode(AUTO) is cancelled after the flight controller took AUTO but
    before a HEARTBEAT showed it: the HEARTBEAT that then shows AUTO is not a takeover,
    and the executor steers on with AUTO expected."""
    async with rig() as r:
        await _in_goto(r)
        hold = HeartbeatHold(r.fc)
        hold.held = True
        task = asyncio.ensure_future(r.ex.set_mode(CopterMode.AUTO, SendOrigin.SKILL))
        await r.until(lambda: r.fc.mode == CopterMode.AUTO, 2)
        task.cancel()
        await r.run(0.05)
        assert task.cancelled() and r.ex.auth.pending_mode is None
        hold.release()
        await r.run(0.5)
        assert r.ex.state == S.TRANSIT, (r.ex.state, r.ex.held_by)
        assert r.ex.auth.expected_mode == CopterMode.AUTO
        assert "executor.mode_late" in r.up.kinds()
        assert "executor.control_taken" not in r.up.kinds()
