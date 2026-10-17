"""The onboard fallback for STOP and FLY HOME when the ground's MAVLink is cut."""

from __future__ import annotations

from onboard_helpers import Rig, at, is_fallback_step, rig

from sd_mav.modes import HOME_MODES, STOP_MODES, CopterMode
from sd_mav.vehicle import SendOrigin
from sd_onboard.executor.override_fallback import FALLBACK_DELAY_S
from sd_schemas.base import new_id
from sd_schemas.executor import ExecutorState as S
from sd_schemas.link import OperatorRelease


async def _in_goto(r: Rig) -> None:
    await r.takeoff()
    lat, lon = at(150, 0)
    await r.send(r.order("goto", {"lat": lat, "lon": lon, "alt_m": 40, "speed_ms": 8}))
    await r.run(3)
    assert r.ex.state == S.TRANSIT


def _modes(msgs: list[object]) -> list[CopterMode]:
    return [CopterMode(int(m.param2)) for m in msgs if is_fallback_step(m)]  # type: ignore[attr-defined]


async def test_stop_via_fallback_when_ground_mavlink_down() -> None:
    """Only the notice arrives (the ground's MAVLink is cut): BRAKE from 1/191 within
    the 1.5 s window plus the step time, and the executor stays held."""
    async with rig() as r:
        await _in_goto(r)
        before = len(r.vehicle.sent)
        t0 = r.clock.monotonic()
        r.ex.on_override(r.notice("stop"))
        await r.run(FALLBACK_DELAY_S - 0.1)
        assert _modes(r.steering_sent(before)) == []
        await r.until(lambda: r.fc.mode in STOP_MODES, 2.5)
        assert r.clock.monotonic() - t0 <= 2.5
        assert r.fc.mode == CopterMode.BRAKE
        assert r.ex.state == S.HELD and r.ex.held_by == "operator_stop"
        assert "executor.override_fallback_used" in r.up.kinds()
        senders = {(m.get_srcSystem(), m.get_srcComponent()) for m in r.steering_sent(before)}
        assert senders == {(r.fc.sysid, 191)}


async def test_no_fallback_when_the_ground_got_through() -> None:
    async with rig() as r:
        await _in_goto(r)
        before = len(r.vehicle.sent)
        r.ex.on_override(r.notice("return_home"))
        await r.ground.set_mode(CopterMode.SMART_RTL, origin=SendOrigin.GROUND)
        await r.run(4)
        assert r.fc.mode in HOME_MODES
        assert r.steering_sent(before) == []
        assert "executor.override_fallback_used" not in r.up.kinds()


async def test_fallback_newer_action_wins() -> None:
    """STOP (seq n) and 0.5 s later FLY HOME (seq n+1), ground cut: home modes by 3 s and
    no BRAKE from 1/191 after the newer notice arrived."""
    async with rig() as r:
        await _in_goto(r)
        r.fc.refuse_modes = {int(CopterMode.BRAKE), int(CopterMode.LOITER)}
        r.ex.on_override(r.notice("stop"))
        await r.run(0.5)
        after_home = len(r.vehicle.sent)
        r.ex.on_override(r.notice("return_home"))
        await r.run(2.5)
        assert r.fc.mode in HOME_MODES
        late = _modes(r.vehicle.sent[after_home:])
        assert late and all(m in HOME_MODES for m in late), late


async def test_fallback_chain_takes_the_next_step_when_refused() -> None:
    async with rig() as r:
        await _in_goto(r)
        r.fc.refuse_modes = {int(CopterMode.SMART_RTL)}
        before = len(r.vehicle.sent)
        r.ex.on_override(r.notice("battery_rtl"))
        await r.run(FALLBACK_DELAY_S + 1.0)
        assert r.fc.mode == CopterMode.RTL
        assert _modes(r.steering_sent(before))[:2] == [CopterMode.SMART_RTL, CopterMode.RTL]


async def test_refused_resume_does_not_cancel_the_fallback() -> None:
    """Resume while the flight controller still flies GUIDED is refused onboard; the
    STOP still stands, so the fallback still sends its steps."""
    async with rig() as r:
        await _in_goto(r)
        before = len(r.vehicle.sent)
        r.ex.on_override(r.notice("stop"))
        await r.run(0.5)
        ref = new_id()
        r.ex.on_release(OperatorRelease(action_seq=r.next_seq(), operator_id="operator:pilot"), ref)
        assert r.up.codes(ref) == ("ONB_RELEASE_MODE_UNSUPPORTED",)
        await r.run(3)
        assert _modes(r.vehicle.sent[before:])[:1] == [CopterMode.BRAKE]
        assert r.fc.mode in STOP_MODES


async def test_landing_cancels_the_fallback() -> None:
    async with rig() as r:
        await _in_goto(r)
        before = len(r.vehicle.sent)
        r.ex.on_override(r.notice("stop"))
        r.ex.over.landed()  # what the executor does at LANDED
        await r.run(3)
        assert _modes(r.vehicle.sent[before:]) == []


async def test_no_fallback_on_the_ground() -> None:
    async with rig() as r:
        await r.install()
        before = len(r.vehicle.sent)
        r.ex.on_override(r.notice("stop"))
        await r.run(4)
        assert r.steering_sent(before) == []
        assert "executor.override_fallback_used" not in r.up.kinds()


async def test_fallback_leaves_an_rc_mode_change_alone() -> None:
    """A mode only a person with an RC transmitter can choose is never overridden."""
    async with rig() as r:
        await _in_goto(r)
        before = len(r.vehicle.sent)
        r.ex.on_override(r.notice("stop"))
        await r.run(0.5)
        r.fc.set_mode_external(CopterMode.ALT_HOLD)
        await r.run(3)
        assert _modes(r.vehicle.sent[before:]) == []
        assert r.fc.mode == CopterMode.ALT_HOLD


async def test_fallback_lets_a_new_failsafe_stand() -> None:
    from pymavlink.dialects.v20 import ardupilotmega as mavlink

    async with rig() as r:
        await _in_goto(r)
        before = len(r.vehicle.sent)
        r.ex.on_override(r.notice("stop"))
        await r.run(0.8)
        r.fc.charge_state = mavlink.MAV_BATTERY_CHARGE_STATE_LOW  # begins now
        await r.run(3)
        assert _modes(r.vehicle.sent[before:]) == []
