"""STOP, FLY HOME and the other notices onboard; Resume; the stop latch; takeover."""

from __future__ import annotations

from uuid import UUID

import pytest
from onboard_helpers import Rig, at, is_fallback_step, rig

from sd_mav.modes import HOME_STEPS, STOP_STEPS, CopterMode
from sd_onboard.executor.override import OVERRIDE_KINDS
from sd_schemas.base import new_id
from sd_schemas.executor import ExecutorState as S
from sd_schemas.link import ManualTakeover, OperatorRelease


async def _in_goto(r: Rig) -> None:
    await r.takeoff()
    lat, lon = at(150, 0)
    await r.send(r.order("goto", {"lat": lat, "lon": lon, "alt_m": 40, "speed_ms": 8}))
    await r.run(3)
    assert r.ex.state == S.TRANSIT


def _release(r: Rig, seq: int | None = None) -> UUID:
    ref = new_id()
    rel = OperatorRelease(action_seq=r._seq if seq is None else seq, operator_id="operator:pilot")
    r.ex.on_release(rel, ref)
    return ref


@pytest.mark.parametrize("kind", sorted(OVERRIDE_KINDS))
async def test_override_notice_kinds(kind: str) -> None:
    rule = OVERRIDE_KINDS[kind]
    async with rig() as r:
        await _in_goto(r)
        before = len(r.vehicle.sent)
        r.ex.on_override(r.notice(kind))
        assert r.ex.state == S.HELD
        assert r.ex.held_by == rule.held_by
        assert r.ex.over.stop_latch is rule.sets_stop_latch
        await r.run(1.4)
        assert r.steering_sent(before) == []  # nothing that steers after the notice
        # The ground's MAVLink did not get through (the flight controller kept its mode):
        # after 1.5 s the fallback sends the first step of the kind's chain itself.
        await r.run(3)
        first = STOP_STEPS[0] if rule.steps == "stop" else HOME_STEPS[0]
        sent = r.steering_sent(before)
        assert sent and all(is_fallback_step(m) for m in sent)
        assert {int(m.param2) for m in sent} == {int(first)}
        assert r.fc.mode == first
        assert ("executor.override_fallback_used", kind) in [
            (e, d.get("kind")) for e, _s, d in r.up.events
        ]
        assert r.ex.state == S.HELD  # the executor stays held
        await r.run(1)
        ref = _release(r)
        await r.run(0.5)
        if rule.battery_rtl:
            assert r.up.codes(ref) == ("ONB_BATTERY_RTL_ACTIVE",)
            assert r.ex.state == S.HELD
        elif rule.steps == "stop":
            assert r.ex.state == S.HOLD and not r.ex.over.stop_latch
            await r.run(3)
            assert r.fc.mode == CopterMode.LOITER  # out of BRAKE as the first own action
        else:
            assert r.ex.state == S.RTL


async def test_old_notice_is_ignored() -> None:
    async with rig() as r:
        await _in_goto(r)
        r.ex.on_override(r.notice("return_home", seq=50))
        assert r.ex.held_by == "operator_home"
        r.ex.on_override(r.notice("stop", seq=10))
        assert r.ex.held_by == "operator_home" and not r.ex.over.stop_latch


async def test_stop_latch() -> None:
    async with rig() as r:
        await _in_goto(r)
        for kind in ("stop", "watchdog_stop"):
            r.ex.on_override(r.notice(kind))
            r.fc.set_mode_external(CopterMode.BRAKE)
            await r.run(1)
            _release(r)  # Resume ...
            await r.run(0.5)
            assert r.ex.state == S.HOLD
            r.ex.on_override(r.notice(kind))  # ... and STOP again, latch set
            lat, lon = at(30, 0)
            # Even in HELD with a latch every AI order is refused.
            for skill, params in (
                ("goto", {"lat": lat, "lon": lon, "alt_m": 40}),
                ("resume", {}),
                ("return_home", {}),
            ):
                cmd = await r.send(r.order(skill, params, issuer="brain:ground"))
                assert r.up.codes(cmd.command_id) == ("ONB_CONTROL_HELD",), skill
            r.fc.set_mode_external(CopterMode.BRAKE)
            await r.run(1)
            # After Resume (latch released) the vehicle is HOLD again: AI orders pass.
            _release(r)
            await r.run(3)
            assert not r.ex.over.stop_latch
        # A latch without HELD (the operator released, then a stop notice arrived late
        # on the ground) still refuses the AI's moves, return home included.
        r.ex.over.stop_latch = True
        for skill in ("return_home", "resume"):
            cmd = await r.send(r.order(skill, {}, issuer="brain:ground"))
            assert r.up.codes(cmd.command_id) == ("ONB_STOP_LATCHED",), skill
        r.ex.over.stop_latch = False
        for kind in ("return_home", "battery_rtl", "watchdog_rtl"):
            r.ex.on_override(r.notice(kind))
            assert not r.ex.over.stop_latch, kind


async def test_home_after_stop_works() -> None:
    async with rig() as r:
        await _in_goto(r)
        r.ex.on_override(r.notice("stop"))
        r.ex.on_override(r.notice("return_home"))
        assert r.ex.held_by == "operator_home"
        assert r.ex.over.stop_latch  # only Resume or landing lifts it


async def test_chat_does_not_release() -> None:
    """No order from the AI can lift STOP: only the Resume click (OperatorRelease)."""
    async with rig() as r:
        await _in_goto(r)
        r.ex.on_override(r.notice("stop"))
        r.fc.set_mode_external(CopterMode.BRAKE)
        before = len(r.vehicle.sent)
        lat, lon = at(30, 30)
        for _ in range(10):
            for skill, params in (("resume", {}), ("goto", {"lat": lat, "lon": lon, "alt_m": 40})):
                cmd = await r.send(r.order(skill, params, issuer="brain:ground"), 0.5)
                assert r.up.codes(cmd.command_id) == ("ONB_CONTROL_HELD",)
        await r.run(30)
        assert r.ex.state == S.HELD
        assert r.steering_sent(before) == []


async def test_release_targets() -> None:
    async with rig() as r:
        await _in_goto(r)
        for mode, expected, code in (
            (CopterMode.GUIDED, S.HELD, "ONB_RELEASE_MODE_UNSUPPORTED"),
            (CopterMode.AUTO, S.HELD, "ONB_RELEASE_MODE_UNSUPPORTED"),
            (CopterMode.ALT_HOLD, S.HELD, "ONB_RELEASE_MODE_UNSUPPORTED"),
            (CopterMode.LOITER, S.HOLD, None),
        ):
            r.ex.on_override(r.notice("stop"))
            r.fc.set_mode_external(mode)
            await r.run(1)
            ref = _release(r)
            assert r.ex.state == expected, mode
            if code is not None:
                assert r.up.codes(ref) == (code,)
        r.ex.on_override(r.notice("return_home"))
        r.fc.set_mode_external(CopterMode.RTL)
        await r.run(1)
        _release(r)
        assert r.ex.state == S.RTL
        await r.until(lambda: r.ex.state == S.LANDED, 200, step=0.05)
        assert not r.ex.over.stop_latch and r.ex.resume_point is None


async def test_stale_release_is_refused() -> None:
    async with rig() as r:
        await _in_goto(r)
        r.ex.on_override(r.notice("stop", seq=30))
        r.fc.set_mode_external(CopterMode.BRAKE)
        await r.run(1)
        ref = _release(r, seq=29)
        assert r.up.codes(ref) == ("ONB_STALE_ORDER",)
        assert r.ex.state == S.HELD


async def test_release_then_resume_continues_the_goto() -> None:
    async with rig() as r:
        await _in_goto(r)
        r.ex.on_override(r.notice("stop"))
        r.fc.set_mode_external(CopterMode.BRAKE)
        await r.run(1)
        assert r.ex.snapshot().resume_available
        _release(r)
        await r.run(3)
        cmd = await r.send(r.order("resume", {}, issuer="brain:ground"))
        assert r.ex.state == S.TRANSIT, r.up.acks[-3:]
        await r.until(lambda: r.ex.state == S.HOLD, 120)
        assert r.up.stages(cmd.command_id)[-1] == "done"
        lat, lon = at(150, 0)
        assert abs(r.fc.lat - lat) < 3e-5 and abs(r.fc.lon - lon) < 3e-5


async def test_stop_on_the_ground_withdraws_the_clearance() -> None:
    async with rig() as r:
        await r.install()
        r.clearance()
        r.ex.on_override(r.notice("stop"))
        assert r.ex.state == S.PREFLIGHT
        cmd = await r.send(r.order("takeoff", {"alt_m": 40}))
        assert r.up.codes(cmd.command_id) == ("ONB_CLEARANCE_MISSING",)


async def test_takeover_pauses_the_ai_and_sends_nothing() -> None:
    """The ground station switches to LOITER and sends the sticks; onboard only lets go."""
    async with rig() as r:
        await _in_goto(r)
        tid = new_id()
        before = len(r.vehicle.sent)
        takeover = ManualTakeover(
            takeover_id=tid, active=True, action_seq=r.next_seq(), operator_id="operator:pilot"
        )
        r.ex.on_takeover(takeover, new_id())
        assert r.ex.state == S.HELD and r.ex.held_by == "operator_manual"
        assert r.ex.snapshot().manual_control
        r.fc.set_mode_external(CopterMode.LOITER)  # the ground station's mode change
        await r.run(3)
        assert r.ex.state == S.HELD and r.ex.held_by == "operator_manual"
        lat, lon = at(30, 0)
        goto = r.order("goto", {"lat": lat, "lon": lon, "alt_m": 40}, issuer="brain:ground")
        cmd = await r.send(goto)
        assert r.up.codes(cmd.command_id) == ("ONB_CONTROL_HELD",)
        await r.run(5)
        assert r.steering_sent(before) == []
        handback = ManualTakeover(
            takeover_id=tid, active=False, action_seq=r.next_seq(), operator_id="operator:pilot"
        )
        r.ex.on_takeover(handback, new_id())
        assert r.ex.state == S.HOLD and not r.ex.over.manual
        resume = await r.send(r.order("resume", {}, issuer="brain:ground"))
        assert "running" in r.up.stages(resume.command_id)


def _takeover(r: Rig, tid: UUID, *, active: bool) -> ManualTakeover:
    return ManualTakeover(
        takeover_id=tid, active=active, action_seq=r.next_seq(), operator_id="operator:pilot"
    )


async def _resume_codes(r: Rig) -> tuple[str, ...]:
    resume = await r.send(r.order("resume", {}, issuer="brain:ground"))
    return r.up.codes(resume.command_id)


async def test_takeover_whose_loiter_comes_before_its_notice_is_not_a_stop() -> None:
    """The ground switches to LOITER at once and its notice can come a moment later: the
    change is the pilot's take-over, not a stop, so the AI may resume after the hand-back."""
    async with rig() as r:
        await _in_goto(r)
        tid = new_id()
        r.fc.set_mode_external(CopterMode.LOITER)  # seen before the notice
        await r.run(0.3)
        assert r.ex.state == S.HELD and r.ex.held_by == "external" and r.ex.over.stop_latch
        r.ex.on_takeover(_takeover(r, tid, active=True), new_id())
        assert r.ex.held_by == "operator_manual" and not r.ex.over.stop_latch
        await r.run(3)
        r.ex.on_takeover(_takeover(r, tid, active=False), new_id())
        assert r.ex.state == S.HOLD
        resume = await r.send(r.order("resume", {}, issuer="brain:ground"))
        assert "running" in r.up.stages(resume.command_id)


async def test_an_unrequested_loiter_without_a_notice_stays_a_stop() -> None:
    async with rig() as r:
        await _in_goto(r)
        r.fc.set_mode_external(CopterMode.LOITER)
        await r.run(3)
        assert r.ex.over.stop_latch and r.ex.held_by == "external"
        resume = await r.send(r.order("resume", {}, issuer="brain:ground"))
        assert "running" not in r.up.stages(resume.command_id)  # held and latched


async def test_a_takeover_notice_outside_the_window_does_not_lift_the_stop() -> None:
    async with rig() as r:
        await _in_goto(r)
        tid = new_id()
        r.fc.set_mode_external(CopterMode.LOITER)
        await r.run(5)  # the classification window is over
        r.ex.on_takeover(_takeover(r, tid, active=True), new_id())
        assert r.ex.over.stop_latch
        r.ex.on_takeover(_takeover(r, tid, active=False), new_id())
        assert await _resume_codes(r) == ("ONB_STOP_LATCHED",)


async def test_a_stop_before_the_takeover_keeps_its_latch() -> None:
    """A real STOP stays latched when the pilot then takes over and hands back."""
    async with rig() as r:
        await _in_goto(r)
        r.ex.on_override(r.notice("stop"))
        r.fc.set_mode_external(CopterMode.BRAKE)
        await r.run(0.3)
        assert r.ex.over.stop_latch
        tid = new_id()
        r.ex.on_takeover(_takeover(r, tid, active=True), new_id())
        assert r.ex.over.stop_latch  # only Resume or landing lifts a STOP
        r.ex.on_takeover(_takeover(r, tid, active=False), new_id())
        assert await _resume_codes(r) == ("ONB_STOP_LATCHED",)


async def test_an_external_brake_is_never_a_takeover() -> None:
    """The ground takes over with LOITER; a BRAKE nobody asked for stays a stop."""
    async with rig() as r:
        await _in_goto(r)
        r.fc.set_mode_external(CopterMode.BRAKE)
        await r.run(0.3)
        r.ex.on_takeover(_takeover(r, new_id(), active=True), new_id())
        assert r.ex.over.stop_latch


async def test_another_mode_change_in_the_window_keeps_the_stop() -> None:
    """LOITER, then the flight controller moves on before the notice: not one clean
    take-over, so the latch stays."""
    async with rig() as r:
        await _in_goto(r)
        r.fc.set_mode_external(CopterMode.LOITER)
        await r.run(0.3)
        r.fc.set_mode_external(CopterMode.BRAKE)
        await r.run(0.3)
        r.ex.on_takeover(_takeover(r, new_id(), active=True), new_id())
        assert r.ex.over.stop_latch


async def test_return_home_refused_is_retried_until_it_flies_home() -> None:
    """SmartRTL and RTL refused: no dead end in RTL; the executor keeps asking."""
    async with rig() as r:
        await _in_goto(r)
        r.fc.refuse_modes = {int(CopterMode.SMART_RTL), int(CopterMode.RTL)}
        cmd = await r.send(r.order("return_home"), 1)
        assert r.ex.state == S.RTL
        await r.run(3)
        assert r.up.codes(cmd.command_id) == ("ONB_MODE_CHANGE_TIMEOUT",)
        assert "executor.return_home_failed" in r.up.kinds()
        # A new order is not answered "done" while the vehicle does not fly home.
        again = await r.send(r.order("return_home"), 1)
        assert r.up.stages(again.command_id)[-1] != "done"
        r.fc.refuse_modes = set()
        await r.run(8)
        assert r.fc.mode in (CopterMode.SMART_RTL, CopterMode.RTL)
        assert "executor.return_home_retry" in r.up.kinds()
        attempts = [d["attempt"] for t, _, d in r.up.events if t == "executor.return_home_retry"]
        assert attempts == list(range(1, len(attempts) + 1))
        await r.until(lambda: r.ex.state == S.LANDED, 300, step=0.05)
