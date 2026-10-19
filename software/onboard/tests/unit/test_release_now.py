"""Resume (Fortsätt) takes effect at once, even within the notice window after STOP."""

from __future__ import annotations

import logging
from uuid import UUID

import pytest
from onboard_helpers import Rig, at, rig
from pymavlink.dialects.v20 import ardupilotmega as mavlink

from sd_mav.modes import CopterMode
from sd_mav.vehicle import ControlNotAllowedError, SendOrigin
from sd_schemas.base import new_id
from sd_schemas.executor import ExecutorState as S
from sd_schemas.link import ManualTakeover, OperatorRelease


async def _in_goto(r: Rig) -> None:
    await r.takeoff()
    lat, lon = at(150, 0)
    await r.send(r.order("goto", {"lat": lat, "lon": lon, "alt_m": 40, "speed_ms": 8}))
    await r.run(3)
    assert r.ex.state == S.TRANSIT


def _release(r: Rig) -> UUID:
    ref = new_id()
    r.ex.on_release(OperatorRelease(action_seq=r._seq, operator_id="operator:pilot"), ref)
    return ref


def _mode_writes(r: Rig, since: int) -> list[int]:
    """The modes asked for with DO_SET_MODE since ``since``, in order."""
    return [
        int(m.param2)
        for m in r.vehicle.sent[since:]
        if m.get_type() == "COMMAND_LONG" and int(m.command) == mavlink.MAV_CMD_DO_SET_MODE
    ]


async def _stopped(r: Rig) -> None:
    """STOP from the ground: the notice, and the ground's MAVLink puts the drone in BRAKE."""
    r.ex.on_override(r.notice("stop"))
    r.fc.set_mode_external(CopterMode.BRAKE)
    await r.run(0.1)
    assert r.ex.state == S.HELD and r.ex.held_by == "operator_stop"


@pytest.mark.parametrize("after_s", [0.5, 5.0])
async def test_release_after_stop_sends_loiter(after_s: float) -> None:
    async with rig() as r:
        await _in_goto(r)
        await _stopped(r)
        await r.run(after_s - 0.1)
        assert r.fc.mode == CopterMode.BRAKE
        before = len(r.vehicle.sent)
        ref = _release(r)
        await r.run(0.5)
        assert r.up.stages(ref)[-1] == "done"
        assert r.ex.state == S.HOLD and not r.ex.over.stop_latch
        assert _mode_writes(r, before) == [int(CopterMode.LOITER)]
        assert r.fc.mode == CopterMode.LOITER
        assert not r.ex.auth.notice_recent(r.ex.now_ms())


async def test_fallback_never_brakes_after_release() -> None:
    async with rig() as r:
        await _in_goto(r)
        await _stopped(r)
        await r.run(0.4)
        before = len(r.vehicle.sent)
        _release(r)
        task = r.ex.fallback._task  # pyright: ignore[reportPrivateUsage]
        await r.run(6)  # well past OVERRIDE_ONBOARD_FALLBACK_S
        assert task is not None and task.done()
        assert int(CopterMode.BRAKE) not in _mode_writes(r, before)
        assert r.fc.mode == CopterMode.LOITER
        assert r.ex.fallback.used == 0
        assert "executor.override_fallback_used" not in r.up.kinds()


async def test_hand_back_also_stops_the_fallback() -> None:
    """Handing back after a takeover keeps the latch but ends the STOP's fallback chain."""
    async with rig() as r:
        await _in_goto(r)
        r.ex.on_override(r.notice("stop"))
        task = r.ex.fallback._task  # pyright: ignore[reportPrivateUsage]
        assert task is not None and not task.done()
        tid = new_id()
        r.ex.on_takeover(_takeover(r, tid, active=True), new_id())
        r.fc.set_mode_external(CopterMode.LOITER)
        await r.run(0.3)
        r.ex.on_takeover(_takeover(r, tid, active=False), new_id())
        await r.run(0.1)
        assert r.ex.state == S.HOLD and r.ex.over.stop_latch
        assert task.done()


def _takeover(r: Rig, tid: UUID, *, active: bool) -> ManualTakeover:
    return ManualTakeover(
        takeover_id=tid, active=active, action_seq=r.next_seq(), operator_id="operator:pilot"
    )


async def test_refused_mode_write_is_logged(caplog: pytest.LogCaptureFixture) -> None:
    async with rig() as r:
        await _in_goto(r)
        await _stopped(r)
        caplog.set_level(logging.WARNING, logger="sd_onboard")
        await r.ex._loiter(SendOrigin.SKILL)  # pyright: ignore[reportPrivateUsage]
        assert any(
            "LOITER refused" in rec.getMessage() and "control_held" in rec.getMessage()
            for rec in caplog.records
        )

        async def refused() -> None:
            raise ControlNotAllowedError("override_notice")

        caplog.clear()
        r.ex._side(refused())  # pyright: ignore[reportPrivateUsage]
        await r.run(0.1)
        assert any(
            rec.levelno == logging.WARNING and "override_notice" in rec.getMessage()
            for rec in caplog.records
        )


async def test_stop_again_after_release_still_blocks() -> None:
    async with rig() as r:
        await _in_goto(r)
        await _stopped(r)
        await r.run(0.3)
        _release(r)
        await r.run(0.3)
        assert r.ex.state == S.HOLD and r.fc.mode == CopterMode.LOITER
        await _stopped(r)  # a new STOP within 2 s of the first one
        assert r.ex.auth.notice_recent(r.ex.now_ms())
        assert r.ex.send_gate(SendOrigin.SKILL) is not None
        before = len(r.vehicle.sent)
        lat, lon = at(30, 0)
        cmd = await r.send(r.order("goto", {"lat": lat, "lon": lon, "alt_m": 40}))
        assert r.up.codes(cmd.command_id) == ("ONB_CONTROL_HELD",)
        await r.run(1.4)
        assert r.steering_sent(before, allow_fallback=True) == []
        assert r.ex.state == S.HELD and r.ex.over.stop_latch
