"""A pressed "Ta över" whose LOITER still waits counts as by hand for the gate and the AI."""

from __future__ import annotations

import datetime as dt

import pytest

from sd_ground.mav.thread import MavSnapshot
from sd_ground.services.gate_context import BrainRate, build_gate_context
from sd_ground.services.situation import _vehicle  # pyright: ignore[reportPrivateUsage]
from sd_mav.vehicle import FcTelemetry
from sd_schemas.api import Mission
from sd_schemas.base import new_id
from sd_schemas.mission import MissionState
from sd_schemas.safety import SafetyLimits

NOW = dt.datetime(2026, 1, 1, tzinfo=dt.UTC)


def _snap(*, manual: bool, pending: bool) -> MavSnapshot:
    return MavSnapshot(
        vehicle_id="sim-01",
        connected=True,
        telemetry=FcTelemetry(),
        now_mono_ms=1_000,
        in_flight=True,
        onboard_alive=True,
        airborne_since=None,
        manual=manual,
        watchdog_phase="ok",
        takeover_pending=pending,
    )


def _mission() -> Mission:
    return Mission(
        id=new_id(),
        vehicle_id="sim-01",
        title="test",
        status=MissionState.ACTIVE,
        paused_by=None,
        returning_cause=None,
        state_seq=1,
        limits=SafetyLimits(),
        active_charter_id=None,
        autonomy_mode=None,
        ai_enabled=True,
        created_by="operator:demo",
        created_at=NOW,
        activated_at=NOW,
        landed_at=None,
        closed_at=None,
        closed_reason=None,
        notes=None,
    )


CASES = [(False, False, False), (True, False, True), (False, True, True), (True, True, True)]


@pytest.mark.parametrize(("manual", "pending", "by_hand"), CASES)
def test_the_gate_sees_a_waiting_takeover_as_manual_control(
    manual: bool, pending: bool, by_hand: bool
) -> None:
    ctx = build_gate_context(
        now_ms=1_000,
        registry={},
        mission=_mission(),
        charter=None,
        charter_installed=False,
        fence_verified=False,
        mav=_snap(manual=manual, pending=pending),
        snapshot=None,
        snapshot_received_ms=None,
        sdlink_up=True,
        latch=None,
        clearance=None,
        rate=BrainRate(),
        brain_pending_approvals=0,
        operator_present=True,
    )
    assert ctx.manual_control is by_hand


@pytest.mark.parametrize(("manual", "pending", "by_hand"), CASES)
def test_the_ai_sees_a_waiting_takeover_as_manual_control(
    manual: bool, pending: bool, by_hand: bool
) -> None:
    vehicle = _vehicle(_snap(manual=manual, pending=pending), None, True, None)
    assert vehicle.manual_control is by_hand
