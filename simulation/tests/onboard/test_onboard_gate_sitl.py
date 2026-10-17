"""The onboard computer's own gate, against ArduCopter SITL: a test ground station with
no gate of its own (``FakeGround``) lets an order through that leaves the charter; the
executor refuses it with its own gate decision (which the ground station logs as
``gate.mismatch``) and nothing is sent to the flight controller."""

from __future__ import annotations

import asyncio
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent))

from test_onboard_sitl import (  # pyright: ignore[reportMissingImports]
    AREA_HALF_M,
    Sim,
    _install,
    _takeoff,
    flight,
    sim,
)

from sd_schemas.executor import ExecutorState as S
from sd_sim import paths

pytestmark = [
    pytest.mark.sitl,
    pytest.mark.slow,
    pytest.mark.skipif(not paths.arducopter_bin().is_file(), reason="ArduCopter SITL is not built"),
]
__all__ = ["sim"]


@pytest.mark.budget_s(150)
async def test_onboard_gate_refuses_what_the_ground_let_through_on_sitl(sim: Sim) -> None:
    async with flight(sim) as f:
        await _install(f)
        await _takeoff(f, 30)
        await f.until(lambda: f.ex.state == S.HOLD, 30, "takeoff done")
        before = len(f.sent)
        lat, lon = f.at(AREA_HALF_M + 60, 0)  # outside the charter area
        cmd = await f.order("goto", {"lat": lat, "lon": lon, "alt_m": 30})
        ack = await f.stage(cmd, {"accepted", "running"}, 10)
        assert ack.stage == "rejected", ack
        assert ack.gate is not None and ack.gate.side == "onboard"
        assert "OUTSIDE_CHARTER" in ack.gate.reason_codes
        await asyncio.sleep(5)
        assert f.steering_since(before) == []
        assert f.ex.state == S.HOLD
