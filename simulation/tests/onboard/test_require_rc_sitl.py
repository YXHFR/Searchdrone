"""The RC transmitter rule for real flight, against ArduCopter SITL (builtin physics).

The simulator loads the real flight profile (``searchdrone-safety-flight.parm``: the RC
failsafe on), and the onboard executor requires the RC link (``require_rc``). The
simulator's RC input is cut with ``SIM_RC_FAIL`` (the transmitter switched off):

* without the RC link the preflight check fails and ``takeoff`` is refused with
  ``ONB_RC_MISSING``; the motors are never armed;
* with the transmitter back on, the same mission takes off.
"""

from __future__ import annotations

import asyncio
import sys
from collections.abc import Iterator
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent))

from test_onboard_sitl import (  # pyright: ignore[reportMissingImports]
    REPO,
    Sim,
    _arms,
    _install,
    _takeoff,
    flight,
    simulator,
)

from sd_schemas.base import new_id
from sd_schemas.link import ClearanceConfirmation
from sd_sim import paths
from sd_sim.sitl import FLIGHT_SAFETY_PARAM_FILE

pytestmark = [
    pytest.mark.sitl,
    pytest.mark.slow,
    pytest.mark.skipif(not paths.arducopter_bin().is_file(), reason="ArduCopter SITL is not built"),
]

FLIGHT_PARM = REPO / "simulation/sitl/params" / FLIGHT_SAFETY_PARAM_FILE


@pytest.fixture
def flight_sim() -> Iterator[Sim]:
    with simulator(FLIGHT_SAFETY_PARAM_FILE) as s:
        yield s


@pytest.mark.budget_s(200)
async def test_takeoff_needs_the_rc_transmitter_on_sitl(flight_sim: Sim) -> None:
    sim = flight_sim
    async with flight(sim, FLIGHT_PARM, require_rc=True) as f:
        await _install(f)
        assert f.ex.preflight["PF12"] is True

        # The pilot switches the transmitter off: the flight controller loses the RC link.
        await asyncio.to_thread(sim.harness.set_sim_param, "SIM_RC_FAIL", 1)
        await f.until(lambda: f.ex.preflight.get("PF12") is False, 15, "PF12 fails without RC")
        assert f.charter is not None and f.session is not None
        c = f.charter
        await f.session.send(
            ClearanceConfirmation(
                clearance_id=new_id(),
                mission_id=c.mission_id,
                vehicle_id="sim-01",
                charter_id=c.charter_id,
                charter_version=c.version,
                valid_for_ms=120_000,
                confirmed_by="operator:pilot",
                origin="local",
            ),
            ttl_ms=5000,
        )
        await f.until(lambda: f.ex.snapshot().clearance_available, 10, "clearance onboard")
        cmd = await f.order("takeoff", {"alt_m": 20})
        ack = await f.stage(cmd, {"accepted", "running"}, 10)
        assert ack.stage == "rejected" and ack.reason_codes == ["ONB_RC_MISSING"], ack
        await asyncio.sleep(5)
        assert _arms(f) == 0
        assert not f.onboard.telemetry.armed

        # Transmitter on again: the same mission takes off.
        await asyncio.to_thread(sim.harness.set_sim_param, "SIM_RC_FAIL", 0)
        await f.until(lambda: f.ex.preflight_ok(), 15, "preflight with RC")
        await _takeoff(f, 20)
        assert _arms(f) == 1
