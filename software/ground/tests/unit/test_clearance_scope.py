"""One "clear around the drone" is bound to its own mission and vehicle: a confirmation
for ``sim-02`` never lets ``sim-01`` take off, even at the same ground station."""

from __future__ import annotations

import sys
from pathlib import Path

# The gate's own test fixtures (a typical charter and ground context).
sys.path.insert(0, str(Path(__file__).resolve().parents[3] / "common/safety/tests/unit"))

from gate_fixtures import (  # pyright: ignore[reportMissingImports]
    CHARTER,
    MISSION,
    NOW,
    ground_ctx,
    req,
)

from sd_ground.services.clearance import Clearances
from sd_safety import evaluate
from sd_schemas.base import new_id


def test_clearance_scoped_to_vehicle() -> None:
    clearances = Clearances()
    other_mission = str(new_id())
    clearances.confirm(
        clearance_id=str(new_id()),
        origin="local",
        mission_id=other_mission,
        vehicle_id="sim-02",
        charter_id=str(CHARTER.charter_id),
        charter_version=CHARTER.version,
        now_ms=NOW,
    )
    assert clearances.get("sim-01") is None  # nothing for sim-01 in the ground's memory
    sim02 = clearances.get("sim-02")
    assert sim02 is not None
    # Even handed to sim-01's own gate, sim-02's confirmation is not valid there.
    d = evaluate(req("takeoff", {"alt_m": 40}), ground_ctx(clearance=sim02.state))
    assert d.decision == "REJECT"
    assert "CLEARANCE_REQUIRED" in d.reason_codes
    assert str(MISSION) != other_mission
