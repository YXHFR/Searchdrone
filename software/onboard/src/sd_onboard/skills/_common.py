"""Small helpers shared by the handlers."""

from __future__ import annotations

import math
from typing import Final

from sd_safety.targets import Plane
from sd_schemas.vehicle import VehicleState

__all__ = ["STILL_MS", "horizontal_m", "is_airborne", "still"]

STILL_MS: Final = 0.5  # slower than this counts as standing still
_AIRBORNE_STATES: Final = frozenset({"in_air", "takeoff", "landing"})
_AIRBORNE_ALT_M: Final = 0.5


def is_airborne(v: VehicleState | None) -> bool:
    """In the air for the executor: the flight controller says so, or armed above 0.5 m.
    (Wider than the gate's own test, which needs "in air" above 2 m before it lets an
    airborne-only order through.)"""
    if v is None:
        return False
    return v.landed_state in _AIRBORNE_STATES or (v.armed and v.alt_rel_home_m > _AIRBORNE_ALT_M)


def horizontal_m(plane: Plane, v: VehicleState, lat: float, lon: float) -> float:
    return math.dist(plane.xy(v.lat, v.lon), plane.xy(lat, lon))


def still(v: VehicleState) -> bool:
    return v.groundspeed_ms < STILL_MS
