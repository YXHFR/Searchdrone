"""Small helpers shared by the handlers."""

from __future__ import annotations

import math
from typing import Final

from sd_safety.targets import Plane
from sd_schemas.vehicle import VehicleState

__all__ = ["STILL_MS", "horizontal_m", "still"]

STILL_MS: Final = 0.5  # slower than this counts as standing still


def horizontal_m(plane: Plane, v: VehicleState, lat: float, lon: float) -> float:
    return math.dist(plane.xy(v.lat, v.lon), plane.xy(lat, lon))


def still(v: VehicleState) -> bool:
    return v.groundspeed_ms < STILL_MS
