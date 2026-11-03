"""Small helpers shared by the handlers."""

from __future__ import annotations

import math
from collections.abc import Sequence
from typing import Final

from sd_safety.keepout import Circle
from sd_safety.targets import Plane, Xy
from sd_schemas.vehicle import VehicleState

__all__ = [
    "KEEPOUT_CLIMB_MARGIN_M",
    "STILL_MS",
    "STOP_WAIT_S",
    "climb_alt",
    "horizontal_m",
    "is_airborne",
    "over_a_person",
    "still",
]

STILL_MS: Final = 0.5  # slower than this counts as standing still
STOP_WAIT_S: Final = 12.0  # a drone never reported still by then goes on from where it is
_AIRBORNE_STATES: Final = frozenset({"in_air", "takeoff", "landing"})
_AIRBORNE_ALT_M: Final = 0.5
# A cylinder has no height of its own; over a person with no safe way out the drone
# takes its top as its radius above home (as far up as out) and climbs this much above.
KEEPOUT_CLIMB_MARGIN_M: Final = 10.0


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


def over_a_person(here: Xy, cyl: Sequence[Circle]) -> bool:
    """Whether ``here`` lies inside any of the cylinders (over a person)."""
    return any(math.dist(here, c) < r for c, r in cyl)


def climb_alt(here: Xy, cyl: Sequence[Circle], alt_m: float, alt_max_m: float) -> float | None:
    """How high (above home) a drone at ``here``, ``alt_m`` up, climbs over the cylinders
    it is inside: their top (the largest radius) plus ``KEEPOUT_CLIMB_MARGIN_M``, at most
    ``alt_max_m``; where it is when it is that high already; None when it is inside none,
    or lower than that and cannot climb (``alt_max_m`` is no higher than it is)."""
    over = [r for c, r in cyl if over_a_person(here, [(c, r)])]
    if not over:
        return None
    want = max(over) + KEEPOUT_CLIMB_MARGIN_M
    if alt_m >= want:
        return alt_m
    top = min(want, alt_max_m)
    return top if top > alt_m else None
