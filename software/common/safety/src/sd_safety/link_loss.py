"""What the drone does when the SD-link goes quiet: the usual rule or the find hover.

:func:`choose_mode` is pure and shared: the onboard executor (``sd_onboard.executor.
link_loss``) acts on it, and the ground station runs the same function on the same inputs
to tell the panel what to expect (``VehicleLive.link_loss_plan``).

* **Normal** (no confirmed find nearby): the running step goes on, ``HOLD`` at
  ``LINK_LOSS_DETECT_S + SDLINK_GRACE_S`` (23 s), home at ``+ SDLINK_HOLD_S`` (53 s).
* **Find**: a confirmed find (a ``Keepout`` with ``source == "confirmed"``) lies within
  ``FIND_ENGAGE_RADIUS_M``: fly to the edge of its cylinder + ``FIND_HOVER_EDGE_MARGIN_M``
  on the line from the find towards the drone, face the find, hover until
  ``LINK_LOSS_DETECT_S + FIND_HOVER_S`` (123 s), then home.

Every time counts from the last inbound SD-link message.
"""

from __future__ import annotations

import math
from collections.abc import Sequence
from dataclasses import dataclass
from typing import Literal

from sd_safety.constants import (
    FIND_ENGAGE_RADIUS_M,
    FIND_HOVER_EDGE_MARGIN_M,
    FIND_HOVER_S,
    FIND_MOVE_DEADLINE_S,
    KEEPOUT_CLIMB_LINK_ALLOWANCE_S,
    LINK_LOSS_DETECT_S,
    SDLINK_GRACE_S,
    SDLINK_HOLD_S,
)
from sd_safety.targets import Plane, polygon_from_geojson
from sd_schemas.base import Id
from sd_schemas.charter import MissionCharter
from sd_schemas.executor import ExecutorState
from sd_schemas.link import Keepout

# Where the find rule may take over: a step that moves under the executor's own control.
# Not TAKEOFF (still climbing), RTL (already going home), HELD (a person steers) or
# FAILSAFE (the flight controller decides).
FIND_STATES = frozenset({ExecutorState.TRANSIT, ExecutorState.SURVEY, ExecutorState.HOLD})
# Closer than this to the find's centre there is no direction "from the find towards the
# drone"; the hover point then lies towards home.
_ON_CENTRE_M = 0.5

# Milliseconds after the loss (the last inbound message).
DETECT_MS = round(LINK_LOSS_DETECT_S * 1000)
HOLD_MS = round((LINK_LOSS_DETECT_S + SDLINK_GRACE_S) * 1000)
HOME_MS = round((LINK_LOSS_DETECT_S + SDLINK_GRACE_S + SDLINK_HOLD_S) * 1000)
FIND_MOVE_DEADLINE_MS = round((LINK_LOSS_DETECT_S + FIND_MOVE_DEADLINE_S) * 1000)
FIND_HOVER_UNTIL_MS = round((LINK_LOSS_DETECT_S + FIND_HOVER_S) * 1000)
# Home regardless, also while the keep-out climb is still under way (113 s).
HOME_CLIMBING_MS = HOME_MS + round(KEEPOUT_CLIMB_LINK_ALLOWANCE_S * 1000)


@dataclass(frozen=True, slots=True)
class NormalPlan:
    """The usual rule: finish the step, hover at 23 s, home at 53 s."""

    mode: Literal["normal"] = "normal"


@dataclass(frozen=True, slots=True)
class FindPlan:
    """Hover beside a confirmed find, facing it."""

    keepout_id: Id
    candidate_id: Id  # the keep-out's source_id
    find_lat: float
    find_lon: float
    radius_m: float
    hover_lat: float
    hover_lon: float
    alt_m: float  # above home, inside the charter's height band
    yaw_deg: float  # bearing from the hover point to the find, 0 = north, clockwise
    distance_m: float  # from the drone to the hover point, horizontally
    mode: Literal["find"] = "find"


Plan = NormalPlan | FindPlan


def inside_charter(lat: float, lon: float, charter: MissionCharter) -> bool:
    """The point lies in the charter's area grown by its fence buffer (the gate's geometry)."""
    plane = Plane(charter.home.lat, charter.home.lon)
    area = polygon_from_geojson(charter.area, plane)
    xy = plane.xy(lat, lon)
    return area.contains(xy) or area.boundary_distance(xy) <= charter.geofence_buffer_m


def nearest_confirmed(
    lat: float, lon: float, keepouts: Sequence[Keepout], plane: Plane
) -> tuple[Keepout, float] | None:
    """The confirmed find closest to (lat, lon) within ``FIND_ENGAGE_RADIUS_M``, and its
    horizontal distance; on a tie the lowest ``keepout_id``."""
    me = plane.xy(lat, lon)
    best: tuple[float, str, Keepout] | None = None
    for k in keepouts:
        if k.source != "confirmed":
            continue
        d = math.dist(me, plane.xy(k.lat, k.lon))
        if d > FIND_ENGAGE_RADIUS_M:
            continue
        key = (d, str(k.keepout_id), k)
        if best is None or key[:2] < best[:2]:
            best = key
    return None if best is None else (best[2], best[0])


def hover_point(
    *, lat: float, lon: float, find: Keepout, home_lat: float, home_lon: float, plane: Plane
) -> tuple[float, float, float]:
    """``P = C + (r + margin) * u``: on the cylinder's edge plus the margin, on the line from
    the find's centre ``C`` towards the drone (towards home when the drone is on ``C``).
    Returns (lat, lon, yaw_deg from P to C)."""
    c = plane.xy(find.lat, find.lon)
    me = plane.xy(lat, lon)
    dx, dy = me[0] - c[0], me[1] - c[1]
    if math.hypot(dx, dy) < _ON_CENTRE_M:
        h = plane.xy(home_lat, home_lon)
        dx, dy = h[0] - c[0], h[1] - c[1]
    norm = math.hypot(dx, dy)
    if norm < 1e-9:  # the find on home and the drone on both: any direction, north
        ux, uy = 0.0, 1.0
    else:
        ux, uy = dx / norm, dy / norm
    reach = find.radius_m + FIND_HOVER_EDGE_MARGIN_M
    p = (c[0] + reach * ux, c[1] + reach * uy)
    yaw = math.degrees(math.atan2(-ux, -uy)) % 360.0
    plat, plon = plane.latlon(*p)
    return plat, plon, yaw


def choose_mode(
    *,
    state: ExecutorState,
    lat: float,
    lon: float,
    alt_rel_home_m: float,
    nav_ok: bool,
    keepouts: Sequence[Keepout],
    charter: MissionCharter,
) -> Plan:
    """Normal (the usual 23/53 s) or the find hover. No side effects.

    The find hover needs all four: a state in ``FIND_STATES``; good navigation
    (``nav_ok``: 3D fix, EKF fine, neither the GPS nor the spoofing guard active); a
    confirmed find within ``FIND_ENGAGE_RADIUS_M``; and a hover point inside the charter.
    """
    if state not in FIND_STATES or not nav_ok:
        return NormalPlan()
    plane = Plane(charter.home.lat, charter.home.lon)
    found = nearest_confirmed(lat, lon, keepouts, plane)
    if found is None:
        return NormalPlan()
    find, _ = found
    plat, plon, yaw = hover_point(
        lat=lat,
        lon=lon,
        find=find,
        home_lat=charter.home.lat,
        home_lon=charter.home.lon,
        plane=plane,
    )
    if not inside_charter(plat, plon, charter):
        return NormalPlan()
    alt = min(max(alt_rel_home_m, charter.alt_min_agl_m), charter.alt_max_agl_m)
    return FindPlan(
        keepout_id=find.keepout_id,
        candidate_id=find.source_id,
        find_lat=find.lat,
        find_lon=find.lon,
        radius_m=find.radius_m,
        hover_lat=plat,
        hover_lon=plon,
        alt_m=alt,
        yaw_deg=yaw,
        distance_m=math.dist(plane.xy(lat, lon), plane.xy(plat, plon)),
    )


@dataclass(frozen=True, slots=True)
class PlanTimes:
    """When each step is due, in milliseconds on the clock ``lost_ms`` is on."""

    lost_ms: int
    hold_ms: int | None  # normal: hover from here
    home_ms: int  # home from here
    hover_until_ms: int | None  # find: hover until here (= home_ms)


def plan_times(plan: Plan, lost_ms: int) -> PlanTimes:
    if isinstance(plan, FindPlan):
        until = lost_ms + FIND_HOVER_UNTIL_MS
        return PlanTimes(lost_ms=lost_ms, hold_ms=None, home_ms=until, hover_until_ms=until)
    return PlanTimes(
        lost_ms=lost_ms, hold_ms=lost_ms + HOLD_MS, home_ms=lost_ms + HOME_MS, hover_until_ms=None
    )
