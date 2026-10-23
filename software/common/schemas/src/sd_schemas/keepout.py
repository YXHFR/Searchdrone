"""Keep-out cylinders around possible persons, and the geometry the schemas need of them.

A ``Keepout`` is a vertical cylinder no route may enter: ``PERSON_KEEPOUT_M`` plus the
position's ``sigma95_m``. The gate (``sd_safety.keepout``), the onboard executor's
reflex and the ``inspect_detection`` plan all build on the two functions here, so the
ground and the drone agree on where an inspection goes:

* :func:`leg_enters`: whether a straight leg comes into a cylinder (leaving one the leg
  starts inside is allowed, as long as it never goes deeper);
* :func:`inspect_geometry`: the point, height and heading of a closer look at a target
  with a camera that looks straight down.

Everything is pure and works in the local east/north plane of ``sd_schemas.geo``.
"""

from __future__ import annotations

import math
from collections.abc import Sequence
from dataclasses import dataclass
from typing import Final, Literal

from pydantic import Field

from sd_schemas.base import Id, SdModel
from sd_schemas.geo import local_xy, plane_offset

__all__ = [
    "INSPECT_AIM_DEG",
    "INSPECT_SPEED_MS",
    "INSPECT_STANDOFF_MARGIN_M",
    "INSPECT_VIEW_DEG",
    "INSPECT_VIEW_MAX_DEG",
    "InspectGeometry",
    "Keepout",
    "inspect_geometry",
    "leg_enters",
]

Xy = tuple[float, float]

INSPECT_VIEW_DEG: Final = 25.0  # the target at most this far from straight down
INSPECT_AIM_DEG: Final = 24.0  # the height aims a degree steeper: the hover's error fits
INSPECT_VIEW_MAX_DEG: Final = 28.0  # beyond this at the highest allowed height: refused
INSPECT_STANDOFF_MARGIN_M: Final = 10.0  # the inspection point keeps this off the cylinder
INSPECT_SPEED_MS: Final = 5.0  # the most an inspection flies at
_INSPECT_DIRECTIONS: Final = 24  # tried around the target, nearest the drone's side first
_LEAVE_COS: Final = 1e-6  # leaving square to the centre: rounding is not "deeper"
_ON_TARGET_M: Final = 1.0  # closer than this there is no "side": the point lies north


class Keepout(SdModel):
    """A no-fly cylinder around a possible person."""

    keepout_id: Id
    lat: float = Field(ge=-90, le=90)
    lon: float = Field(ge=-180, le=180)
    radius_m: float = Field(gt=0, le=500)  # person_keepout_m + sigma95_m
    source: Literal["detection", "candidate", "confirmed"]
    source_id: Id  # the detection_id or candidate_id


def leg_enters(a: Xy, b: Xy, c: Xy, r: float) -> bool:
    """Whether the straight leg ``a`` -> ``b`` comes inside the circle (``c``, ``r``).

    A leg that starts inside may leave, but never go deeper: it is refused when it heads
    towards the centre at all (square to it, the distance still grows; a rounding error's
    worth of angle is not counted). Touching the edge from outside is allowed."""
    ax, ay = a[0] - c[0], a[1] - c[1]
    dx, dy = b[0] - a[0], b[1] - a[1]
    if math.hypot(ax, ay) < r:
        return dx * ax + dy * ay < -_LEAVE_COS * math.hypot(dx, dy) * math.hypot(ax, ay)
    seg2 = dx * dx + dy * dy
    t = 0.0 if seg2 == 0.0 else max(0.0, min(1.0, -(ax * dx + ay * dy) / seg2))
    return math.hypot(ax + t * dx, ay + t * dy) < r


@dataclass(frozen=True, slots=True)
class InspectGeometry:
    lat: float
    lon: float
    alt_m: float  # above home
    yaw_deg: float  # 0 = north, clockwise: the target lies along the picture's long side
    standoff_m: float  # horizontal distance from the point to the target
    view_deg: float  # the target's angle from straight down


def inspect_geometry(
    *,
    target_lat: float,
    target_lon: float,
    target_radius_m: float,
    standoff_m: float,
    here_lat: float,
    here_lon: float,
    alt_min_m: float,
    alt_max_m: float,
    keepouts: Sequence[Keepout] = (),
) -> InspectGeometry | None:
    """Where to look at a target from, or ``None`` when no height in ``alt_min_m`` ..
    ``alt_max_m`` keeps it within ``INSPECT_VIEW_MAX_DEG`` of straight down.

    The standoff is the order's, but at least ``INSPECT_STANDOFF_MARGIN_M`` outside every
    cylinder that holds the target (its own ``target_radius_m`` included). The height
    puts the target ``INSPECT_AIM_DEG`` from straight down, within the band (so a drone
    a metre off the point still sees it within ``INSPECT_VIEW_DEG``). The point
    lies on the drone's side of the target; if a straight leg there would enter another
    cylinder, or the point lies in one, the next direction around is tried (15 degrees
    at a time, nearest first). The heading puts the target to the drone's right, along
    the picture's long side (the camera's x axis is the drone's right)."""
    reach = target_radius_m
    t = (0.0, 0.0)
    near = [(local_xy(target_lat, target_lon, k.lat, k.lon), k.radius_m) for k in keepouts]
    for c, r in near:
        d = math.hypot(*c)
        if d < r:
            reach = max(reach, d + r)
    standoff = max(standoff_m, reach + INSPECT_STANDOFF_MARGIN_M)
    if alt_max_m < alt_min_m:
        return None
    alt = min(max(standoff / math.tan(math.radians(INSPECT_AIM_DEG)), alt_min_m), alt_max_m)
    view = math.degrees(math.atan2(standoff, alt))
    if view > INSPECT_VIEW_MAX_DEG + 1e-9:
        return None
    here = local_xy(target_lat, target_lon, here_lat, here_lon)
    side = math.atan2(here[0], here[1]) if math.hypot(*here) >= _ON_TARGET_M else 0.0
    step = 2 * math.pi / _INSPECT_DIRECTIONS
    tries = [0] + [s * i for i in range(1, _INSPECT_DIRECTIONS // 2 + 1) for s in (1, -1)]
    for i in tries[:_INSPECT_DIRECTIONS]:
        bearing = side + i * step
        p = (standoff * math.sin(bearing), standoff * math.cos(bearing))
        if any(math.dist(p, c) < r or leg_enters(here, p, c, r) for c, r in near):
            continue
        lat, lon = plane_offset(target_lat, target_lon, p[0], p[1])
        to_target = math.degrees(math.atan2(t[0] - p[0], t[1] - p[1]))
        return InspectGeometry(
            lat=lat,
            lon=lon,
            alt_m=alt,
            yaw_deg=(to_target - 90.0) % 360.0,
            standoff_m=standoff,
            view_deg=view,
        )
    return None
