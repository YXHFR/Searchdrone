"""Keep-out cylinders: the gate's check of a route and the executor's way around them.

A route may never enter a cylinder (``sd_schemas.keepout.leg_enters``): a leg that starts
inside one may only leave it without going deeper. An orbit's ring keeps
``ORBIT_KEEPOUT_MARGIN_M`` from every cylinder (radius at least the cylinder's + 10 m
around one at its centre).

:func:`reroute` replaces every leg that crosses a cylinder with a way around it: along
the two tangents to a circle ``KEEPOUT_ARC_MARGIN_M`` outside the cylinder and an arc of
corners between them, on the side that stays inside the charter (the shorter if both
do). Corners lie just outside that circle, so no straight leg between them comes
closer. A route that starts inside a cylinder first leaves it, along the given direction
(the leg being flown) or straight out from the centre. Pure functions in the plane of
``sd_safety.targets.Plane``.
"""

from __future__ import annotations

import math
from collections.abc import Sequence
from itertools import pairwise
from typing import Final

from sd_safety.targets import Plane, Polygon, Xy
from sd_schemas.base import LatLon
from sd_schemas.keepout import Keepout, leg_enters
from sd_schemas.skills import OrbitRing

__all__ = [
    "KEEPOUT_ARC_MARGIN_M",
    "ORBIT_KEEPOUT_MARGIN_M",
    "Circle",
    "circles",
    "reroute",
    "ring_violation",
    "route_violation",
    "xy_route_violation",
]

KEEPOUT_ARC_MARGIN_M: Final = 5.0  # a way around keeps this outside the cylinder
ORBIT_KEEPOUT_MARGIN_M: Final = 10.0  # an orbit's ring keeps this from every cylinder
_ARC_STEP: Final = math.radians(15.0)  # the most an arc turns between two corners
_MAX_DETOURS: Final = 100  # more ways around than this: overlapping cylinders, give up
_EXIT_EPS_M: Final = 0.5  # past the circle when leaving a cylinder

Circle = tuple[Xy, float]  # centre, radius


def circles(keepouts: Sequence[Keepout], plane: Plane) -> list[Circle]:
    return [(plane.xy(k.lat, k.lon), k.radius_m) for k in keepouts]


def xy_route_violation(route: Sequence[Xy], cyl: Sequence[Circle]) -> int | None:
    """The index into ``cyl`` of the first cylinder the route enters, or None. A route of
    one point is a point: inside a cylinder is a violation."""
    if len(route) == 1:
        p = route[0]
        return next((i for i, (c, r) in enumerate(cyl) if math.dist(p, c) < r), None)
    for a, b in pairwise(route):
        for i, (c, r) in enumerate(cyl):
            if leg_enters(a, b, c, r):
                return i
    return None


def route_violation(
    route: Sequence[LatLon], keepouts: Sequence[Keepout], plane: Plane
) -> Keepout | None:
    """The first cylinder the route (corners, current position first) enters, or None."""
    if not route or not keepouts:
        return None
    found = xy_route_violation([plane.point(p) for p in route], circles(keepouts, plane))
    return None if found is None else keepouts[found]


def ring_violation(ring: OrbitRing, keepouts: Sequence[Keepout], plane: Plane) -> Keepout | None:
    """The first cylinder within ``ORBIT_KEEPOUT_MARGIN_M`` of the ring's band, or None."""
    centre = plane.xy(ring.lat, ring.lon)
    for k in keepouts:
        d = math.dist(centre, plane.xy(k.lat, k.lon))
        reach = k.radius_m + ORBIT_KEEPOUT_MARGIN_M
        if ring.outer_m > d - reach and ring.inner_m < d + reach:
            return k
    return None


# --- the way around --------------------------------------------------------------------


def reroute(
    route: Sequence[Xy],
    cyl: Sequence[Circle],
    *,
    area: Polygon | None = None,
    exit_dir: Xy | None = None,
) -> list[Xy] | None:
    """``route`` (current position first) with every crossing of a cylinder replaced by
    a way around it, or None when there is none that stays inside ``area``.

    Corners inside a cylinder are dropped (the rest of the route never goes back in).
    When the position is inside one and the first leg would go deeper, the route first
    leaves it: along ``exit_dir`` when given (the leg being flown), else straight out
    from the centre. Every leg after that is clear of the cylinders, and the whole route
    must lie inside ``area``."""
    if not route:
        return []
    here = route[0]
    out = [here]
    exit_point = _exit(here, route[1] if len(route) > 1 else None, cyl, exit_dir)
    if exit_point is not None:
        out.append(exit_point)
    out += [p for p in route[1:] if not any(math.dist(p, c) < r for c, r in cyl)]
    start = 0 if exit_point is None else 1  # the leg out of a cylinder is not re-checked
    i, detours = start, 0
    while i < len(out) - 1:
        a, b = out[i], out[i + 1]
        hit = _first_hit(a, b, cyl)
        if hit is None:
            i += 1
            continue
        detours += 1
        if detours > _MAX_DETOURS:
            return None
        way = _around(a, b, cyl[hit], area)
        if way is None:
            return None
        out[i + 1 : i + 1] = way
    if len(out) == 1:
        return out  # nothing left to fly
    # Every leg from ``start`` on is now clear of the cylinders (the loop above); the
    # area is checked for the legs kept as they were too.
    if area is not None and any(area.leg_outside_by(a, b) > 0.0 for a, b in pairwise(out[start:])):
        return None
    return out


def _exit(here: Xy, nxt: Xy | None, cyl: Sequence[Circle], exit_dir: Xy | None) -> Xy | None:
    """Where to leave the cylinders ``here`` is in, or None if it is in none or the first
    leg leaves without going deeper."""
    inside = [(c, r) for c, r in cyl if math.dist(here, c) < r]
    if not inside:
        return None
    if nxt is not None and not any(leg_enters(here, nxt, c, r) for c, r in inside):
        return None
    if exit_dir is not None and math.hypot(*exit_dir) > 1e-9:
        n = math.hypot(*exit_dir)
        u = (exit_dir[0] / n, exit_dir[1] / n)
    else:
        c0 = inside[0][0]
        dx, dy = here[0] - c0[0], here[1] - c0[1]
        n = math.hypot(dx, dy)
        u = (0.0, 1.0) if n < 1e-9 else (dx / n, dy / n)
    far = 0.0
    for c, r in inside:
        # here + t*u leaves the circle (c, r + margin) at the larger root of |p - c| = R.
        ox, oy = here[0] - c[0], here[1] - c[1]
        big_r = r + KEEPOUT_ARC_MARGIN_M
        bq = ox * u[0] + oy * u[1]
        cq = ox * ox + oy * oy - big_r * big_r
        far = max(far, -bq + math.sqrt(bq * bq - cq) + _EXIT_EPS_M)
    return (here[0] + far * u[0], here[1] + far * u[1])


def _first_hit(a: Xy, b: Xy, cyl: Sequence[Circle]) -> int | None:
    """The cylinder the leg reaches first (by its closest approach along the leg)."""
    best: tuple[float, int] | None = None
    dx, dy = b[0] - a[0], b[1] - a[1]
    seg2 = dx * dx + dy * dy
    for i, (c, r) in enumerate(cyl):
        if not leg_enters(a, b, c, r):
            continue
        t = 0.0 if seg2 == 0.0 else -((a[0] - c[0]) * dx + (a[1] - c[1]) * dy) / seg2
        if best is None or t < best[0]:
            best = (t, i)
    return None if best is None else best[1]


def _around(a: Xy, b: Xy, circle: Circle, area: Polygon | None) -> list[Xy] | None:
    """The corners of the way around ``circle`` from ``a`` to ``b`` (both ends excluded)."""
    options: list[tuple[bool, float, list[Xy]]] = []
    for sign in (1.0, -1.0):
        corners = _arc(a, b, circle, sign)
        path = [a, *corners, b]
        inside = area is None or all(area.leg_outside_by(p, q) <= 0.0 for p, q in pairwise(path))
        length = sum(math.dist(p, q) for p, q in pairwise(path))
        options.append((inside, length, corners))
    options.sort(key=lambda o: (not o[0], o[1]))
    inside, _, corners = options[0]
    return corners if inside else None


def _arc(a: Xy, b: Xy, circle: Circle, sign: float) -> list[Xy]:
    """Counter-clockwise (``sign`` 1) or clockwise (-1): tangent from ``a`` to the circle
    of radius r + margin, along it, tangent to ``b``."""
    c, r = circle
    big_r = r + KEEPOUT_ARC_MARGIN_M

    def polar(p: Xy) -> tuple[float, float]:
        dx, dy = p[0] - c[0], p[1] - c[1]
        d = math.hypot(dx, dy)
        return math.atan2(dy, dx), (math.acos(big_r / d) if d > big_r else 0.0)

    phi_a, alpha = polar(a)
    phi_b, beta = polar(b)
    first = phi_a + sign * alpha
    last = phi_b - sign * beta
    sweep = (sign * (last - first)) % (2 * math.pi)
    n = max(1, math.ceil(sweep / _ARC_STEP))
    half = sweep / n / 2
    corner_r = big_r / math.cos(half)
    out: list[Xy] = []
    for k in range(n + 1):
        t = first + sign * sweep * k / n
        out.append((c[0] + corner_r * math.cos(t), c[1] + corner_r * math.sin(t)))
    return out
