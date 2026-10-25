"""Keep-out cylinders: the gate's check of a route and the executor's way around them.

A route may never enter a cylinder (``sd_schemas.keepout.leg_enters``): a leg that starts
inside one may only leave it without going deeper. An orbit's ring keeps
``ORBIT_KEEPOUT_MARGIN_M`` from every cylinder (radius at least the cylinder's + 10 m
around one at its centre).

:func:`reroute` replaces every leg that crosses a cylinder with a way around it: along
the two tangents to a circle ``KEEPOUT_ARC_MARGIN_M`` outside the cylinder and an arc of
corners between them, on the side that stays inside the charter (the shorter if both
do). Corners lie just outside that circle, so no straight leg between them comes
closer. Cylinders whose margins overlap (one person seen again and again) are passed as
one group (``_GROUP_MAX`` at most, ``_GROUP_WORK_MAX`` leg checks in one reroute): the
shortest way through the corners around each of them along legs that come no closer
than the margin (an end already inside it: no closer than that end) and stay inside the
charter. A leg of a way around that comes closer than the margin to another cylinder
gets a way around that one too, so every way around keeps the margin from every
cylinder, not only from the ones it was made for. A route that starts inside cylinders
first leaves them along a straight leg that goes deeper into none of them and enters no
other: the given direction (the leg being flown), away from their centres, or the
nearest direction that works. Pure functions in the plane of ``sd_safety.targets.Plane``.
"""

from __future__ import annotations

import math
from collections.abc import Sequence
from itertools import pairwise
from typing import Final

from sd_safety.targets import Plane, Polygon, Xy, point_segment_distance
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
_MAX_DETOURS: Final = 100  # a backstop: a route that keeps needing ways around, given up
# Overlapping cylinders passed as one group: at most this many (the search grows with the
# square of their corners); a larger cluster has no way around.
_GROUP_MAX: Final = 20
# The grouped ways around in one reroute check at most this many legs against a cylinder
# (about a tenth of a second here, well under the reflex's reaction time on board); more
# and there is no way around: the drone holds instead of planning on and on.
_GROUP_WORK_MAX: Final = 50_000
_MARGIN_EPS_M: Final = 1e-6  # a leg along a margin polygon's edge touches the margin
_EXIT_EPS_M: Final = 0.5  # past the circle when leaving a cylinder
_EXIT_STEP: Final = math.radians(5.0)  # the directions tried when leaving cylinders

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
    When the position is inside cylinders and the first leg would go deeper, the route
    first leaves them (``_exit``): along ``exit_dir`` when given (the leg being flown)
    if that works, else away from their centres. Every leg after that is clear of the
    cylinders, and the whole route must lie inside ``area``."""
    if not route:
        return []
    here = route[0]
    out = [here]
    way_out = _exit(here, route[1] if len(route) > 1 else None, cyl, exit_dir)
    if way_out is None:
        return None
    out += way_out
    out += [p for p in route[1:] if not any(math.dist(p, c) < r for c, r in cyl)]
    made = [False] * len(out)  # a corner of a way around (its legs keep the margin)
    start = len(way_out)  # the leg out of the cylinders is clear already (``_exit``)
    i, detours = start, 0
    work = [_GROUP_WORK_MAX]  # what the grouped ways around may still check
    while i < len(out) - 1:
        a, b = out[i], out[i + 1]
        hit = _first_hit(a, b, cyl, margin=made[i] or made[i + 1])
        if hit is None:
            i += 1
            continue
        detours += 1
        if detours > _MAX_DETOURS:
            return None
        group = _group(cyl, hit)
        if len(group) > _GROUP_MAX:
            return None
        if len(group) == 1:
            way = _around(a, b, cyl[hit], area)
        else:
            way = _around_group(a, b, group, area, work)
        if way is None:
            return None
        out[i + 1 : i + 1] = way
        made[i + 1 : i + 1] = [True] * len(way)
    if len(out) == 1:
        return out  # nothing left to fly
    # Every leg from ``start`` on is now clear of the cylinders (the loop above); the
    # area is checked for every leg, the way out and the legs kept as they were too.
    if area is not None and any(area.leg_outside_by(a, b) > 0.0 for a, b in pairwise(out)):
        return None
    return out


def _exit(here: Xy, nxt: Xy | None, cyl: Sequence[Circle], exit_dir: Xy | None) -> list[Xy] | None:
    """The way out of the cylinders ``here`` is in: no corner when it is in none or the
    first leg leaves without going deeper, else one corner past all of them; None when
    no straight way out goes deeper into none of them and enters no other.

    The direction is the first that works of: ``exit_dir``, away from the centres
    together, straight out from each centre, square to each (nearest the next corner
    first), then every ``_EXIT_STEP`` around, nearest the first of these first."""
    inside = [(c, r) for c, r in cyl if math.dist(here, c) < r]
    if not inside:
        return []
    if nxt is not None and not any(leg_enters(here, nxt, c, r) for c, r in inside):
        return []
    away = [u for c, _ in inside if (u := _unit((here[0] - c[0], here[1] - c[1])))]
    # Square to a centre the distance grows too: between two centres the only way out.
    ahead = _unit((nxt[0] - here[0], nxt[1] - here[1])) if nxt is not None else None
    square = [s for u in away for s in ((-u[1], u[0]), (u[1], -u[0]))]
    if ahead is not None:
        square.sort(key=lambda s: -(s[0] * ahead[0] + s[1] * ahead[1]))
    preferred = [
        _unit(exit_dir) if exit_dir is not None else None,
        _unit((sum(u[0] for u in away), sum(u[1] for u in away))),
        *away,
        *square,
    ]
    candidates = [u for u in preferred if u is not None]
    heading = math.atan2(candidates[0][1], candidates[0][0]) if candidates else 0.0
    steps = round(2 * math.pi / _EXIT_STEP)
    for k in sorted(range(steps), key=lambda k: min(k, steps - k)):
        a = heading + k * _EXIT_STEP
        candidates.append((math.cos(a), math.sin(a)))
    for u in candidates:
        p = _leave(here, u, inside)
        if not any(leg_enters(here, p, c, r) for c, r in cyl):
            return [p]
    return None


def _unit(v: Xy) -> Xy | None:
    n = math.hypot(*v)
    return None if n < 1e-9 else (v[0] / n, v[1] / n)


def _leave(here: Xy, u: Xy, inside: Sequence[Circle]) -> Xy:
    """The point along ``u`` from ``here`` just past every circle (c, r + margin)."""
    far = 0.0
    for c, r in inside:
        # here + t*u leaves the circle (c, r + margin) at the larger root of |p - c| = R.
        ox, oy = here[0] - c[0], here[1] - c[1]
        big_r = r + KEEPOUT_ARC_MARGIN_M
        bq = ox * u[0] + oy * u[1]
        cq = ox * ox + oy * oy - big_r * big_r
        far = max(far, -bq + math.sqrt(bq * bq - cq) + _EXIT_EPS_M)
    return (here[0] + far * u[0], here[1] + far * u[1])


def _first_hit(a: Xy, b: Xy, cyl: Sequence[Circle], *, margin: bool = False) -> int | None:
    """The cylinder the leg reaches first (by its closest approach along the leg). With
    ``margin`` (a leg of a way around) a leg that comes closer than the margin to a
    cylinder (an end already inside it: closer than that end) reaches it too."""
    best: tuple[float, int] | None = None
    dx, dy = b[0] - a[0], b[1] - a[1]
    seg2 = dx * dx + dy * dy
    for i, (c, r) in enumerate(cyl):
        if not leg_enters(a, b, c, r) and not (margin and _within_margin(a, b, (c, r))):
            continue
        t = 0.0 if seg2 == 0.0 else -((a[0] - c[0]) * dx + (a[1] - c[1]) * dy) / seg2
        if best is None or t < best[0]:
            best = (t, i)
    return None if best is None else best[1]


def _within_margin(p: Xy, q: Xy, circle: Circle) -> bool:
    """The leg comes closer to the cylinder than the margin, or than an end already
    inside the margin (a leg along a margin polygon's edge touches it: not closer)."""
    c, r = circle
    keep = min(r + KEEPOUT_ARC_MARGIN_M, math.dist(p, c), math.dist(q, c))
    return point_segment_distance(c, p, q) < keep - _MARGIN_EPS_M


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


def _group(cyl: Sequence[Circle], hit: int) -> list[Circle]:
    """``cyl[hit]`` and every cylinder linked to it by overlapping margins: a way around
    one of them alone would run into the next."""
    reach = 2 * KEEPOUT_ARC_MARGIN_M
    group, todo = [hit], [hit]
    while todo:
        c, r = cyl[todo.pop()]
        for j, (cj, rj) in enumerate(cyl):
            if j not in group and math.dist(c, cj) < r + rj + reach:
                group.append(j)
                todo.append(j)
    return [cyl[j] for j in group]


def _around_group(
    a: Xy, b: Xy, group: Sequence[Circle], area: Polygon | None, work: list[int]
) -> list[Xy] | None:
    """The corners of the shortest way from ``a`` to ``b`` past cylinders whose margins
    overlap (both ends excluded): through the corners of a polygon just outside each
    margin circle that lie outside every other margin, along legs that keep the margin
    from every cylinder (a leg with an end inside a margin: no closer than that end) and
    stay inside ``area``; None when there is no such way, or none found before ``work``
    (leg checks against a cylinder, shared by one reroute) runs out."""
    steps = round(2 * math.pi / _ARC_STEP)
    corners: list[Xy] = []
    for c, r in group:
        corner_r = (r + KEEPOUT_ARC_MARGIN_M) / math.cos(math.pi / steps)
        for k in range(steps):
            t = 2 * math.pi * k / steps
            p = (c[0] + corner_r * math.cos(t), c[1] + corner_r * math.sin(t))
            if all(math.dist(p, c2) >= r2 + KEEPOUT_ARC_MARGIN_M for c2, r2 in group):
                corners.append(p)
    nodes = [a, *corners, b]
    end = len(nodes) - 1

    def clear(p: Xy, q: Xy) -> bool:
        work[0] -= len(group)
        for c, r in group:
            if leg_enters(p, q, c, r) or _within_margin(p, q, (c, r)):
                return False
        return area is None or area.leg_outside_by(p, q) <= 0.0

    best = [math.inf] * len(nodes)
    prev = [-1] * len(nodes)
    best[0] = 0.0
    todo = set(range(len(nodes)))
    while True:  # ends at b, or when nothing left can be reached
        i = min(todo, key=best.__getitem__)
        if best[i] == math.inf or work[0] <= 0:
            return None
        if i == end:
            path: list[Xy] = []
            while prev[i] > 0:
                i = prev[i]
                path.append(nodes[i])
            return path[::-1]
        todo.discard(i)
        for j in todo:
            d = best[i] + math.dist(nodes[i], nodes[j])
            if d < best[j] and clear(nodes[i], nodes[j]):
                best[j], prev[j] = d, i


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
