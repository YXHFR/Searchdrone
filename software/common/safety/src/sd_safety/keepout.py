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
first leaves them along a straight leg that goes deeper into none of them, enters no
other and stays inside the charter, ending at least ``_EXIT_EDGE_M`` (or the caller's
reserve) short of its edge along that leg: the given direction (the leg being flown),
away from their centres, or the nearest direction that works. When the rest of the
route has no way around after that way out, the next ways out are tried
(``_EXIT_TRIES`` at most). Pure functions in the plane of ``sd_safety.targets.Plane``.

**Passage** (owner decision 2026-11-03): a cylinder has no top; it is a vertical cylinder
of any height. A search route (``KEEPOUT_PASS_SKILLS``) of a charter that allows overflight
may cross the cylinder of an UNCONFIRMED person (``UNCONFIRMED_SOURCES``: a detection, an
open candidate) straight through (:func:`passes`): no corner inside it (no hover, turn or
end of a leg), no descent while inside, and at most ``Passage.max_chord_m`` of the leg
inside it (``KEEPOUT_PASS_MAX_S`` at the route's speed). Only the first leg, from where the
drone is, may start inside one (the cylinder came around the drone). Confirmed finds, and
every cylinder without a passage (live without the permit, every other move), are never
entered. :func:`reroute` with a passage first drops the corners inside unconfirmed
cylinders and joins their neighbours straight; a cylinder that a leg then may not pass is
gone around like a confirmed one. :func:`escape` is the way out of a drone boxed in: the
shortest straight leg to a point clear of every cylinder that enters no confirmed one
(an unconfirmed one may be crossed even without a passage: staying over a possible person
is worse).
"""

from __future__ import annotations

import itertools
import math
from collections.abc import Iterator, Sequence
from dataclasses import dataclass
from itertools import pairwise
from typing import Final

from sd_safety.constants import GOTO_ALT_STEP_TOL_M, KEEPOUT_PASS_MAX_S
from sd_safety.policy import KEEPOUT_PASS_SKILLS
from sd_safety.targets import Plane, Polygon, Xy, point_segment_distance
from sd_schemas.base import LatLon, LatLonAlt
from sd_schemas.charter import MissionCharter
from sd_schemas.keepout import Keepout, leg_enters
from sd_schemas.skills import OrbitRing

__all__ = [
    "KEEPOUT_ARC_MARGIN_M",
    "ORBIT_KEEPOUT_MARGIN_M",
    "UNCONFIRMED_SOURCES",
    "Circle",
    "Passage",
    "chord_m",
    "circles",
    "escape",
    "passage_for",
    "passes",
    "reroute",
    "ring_violation",
    "route_violation",
    "split",
    "unconfirmed",
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
_EXIT_EDGE_M: Final = 2.0  # room along a way out's leg past its end, inside the charter
# Ways out tried in turn when the rest of the route finds no way. The ways around single
# cylinders and the checks against the area run once per try (up to _EXIT_TRIES times);
# the grouped ways around share one _GROUP_WORK_MAX. The slowest call fuzzing found
# took about 0.1 s on the reference fuzz machine (a measurement, not a bound).
_EXIT_TRIES: Final = 4
# A reroute with a passage plans again with the unconfirmed cylinders its legs may not pass
# counted as confirmed ones, at most this many times (each try is one ``reroute``).
_PROMOTE_TRIES: Final = 4
_ESCAPE_STEP: Final = math.radians(5.0)  # the directions an escape tries

# The sources of a cylinder around a person nobody has confirmed: a search route may pass
# through them (with a passage). Any other source (a confirmed find, and any source added
# later, such as an operator's marking) keeps the full cylinder.
UNCONFIRMED_SOURCES: Final = frozenset({"detection", "candidate"})

Circle = tuple[Xy, float]  # centre, radius


@dataclass(frozen=True, slots=True)
class Passage:
    """How a search route may cross unconfirmed cylinders: straight through, at most
    ``max_chord_m`` of a leg inside each."""

    max_chord_m: float

    @classmethod
    def at(cls, speed_ms: float) -> Passage:
        """``KEEPOUT_PASS_MAX_S`` at the route's speed."""
        return cls(max_chord_m=max(0.0, speed_ms) * KEEPOUT_PASS_MAX_S)


def passage_for(
    skill: str, charter: MissionCharter | None, speed_ms: float | None
) -> Passage | None:
    """The passage of a ``skill`` flown at ``speed_ms`` under ``charter``: only for the
    search routes (``KEEPOUT_PASS_SKILLS``) of a charter that allows overflight, with a
    known speed; None (no cylinder may be entered) otherwise."""
    if charter is None or not charter.overflight_allowed or skill not in KEEPOUT_PASS_SKILLS:
        return None
    if speed_ms is None or not math.isfinite(speed_ms) or speed_ms <= 0.0:
        return None
    return Passage.at(speed_ms)


def unconfirmed(k: Keepout) -> bool:
    return k.source in UNCONFIRMED_SOURCES


def split(keepouts: Sequence[Keepout]) -> tuple[list[Keepout], list[Keepout]]:
    """(the confirmed cylinders, the unconfirmed ones)."""
    return [k for k in keepouts if not unconfirmed(k)], [k for k in keepouts if unconfirmed(k)]


def circles(keepouts: Sequence[Keepout], plane: Plane) -> list[Circle]:
    return [(plane.xy(k.lat, k.lon), k.radius_m) for k in keepouts]


def chord_m(a: Xy, b: Xy, c: Xy, r: float) -> float:
    """How much of the straight leg ``a`` -> ``b`` lies inside the circle (``c``, ``r``)."""
    dx, dy = b[0] - a[0], b[1] - a[1]
    seg = math.hypot(dx, dy)
    if seg == 0.0:
        return 0.0
    ux, uy = dx / seg, dy / seg
    ox, oy = a[0] - c[0], a[1] - c[1]
    bq = ox * ux + oy * uy
    disc = bq * bq - (ox * ox + oy * oy - r * r)
    if disc <= 0.0:
        return 0.0
    root = math.sqrt(disc)
    return max(0.0, min(seg, -bq + root) - max(0.0, -bq - root))


def passes(a: Xy, b: Xy, c: Xy, r: float, passage: Passage, *, first: bool) -> bool:
    """Whether a search route may fly the leg ``a`` -> ``b`` past the unconfirmed circle
    (``c``, ``r``): it ends outside it (no corner, hover or turn inside), starts outside it
    unless it is the ``first`` leg (from where the drone is), and has at most
    ``passage.max_chord_m`` inside it. A leg that never comes inside passes."""
    if math.dist(b, c) < r:
        return False
    if not first and math.dist(a, c) < r:
        return False
    return chord_m(a, b, c, r) <= passage.max_chord_m


def xy_route_violation(
    route: Sequence[Xy],
    cyl: Sequence[Circle],
    soft: Sequence[Circle] = (),
    passage: Passage | None = None,
    *,
    descends: Sequence[bool] = (),
) -> int | None:
    """The index into ``[*cyl, *soft]`` of the first cylinder the route enters, or None. A
    route of one point is a point: inside a cylinder is a violation. ``soft`` are the
    unconfirmed cylinders: with a ``passage`` a leg may pass through them (:func:`passes`),
    except a leg that ``descends`` (by index); without one they are like ``cyl``."""
    if len(route) == 1:
        p = route[0]
        return next((i for i, (c, r) in enumerate([*cyl, *soft]) if math.dist(p, c) < r), None)
    for j, (a, b) in enumerate(pairwise(route)):
        for i, (c, r) in enumerate(cyl):
            if leg_enters(a, b, c, r):
                return i
        full = passage is None or (j < len(descends) and descends[j])
        for i, (c, r) in enumerate(soft):
            if full:
                if leg_enters(a, b, c, r):
                    return len(cyl) + i
            elif passage is not None and not passes(a, b, c, r, passage, first=j == 0):
                return len(cyl) + i
    return None


def _descends(route: Sequence[LatLon]) -> list[bool]:
    """Per leg: whether it ends lower than it starts (by more than the altitude's step)."""
    out: list[bool] = []
    for a, b in pairwise(route):
        if isinstance(a, LatLonAlt) and isinstance(b, LatLonAlt):
            out.append(b.alt_amsl_m < a.alt_amsl_m - GOTO_ALT_STEP_TOL_M)
        else:
            out.append(False)
    return out


def route_violation(
    route: Sequence[LatLon],
    keepouts: Sequence[Keepout],
    plane: Plane,
    passage: Passage | None = None,
) -> Keepout | None:
    """The first cylinder the route (corners, current position first) enters, or None.
    With a ``passage`` its legs may pass through unconfirmed cylinders (:func:`passes`),
    never while descending (a route with heights)."""
    if not route or not keepouts:
        return None
    pts = [plane.point(p) for p in route]
    if passage is None:
        found = xy_route_violation(pts, circles(keepouts, plane))
        return None if found is None else keepouts[found]
    hard, soft = split(keepouts)
    found = xy_route_violation(
        pts, circles(hard, plane), circles(soft, plane), passage, descends=_descends(route)
    )
    return None if found is None else [*hard, *soft][found]


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
    exit_reserve_m: float = 0.0,
    soft: Sequence[Circle] = (),
    passage: Passage | None = None,
) -> list[Xy] | None:
    """``route`` (current position first) with every crossing of a cylinder replaced by
    a way around it, or None when there is none that stays inside ``area``.

    ``soft`` are the unconfirmed cylinders. Without a ``passage`` they count as ``cyl``.
    With one, the corners inside them are dropped and the route is planned around ``cyl``
    alone; each unconfirmed cylinder one of its legs may not pass (:func:`passes`) then
    counts as confirmed and the route is planned again (``_PROMOTE_TRIES`` at most), so the
    result is clear by :func:`xy_route_violation` with the same passage.

    Corners inside a cylinder are dropped (the rest of the route never goes back in).
    When the position is inside cylinders and the first leg would go deeper, the route
    first leaves them (``_exits``): along ``exit_dir`` when given (the leg being flown)
    if that works, else away from their centres, never by a way that leaves ``area``
    (near the charter's edge the way out turns back in). The way out's leg keeps
    ``exit_reserve_m`` (at least ``_EXIT_EDGE_M``) of room past its end, clear of every
    cylinder and inside ``area``, for a caller that flies on past it. Every leg after
    that is clear of the cylinders, and the whole route must lie inside ``area``; when
    the rest finds no way around, the next way out is tried (``_EXIT_TRIES`` at most)."""
    if passage is None or not soft:
        return _reroute(route, [*cyl, *soft], area, exit_dir, exit_reserve_m)
    if not route:
        return []
    hard, left = list(cyl), list(soft)
    pts = [route[0], *(p for p in route[1:] if not any(math.dist(p, c) < r for c, r in soft))]
    for _ in range(_PROMOTE_TRIES):
        out = _reroute(pts, hard, area, exit_dir, exit_reserve_m)
        if out is None:
            return None
        legs = list(pairwise(out))
        bad = [
            s
            for s in left
            if not all(passes(a, b, *s, passage, first=j == 0) for j, (a, b) in enumerate(legs))
        ]
        if not bad:
            return out
        hard += bad
        left = [s for s in left if s not in bad]
    return None


def _reroute(
    route: Sequence[Xy],
    cyl: Sequence[Circle],
    area: Polygon | None,
    exit_dir: Xy | None,
    exit_reserve_m: float,
) -> list[Xy] | None:
    if not route:
        return []
    here = route[0]
    rest = [p for p in route[1:] if not any(math.dist(p, c) < r for c, r in cyl)]
    nxt = route[1] if len(route) > 1 else None
    work = [_GROUP_WORK_MAX]  # what the grouped ways around may still check, every try
    ways = _exits(here, nxt, cyl, exit_dir, area, exit_reserve_m)
    for way_out in itertools.islice(ways, _EXIT_TRIES):
        out = _around_all([here, *way_out, *rest], len(way_out), cyl, area, work)
        if out is not None:
            return out
    return None


def _around_all(
    out: list[Xy], start: int, cyl: Sequence[Circle], area: Polygon | None, work: list[int]
) -> list[Xy] | None:
    """``out`` with every leg from ``start`` on that crosses a cylinder replaced by a way
    around it; None when there is none inside ``area``."""
    made = [False] * len(out)  # a corner of a way around (its legs keep the margin)
    i, detours = start, 0  # the leg out of the cylinders is clear already (``_exits``)
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


def _exits(
    here: Xy,
    nxt: Xy | None,
    cyl: Sequence[Circle],
    exit_dir: Xy | None,
    area: Polygon | None,
    reserve_m: float,
) -> Iterator[list[Xy]]:
    """The ways out of the cylinders ``here`` is in, best first: no corner when it is in
    none or the first leg leaves without going deeper, else one corner past all of them
    along a straight leg that goes deeper into none of them, enters no other and stays
    inside ``area`` with ``max(reserve_m, _EXIT_EDGE_M)`` of room past its end. None at
    all when there is no such leg.

    The directions in turn: ``exit_dir``, away from the centres together, straight out
    from each centre, square to each (nearest the next corner first), then every
    ``_EXIT_STEP`` around, nearest the first of these first."""
    inside = [(c, r) for c, r in cyl if math.dist(here, c) < r]
    if not inside:
        yield []
        return
    if nxt is not None and not any(leg_enters(here, nxt, c, r) for c, r in inside):
        yield []
        return
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
    room = max(reserve_m, _EXIT_EDGE_M)
    seen: list[Xy] = []
    for u in candidates:
        if any(u[0] * v[0] + u[1] * v[1] > 1.0 - 1e-9 for v in seen):
            continue  # the same direction again
        seen.append(u)
        p = _leave(here, u, inside)
        far = (p[0] + room * u[0], p[1] + room * u[1])
        if any(leg_enters(here, far, c, r) for c, r in cyl):
            continue
        if area is not None and area.leg_outside_by(here, far) > 0.0:
            continue
        yield [p]


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


# --- boxed in ---------------------------------------------------------------------------


def escape(
    here: Xy, hard: Sequence[Circle], soft: Sequence[Circle], area: Polygon | None
) -> Xy | None:
    """The way out of a drone boxed in over a possible person: the nearest point clear of
    every cylinder by ``KEEPOUT_ARC_MARGIN_M`` along a straight leg from ``here`` that goes
    deeper into no confirmed cylinder (``hard``) and enters none, and stays inside
    ``area``. The leg may cross unconfirmed cylinders (``soft``) with or without a passage:
    flying through one is shorter over a possible person than hovering in it. None when
    ``here`` lies in no cylinder, or no such leg exists (the caller flies home)."""
    every = [*hard, *soft]
    if not any(math.dist(here, c) < r for c, r in every):
        return None
    best: tuple[float, Xy] | None = None
    for k in range(round(2 * math.pi / _ESCAPE_STEP)):
        u = (math.cos(k * _ESCAPE_STEP), math.sin(k * _ESCAPE_STEP))
        t = _clear_along(here, u, every)
        if best is not None and t >= best[0]:
            continue
        p = (here[0] + t * u[0], here[1] + t * u[1])
        if any(leg_enters(here, p, c, r) for c, r in hard):
            continue
        if area is not None and area.leg_outside_by(here, p) > 0.0:
            continue
        best = (t, p)
    return None if best is None else best[1]


def _clear_along(here: Xy, u: Xy, cyl: Sequence[Circle]) -> float:
    """How far along the unit direction ``u`` from ``here`` the first point lies that is
    clear of every circle (c, r + ``KEEPOUT_ARC_MARGIN_M``), plus ``_EXIT_EPS_M``."""
    spans: list[tuple[float, float]] = []
    for c, r in cyl:
        ox, oy = here[0] - c[0], here[1] - c[1]
        big_r = r + KEEPOUT_ARC_MARGIN_M
        bq = ox * u[0] + oy * u[1]
        disc = bq * bq - (ox * ox + oy * oy - big_r * big_r)
        if disc > 0.0:
            root = math.sqrt(disc)
            spans.append((-bq - root, -bq + root))
    t = 0.0
    for lo, hi in sorted(spans):
        if lo > t:
            break
        if hi > t:
            t = hi + _EXIT_EPS_M
    return t
