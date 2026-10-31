"""The keep-out reflex (E8): what a new cylinder does to the order being flown.

When a cylinder comes (from the drone's own detections or the ground's install), the
executor checks the rest of the running order's route against every cylinder in force.
Nothing changes when it stays clear. A search mission (``survey_sector``, ``follow_path``
and ``resume``: a ``SurveyRun``) is planned again around the cylinders, through the same
``SurveyRun.plan`` as a new order, and checked once more: the whole new route must stay
clear of every cylinder and inside the charter. When the drone is outside every cylinder
and the route enters one within the stop distance (:func:`stop_m`), it cannot turn away
at its speed: it stops in LOITER first, and the rest is planned again from where it
stopped, the same way. Anything else, or a search with no way around, hovers
(``KEEPOUT_VIOLATION``). A drone inside a cylinder whose leg already leads out of it
flies on; one whose leg leads deeper first leaves straight away from the centre
(``sd_safety.keepout.reroute``). The link-loss find move never ends in a hover inside a
cylinder while a way out exists (:func:`way_out`).

A handler that flies one straight leg at a time (``Replans``: ``inspect_detection``) is
checked from where the drone is on every tick while it flies, every cylinder counted
(none it may cross): its leg is planned again from there when it would enter one within
the stop distance, through the whole gate, and held when there is no such leg. Inside a
cylinder over a person with no such leg (the gate refuses every way out), it climbs
straight up to above the cylinder (:func:`climb_alt`) within the charter's height, then
hovers; when it cannot climb it hovers where it is. Either way the operator is alarmed.

Local cylinders (from the drone's own detections) give way once a ground install at
least ``LOCAL_KEEPOUT_HANDOVER_MS`` later holds a cylinder that covers them; until then,
and while the link is down, they stay. A detection whose cylinder mostly overlaps a local
one (:func:`merge_local`) grows that one to cover both, so one person seen again and
again keeps one cylinder. With ``LOCAL_KEEPOUTS_MAX`` of them a further detection stops a
moving order in a hover.
"""

from __future__ import annotations

import math
from collections.abc import Callable, Sequence
from itertools import pairwise
from typing import Final, Protocol, runtime_checkable

from sd_onboard.skills.base import PlanContext, SkillHandler, SkillPlan
from sd_onboard.skills.survey_sector import SurveyRun
from sd_safety.keepout import Circle, circles, reroute, xy_route_violation
from sd_safety.targets import Plane, Polygon, Xy, point_segment_distance, polygon_from_geojson
from sd_schemas.base import LatLonAlt, new_id
from sd_schemas.charter import MissionCharter
from sd_schemas.keepout import KEEPOUT_RADIUS_MAX_M, Keepout, leg_enters

__all__ = [
    "KEEPOUT_CLIMB_MARGIN_M",
    "LOCAL_KEEPOUTS_MAX",
    "LOCAL_KEEPOUT_HANDOVER_MS",
    "REROUTABLE",
    "FliesSurveyRun",
    "Replans",
    "blocks_find_move",
    "climb_alt",
    "contained",
    "entry_m",
    "leg_entry_m",
    "merge_local",
    "remaining",
    "route_clear",
    "stop_m",
    "survey_run",
    "way_out",
]

# The orders whose rest is a search to re-plan. ``orbit`` also flies a SurveyRun, but its
# ring is the order itself: a ring a cylinder now crosses is held, not bent.
REROUTABLE: Final = frozenset({"survey_sector", "follow_path", "resume"})
LOCAL_KEEPOUT_HANDOVER_MS: Final = 60_000  # a ground install this much later replaces it
LOCAL_KEEPOUTS_MAX: Final = 100
# A cylinder has no height of its own; over a person with no safe way out the drone
# takes its top as its radius above home (as far up as out) and climbs this much above.
KEEPOUT_CLIMB_MARGIN_M: Final = 10.0
_NOTHING_LEFT_M: Final = 2.0  # closer to the end than this: hovering, not flying
_STOP_MARGIN_M: Final = 5.0  # an order is held this far (plus its stop) short of a cylinder
_STOP_REACT_S: Final = 1.5  # the reflex's tick, the link to the flight controller, LOITER
_STOP_DECEL_MS2: Final = 2.0  # LOITER's braking, well under ArduCopter's default
# A way out planned a tick ago, seen from where the drone has moved on to, may come this
# much closer to a centre before it counts as going deeper (the drone keeps leaving).
_DEEPER_TOL_M: Final = 1.0
_MERGE_SHARE: Final = 0.5  # centres at most this share of the smaller radius apart: merged
_MERGE_EPS_M: Final = 0.1  # a merged cylinder grows this much more: rounding never shrinks it
_MERGE_GROW_MAX: Final = 2.0  # a merged cylinder: at most this many new detection radii


@runtime_checkable
class FliesSurveyRun(Protocol):
    """A handler that flies its order as a ``SurveyRun`` (an AUTO mission)."""

    run: SurveyRun | None


@runtime_checkable
class Replans(Protocol):
    """A handler that flies one straight leg at a time and can plan it again from where
    the drone is (``inspect_detection``)."""

    def leg_end(self) -> tuple[float, float] | None:
        """Where the leg being flown ends (lat, lon); None when no leg is being flown."""
        ...

    def exiting(self) -> bool:
        """Whether the leg being flown is a way out of cylinders."""
        ...

    def replan(self, ctx: PlanContext, allowed: Callable[[list[LatLonAlt]], bool]) -> bool:
        """Plan the leg again from ``ctx`` and fly it if ``allowed`` takes its path;
        False (nothing changed) when there is no such leg."""
        ...

    def climbing(self) -> bool:
        """Whether the leg being flown is the climb out of reach (:meth:`climb`)."""
        ...

    def climb(self, ctx: PlanContext, alt_m: float) -> bool:
        """Climb straight up from where the drone is to ``alt_m`` (above home), then end
        the order in the keep-out hover; False when no leg is being flown."""
        ...


def survey_run(handler: SkillHandler) -> SurveyRun | None:
    return handler.run if isinstance(handler, FliesSurveyRun) else None


def remaining(handler: SkillHandler, plan: SkillPlan, here: Xy, plane: Plane) -> list[Xy]:
    """The corners still to fly (current position excluded); [] when only a hover is left."""
    run = survey_run(handler)
    if run is not None:
        return [plane.point(p) for p in run.waypoints[run.next_index :]]
    if isinstance(handler, Replans):
        leg = handler.leg_end()
        if leg is None:
            return []
        end = plane.xy(*leg)
    elif not plan.path:
        return []
    else:
        end = plane.xy(plan.path[-1].lat, plan.path[-1].lon)
    return [] if math.dist(here, end) < _NOTHING_LEFT_M else [end]


def route_clear(
    route: list[Xy], keepouts: tuple[Keepout, ...], plane: Plane, charter: MissionCharter
) -> bool:
    """No leg enters a cylinder and every leg lies inside the charter's area."""
    if xy_route_violation(route, circles(keepouts, plane)) is not None:
        return False
    area = polygon_from_geojson(charter.area, plane)
    return all(area.leg_outside_by(a, b) <= 0.0 for a, b in pairwise(route))


def blocks_find_move(here: Xy, point: Xy, others: list[Circle]) -> bool:
    """Whether the link-loss find's move from ``here`` to its hover ``point`` would enter
    one of ``others`` (every cylinder in force but the find's own) or end inside one."""
    return (
        xy_route_violation([here, point], others) is not None
        or xy_route_violation([point], others) is not None
    )


def contained(k: Keepout, others: tuple[Keepout, ...], plane: Plane) -> bool:
    """Whether ``k`` lies wholly inside one of ``others`` (it adds nothing)."""
    c = plane.xy(k.lat, k.lon)
    return any(math.dist(c, plane.xy(o.lat, o.lon)) + k.radius_m <= o.radius_m for o in others)


def entry_m(
    route: list[Xy], cyl: list[Circle], *, through: frozenset[int] = frozenset()
) -> float | None:
    """How far along ``route`` (from its first point) it first enters a cylinder, or None.
    A leg that starts inside and heads deeper enters at once, except that the first leg may
    cross the cylinders (indices into ``cyl``) in ``through``: it leaves those along the
    leg, and only a later leg may not go in again."""
    done = 0.0
    for i, (a, b) in enumerate(pairwise(route)):
        length = math.dist(a, b)
        hits = [
            _leg_entry(a, b, c, r)
            for j, (c, r) in enumerate(cyl)
            if leg_enters(a, b, c, r) and not (i == 0 and j in through)
        ]
        if hits:
            return done + min(hits) * length
        done += length
    return None


def leg_entry_m(here: Xy, end: Xy, cyl: list[Circle]) -> float | None:
    """How far along the straight leg ``here`` -> ``end`` it first comes inside a
    cylinder it is outside of; 0.0 when it ends inside one it is in, or comes more than
    ``_DEEPER_TOL_M`` closer to its centre; None when clear."""
    length = math.dist(here, end)
    hits: list[float] = []
    for c, r in cyl:
        now = math.dist(here, c)
        if now < r:
            if math.dist(end, c) < r or point_segment_distance(c, here, end) < now - _DEEPER_TOL_M:
                hits.append(0.0)
        elif leg_enters(here, end, c, r):
            hits.append(_leg_entry(here, end, c, r) * length)
    return min(hits, default=None)


def _leg_entry(a: Xy, b: Xy, c: Xy, r: float) -> float:
    """Where (0..1) the leg a -> b, which enters the circle, first comes inside it."""
    ax, ay = a[0] - c[0], a[1] - c[1]
    dx, dy = b[0] - a[0], b[1] - a[1]
    seg2 = dx * dx + dy * dy
    half = ax * dx + ay * dy
    disc = half * half - seg2 * (ax * ax + ay * ay - r * r)
    if seg2 == 0.0 or disc <= 0.0 or math.hypot(ax, ay) < r:
        return 0.0
    return max(0.0, (-half - math.sqrt(disc)) / seg2)


def climb_alt(here: Xy, cyl: Sequence[Circle], alt_m: float, alt_max_m: float) -> float | None:
    """How high (above home) a drone at ``here``, ``alt_m`` up, climbs over the cylinders
    it is inside: their top (the largest radius) plus ``KEEPOUT_CLIMB_MARGIN_M``, at most
    ``alt_max_m``; where it is when it is that high already; None when it is inside none,
    or lower than that and cannot climb (``alt_max_m`` is no higher than it is)."""
    over = [r for c, r in cyl if math.dist(here, c) < r]
    if not over:
        return None
    want = max(over) + KEEPOUT_CLIMB_MARGIN_M
    if alt_m >= want:
        return alt_m
    top = min(want, alt_max_m)
    return top if top > alt_m else None


def stop_m(speed_ms: float) -> float:
    """How far short of a cylinder a moving order must be held to stop outside it."""
    v = max(0.0, speed_ms)
    return _STOP_MARGIN_M + v * _STOP_REACT_S + v * v / (2.0 * _STOP_DECEL_MS2)


def way_out(here: Xy, cyl: list[Circle], area: Polygon) -> Xy | None:
    """The point just outside every cylinder ``here`` lies in, along a straight leg that
    goes deeper into none of them and enters no other, inside ``area``
    (``sd_safety.keepout.reroute``); None when ``here`` is in none or no such leg exists."""
    way = reroute([here], cyl, area=area)
    return None if way is None or len(way) < 2 else way[1]


def merge_local(
    k: Keepout, local: Sequence[Keepout], plane: Plane
) -> tuple[Keepout, Keepout] | None:
    """The local cylinder ``k`` mostly overlaps (the centres at most ``_MERGE_SHARE`` of
    the smaller radius apart, the nearest first) and one new cylinder that covers both, or
    None. The new one is the smallest circle around both (never smaller than either)
    plus ``_MERGE_EPS_M``; none when that would be more than ``_MERGE_GROW_MAX`` times
    ``k``'s radius (a person who moves on gets a new cylinder instead of dragging one
    along) or larger than a Keepout may be."""
    c = plane.xy(k.lat, k.lon)
    dists = [(math.dist(c, plane.xy(o.lat, o.lon)), i) for i, o in enumerate(local)]
    near = sorted(
        (d, i) for d, i in dists if d <= _MERGE_SHARE * min(k.radius_m, local[i].radius_m)
    )
    if not near:
        return None
    old = local[near[0][1]]
    centre, radius = _enclose((plane.xy(old.lat, old.lon), old.radius_m), (c, k.radius_m))
    radius += _MERGE_EPS_M
    if radius > min(_MERGE_GROW_MAX * k.radius_m, KEEPOUT_RADIUS_MAX_M):
        return None
    lat, lon = plane.latlon(*centre)
    grown = old.model_copy(
        update={"keepout_id": new_id(), "lat": lat, "lon": lon, "radius_m": radius}
    )
    return old, grown


def _enclose(a: Circle, b: Circle) -> Circle:
    """The smallest circle that holds both."""
    (ca, ra), (cb, rb) = a, b
    d = math.dist(ca, cb)
    if d + rb <= ra:
        return a
    if d + ra <= rb:
        return b
    big = (d + ra + rb) / 2.0
    t = (big - ra) / d
    return (ca[0] + t * (cb[0] - ca[0]), ca[1] + t * (cb[1] - ca[1])), big
