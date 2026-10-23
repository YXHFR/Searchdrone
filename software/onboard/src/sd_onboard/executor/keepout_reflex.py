"""The keep-out reflex (E8): what a new cylinder does to the order being flown.

When a cylinder comes (from the drone's own detections or the ground's install), the
executor checks the rest of the running order's route against every cylinder in force.
Nothing changes when it stays clear. A search mission (``survey_sector``, ``follow_path``
and ``resume``: a ``SurveyRun``) is planned again around the cylinders, through the same
``SurveyRun.plan`` as a new order, and checked once more: the whole new route must stay
clear of every cylinder and inside the charter. Anything else, or a search with no way
around, hovers (``KEEPOUT_VIOLATION``). A drone inside a cylinder whose leg already
leads out of it flies on; one whose leg leads deeper first leaves straight away from the
centre (``sd_safety.keepout.reroute``).

Local cylinders (from the drone's own detections) give way once a ground install at
least ``LOCAL_KEEPOUT_HANDOVER_MS`` later holds a cylinder that covers them; until then,
and while the link is down, they stay. With ``LOCAL_KEEPOUTS_MAX`` of them a further
detection stops a moving order in a hover.
"""

from __future__ import annotations

import math
from itertools import pairwise
from typing import Final, Protocol, runtime_checkable

from sd_onboard.skills.base import SkillHandler, SkillPlan
from sd_onboard.skills.survey_sector import SurveyRun
from sd_safety.keepout import Circle, circles, xy_route_violation
from sd_safety.targets import Plane, Xy, polygon_from_geojson
from sd_schemas.charter import MissionCharter
from sd_schemas.keepout import Keepout, leg_enters

__all__ = [
    "LOCAL_KEEPOUTS_MAX",
    "LOCAL_KEEPOUT_HANDOVER_MS",
    "REROUTABLE",
    "FliesSurveyRun",
    "contained",
    "entry_m",
    "remaining",
    "route_clear",
    "stop_m",
    "survey_run",
]

# The orders whose rest is a search to re-plan. ``orbit`` also flies a SurveyRun, but its
# ring is the order itself: a ring a cylinder now crosses is held, not bent.
REROUTABLE: Final = frozenset({"survey_sector", "follow_path", "resume"})
LOCAL_KEEPOUT_HANDOVER_MS: Final = 60_000  # a ground install this much later replaces it
LOCAL_KEEPOUTS_MAX: Final = 100
_NOTHING_LEFT_M: Final = 2.0  # closer to the end than this: hovering, not flying
_STOP_MARGIN_M: Final = 5.0  # an order is held this far (plus its stop) short of a cylinder
_STOP_REACT_S: Final = 1.5  # the reflex's tick, the link to the flight controller, LOITER
_STOP_DECEL_MS2: Final = 2.0  # LOITER's braking, well under ArduCopter's default


@runtime_checkable
class FliesSurveyRun(Protocol):
    """A handler that flies its order as a ``SurveyRun`` (an AUTO mission)."""

    run: SurveyRun | None


def survey_run(handler: SkillHandler) -> SurveyRun | None:
    return handler.run if isinstance(handler, FliesSurveyRun) else None


def remaining(handler: SkillHandler, plan: SkillPlan, here: Xy, plane: Plane) -> list[Xy]:
    """The corners still to fly (current position excluded); [] when only a hover is left."""
    run = survey_run(handler)
    if run is not None:
        return [plane.point(p) for p in run.waypoints[run.next_index :]]
    if not plan.path:
        return []
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


def stop_m(speed_ms: float) -> float:
    """How far short of a cylinder a moving order must be held to stop outside it."""
    v = max(0.0, speed_ms)
    return _STOP_MARGIN_M + v * _STOP_REACT_S + v * v / (2.0 * _STOP_DECEL_MS2)
