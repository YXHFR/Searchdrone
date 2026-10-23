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
from sd_safety.keepout import circles, xy_route_violation
from sd_safety.targets import Plane, Xy, polygon_from_geojson
from sd_schemas.charter import MissionCharter
from sd_schemas.keepout import Keepout

__all__ = [
    "LOCAL_KEEPOUTS_MAX",
    "LOCAL_KEEPOUT_HANDOVER_MS",
    "REROUTABLE",
    "FliesSurveyRun",
    "contained",
    "remaining",
    "route_clear",
    "survey_run",
]

REROUTABLE: Final = frozenset({"survey_sector", "follow_path", "resume"})
LOCAL_KEEPOUT_HANDOVER_MS: Final = 60_000  # a ground install this much later replaces it
LOCAL_KEEPOUTS_MAX: Final = 100
_NOTHING_LEFT_M: Final = 2.0  # closer to the end than this: hovering, not flying


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


def contained(k: Keepout, others: tuple[Keepout, ...], plane: Plane) -> bool:
    """Whether ``k`` lies wholly inside one of ``others`` (it adds nothing)."""
    c = plane.xy(k.lat, k.lon)
    return any(math.dist(c, plane.xy(o.lat, o.lon)) + k.radius_m <= o.radius_m for o in others)
