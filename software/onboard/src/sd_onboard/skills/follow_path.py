"""``follow_path``: fly a line the pilot drew, point by point, then hover at its end.

The points come from the drawn path (sd-brain resolves ``pa-<n>`` and the ground checks
it; the model never writes coordinates). They are flown as an AUTO mission like a search
(``SurveyRun``): home, speed, the points, a 30 s loiter at the last one and
return-to-launch, so the flight ends safely even if the onboard computer dies. At the
last point the executor switches to LOITER (HOLD). STOP, FLY HOME and the pilot's
takeover act on it like on any other step, and ``resume`` flies the points still left
(in TRANSIT, with the battery checked again).
"""

from __future__ import annotations

import math
from itertools import pairwise
from typing import Any, ClassVar

from sd_mav.modes import CopterMode
from sd_onboard.skills import register_handler
from sd_onboard.skills.base import (
    ExecutorIO,
    PlanContext,
    SkillPlan,
    SkillStatus,
    TickContext,
)
from sd_onboard.skills.survey_sector import DEFAULT_SPEED_MS, SurveyRun
from sd_safety.targets import Plane
from sd_schemas.base import LatLon, LatLonAlt
from sd_schemas.executor import ExecutorState, ResumePoint
from sd_schemas.skills import FollowPathParams, SkillParams
from sd_schemas.vehicle import VehicleState

__all__ = ["FollowPathHandler", "path_progress_pct"]

S = ExecutorState


def path_progress_pct(
    plane: Plane, points: list[LatLon], next_index: int, lat: float, lon: float
) -> float:
    """How far along the line the vehicle is, in % of its length from the first point.

    ``next_index`` is the point the mission is flying to. The vehicle's position counts
    on the leg it is on (projected onto it), so the number grows smoothly.
    """
    xy = [plane.xy(p.lat, p.lon) for p in points]
    legs = [math.dist(a, b) for a, b in pairwise(xy)]
    total = sum(legs)
    if total <= 0 or next_index <= 0:
        return 0.0
    if next_index >= len(points):
        return 100.0
    done = sum(legs[: next_index - 1])
    (ax, ay), (bx, by) = xy[next_index - 1], xy[next_index]
    vx, vy = plane.xy(lat, lon)
    leg = legs[next_index - 1]
    along = 0.0 if leg <= 0 else ((vx - ax) * (bx - ax) + (vy - ay) * (by - ay)) / leg
    return max(0.0, min(100.0, 100.0 * (done + max(0.0, min(leg, along))) / total))


@register_handler
class FollowPathHandler:
    name: ClassVar[str] = "follow_path"
    TRANSITIONS_EXTRA: ClassVar[dict[Any, Any]] = {}  # T7b in the table: into TRANSIT

    def __init__(self) -> None:
        self.run: SurveyRun | None = None
        self._home: LatLonAlt | None = None
        self._plane: Plane | None = None
        self._progress = 0.0

    def plan(self, params: SkillParams, ctx: PlanContext) -> SkillPlan:
        if not isinstance(params, FollowPathParams):
            raise TypeError("expected FollowPathParams")
        speed = params.speed_ms or min(DEFAULT_SPEED_MS, ctx.charter.max_speed_ms)
        self.run = SurveyRun(
            ctx.command_id,
            list(params.points),
            params.alt_m,
            speed,
            target_state=S.TRANSIT,
        )
        self._home = ctx.charter.home
        self._plane = ctx.plane
        return self.run.plan(ctx)

    async def start(self, plan: SkillPlan, io: ExecutorIO) -> None:
        if self.run is None or self._home is None:
            raise RuntimeError("planned before running")
        await self.run.fly(plan, io, self._home)

    def on_tick(self, ctx: TickContext) -> SkillStatus:
        run, plane = self.run, self._plane
        if run is not None and plane is not None:
            self._progress = max(
                self._progress,
                path_progress_pct(
                    plane, run.waypoints, run.next_index, ctx.vehicle.lat, ctx.vehicle.lon
                ),
            )
        return SkillStatus(stage="running", progress_pct=round(self._progress, 1))

    async def cancel(self, io: ExecutorIO) -> None:
        await io.set_mode(CopterMode.LOITER)

    def resume_point(self, vehicle: VehicleState) -> ResumePoint | None:
        return self.run.resume_point(vehicle) if self.run else None
