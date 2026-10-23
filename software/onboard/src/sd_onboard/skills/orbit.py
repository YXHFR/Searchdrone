"""``orbit``: circle a point as a small AUTO mission, then hover.

The ring is ``sd_schemas.skills.orbit_ring``: 24 corners just outside the circle,
starting at the one nearest the drone, flown ``turns`` times and back to the first
corner, at ``ORBIT_SPEED_MS`` (the charter's limit if lower). It is a mission of plain
waypoints (``SurveyRun``), so it ends with the same loiter and return-to-launch if the
onboard computer stops, and its speed is the speed item's. The gate has checked the
whole ring before it starts: inside the charter and 10 m from every keep-out cylinder.
"""

from __future__ import annotations

from typing import Any, ClassVar, Final

from sd_mav.modes import CopterMode
from sd_onboard.skills import register_handler
from sd_onboard.skills.base import (
    ExecutorIO,
    PlanContext,
    SkillPlan,
    SkillStatus,
    TickContext,
)
from sd_onboard.skills.survey_sector import SurveyRun
from sd_safety.constants import EXECUTOR_DEFAULT_SPEED_MS
from sd_schemas.base import LatLon, LatLonAlt
from sd_schemas.executor import ExecutorState, ResumePoint
from sd_schemas.skills import OrbitParams, SkillParams, orbit_ring
from sd_schemas.vehicle import VehicleState

__all__ = ["ORBIT_SPEED_MS", "OrbitHandler", "orbit_points"]

ORBIT_SPEED_MS: Final = EXECUTOR_DEFAULT_SPEED_MS


def orbit_points(params: OrbitParams, start: LatLon) -> list[LatLon]:
    ring = orbit_ring(LatLon(lat=params.lat, lon=params.lon), params.radius_m, start)
    return [*(ring[:-1] * params.turns), ring[-1]]


@register_handler
class OrbitHandler:
    name: ClassVar[str] = "orbit"
    TRANSITIONS_EXTRA: ClassVar[dict[Any, Any]] = {}

    def __init__(self) -> None:
        self.run: SurveyRun | None = None
        self._home: LatLonAlt | None = None

    def plan(self, params: SkillParams, ctx: PlanContext) -> SkillPlan:
        if not isinstance(params, OrbitParams):
            raise TypeError("expected OrbitParams")
        here = LatLon(lat=ctx.vehicle.lat, lon=ctx.vehicle.lon)
        speed = min(ORBIT_SPEED_MS, ctx.charter.max_speed_ms)
        self.run = SurveyRun(
            ctx.command_id,
            orbit_points(params, here),
            params.alt_m,
            speed,
            target_state=ExecutorState.TRANSIT,
        )
        self._home = ctx.charter.home
        return self.run.plan(ctx)

    async def start(self, plan: SkillPlan, io: ExecutorIO) -> None:
        if self.run is None or self._home is None:
            raise RuntimeError("planned before running")
        await self.run.fly(plan, io, self._home)

    def on_tick(self, ctx: TickContext) -> SkillStatus:
        return SkillStatus(stage="running", progress_pct=self.run.progress() if self.run else None)

    async def cancel(self, io: ExecutorIO) -> None:
        await io.set_mode(CopterMode.LOITER)

    def resume_point(self, vehicle: VehicleState) -> ResumePoint | None:
        return None  # an interrupted orbit is ordered again, not resumed
