"""``resume``: continue the step that ``hold`` (or STOP, a limit, a mode change) broke off.

A goto flies on to its saved target. A search or a drawn path uploads a new mission: home,
speed, the point where it was interrupted, the points still left, loiter and return (a
path flies on in TRANSIT, a search in SURVEY).
"""

from __future__ import annotations

from typing import Any, ClassVar

from sd_mav.modes import CopterMode
from sd_onboard.skills import register_handler
from sd_onboard.skills.base import (
    ExecutorIO,
    PlanContext,
    PlanError,
    SkillPlan,
    SkillStatus,
    TickContext,
)
from sd_onboard.skills.goto import GotoHandler, goto_plan
from sd_onboard.skills.survey_sector import SurveyRun
from sd_schemas.executor import ExecutorState, GotoResumePoint, ResumePoint
from sd_schemas.skills import SkillParams
from sd_schemas.vehicle import VehicleState

__all__ = ["ResumeHandler"]


@register_handler
class ResumeHandler:
    name: ClassVar[str] = "resume"
    TRANSITIONS_EXTRA: ClassVar[dict[Any, Any]] = {}

    def __init__(self) -> None:
        self._goto: GotoHandler | None = None
        self.run: SurveyRun | None = None
        self._home: Any = None

    def plan(self, params: SkillParams, ctx: PlanContext) -> SkillPlan:
        rp = ctx.resume_point
        if rp is None:
            raise PlanError("ONB_NOTHING_TO_RESUME")
        if isinstance(rp, GotoResumePoint):
            goto = GotoHandler()
            goto.prepare(ctx, rp.lat, rp.lon, rp.alt_m, rp.speed_ms)
            goto.command_id = rp.command_id
            self._goto = goto
            return goto_plan(ctx, rp.lat, rp.lon, rp.alt_m, rp.speed_ms)
        self.run = SurveyRun(
            rp.command_id,
            rp.waypoints,
            rp.alt_m,
            rp.speed_ms,
            base_index=rp.next_index,
            interrupted_at=rp.interrupted_at,
            target_state=ExecutorState.TRANSIT if rp.path else ExecutorState.SURVEY,
        )
        self._home = ctx.charter.home
        return self.run.plan(ctx)

    async def start(self, plan: SkillPlan, io: ExecutorIO) -> None:
        if self._goto is not None:
            await self._goto.start(plan, io)
        else:
            if self.run is None:
                raise RuntimeError("planned before running")
            await self.run.fly(plan, io, self._home)

    def on_tick(self, ctx: TickContext) -> SkillStatus:
        if self._goto is not None:
            return self._goto.on_tick(ctx)
        return SkillStatus(stage="running", progress_pct=self.run.progress() if self.run else None)

    async def cancel(self, io: ExecutorIO) -> None:
        if self._goto is not None:
            await self._goto.cancel(io)
        elif self.run is not None:
            await io.set_mode(CopterMode.LOITER)

    def resume_point(self, vehicle: VehicleState) -> ResumePoint | None:
        if self._goto is not None:
            return self._goto.resume_point(vehicle)
        return self.run.resume_point(vehicle) if self.run else None
