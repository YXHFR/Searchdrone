"""``hold``: stop and hover (LOITER) until the next order. No time limit onboard."""

from __future__ import annotations

from typing import Any, ClassVar, Final

from sd_mav.modes import CopterMode
from sd_onboard.skills import register_handler
from sd_onboard.skills._common import still
from sd_onboard.skills.base import (
    ExecutorIO,
    MavAction,
    PlanContext,
    SkillPlan,
    SkillStatus,
    TickContext,
    mode_failed,
)
from sd_schemas.executor import ExecutorState, ResumePoint
from sd_schemas.skills import SkillParams
from sd_schemas.vehicle import VehicleState

__all__ = ["HoldHandler"]

TICK_S: Final = 0.1


@register_handler
class HoldHandler:
    name: ClassVar[str] = "hold"
    TRANSITIONS_EXTRA: ClassVar[dict[Any, Any]] = {}

    def plan(self, params: SkillParams, ctx: PlanContext) -> SkillPlan:
        return SkillPlan(
            path=[],
            target_state=ExecutorState.HOLD,
            speed_ms=None,
            expected_duration_s=0.0,
            mav_actions=[MavAction(kind="set_mode", params={"mode": "LOITER"})],
        )

    async def start(self, plan: SkillPlan, io: ExecutorIO) -> None:
        if io.vehicle.fc_mode != CopterMode.LOITER.name and not await io.set_mode(
            CopterMode.LOITER
        ):
            raise mode_failed(CopterMode.LOITER)
        while not still(io.vehicle):
            await io.sleep(TICK_S)

    def on_tick(self, ctx: TickContext) -> SkillStatus:
        return SkillStatus(stage="running")

    async def cancel(self, io: ExecutorIO) -> None:
        return None  # already in LOITER

    def resume_point(self, vehicle: VehicleState) -> ResumePoint | None:
        return None
