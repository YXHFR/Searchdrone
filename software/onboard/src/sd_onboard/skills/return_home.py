"""``return_home``: SmartRTL, else RTL. ArduPilot flies home, lands and disarms itself.

Done when the executor sees the vehicle landed and disarmed (LANDED).
"""

from __future__ import annotations

import asyncio
from typing import Any, ClassVar

from sd_mav.modes import CopterMode
from sd_onboard.skills import register_handler
from sd_onboard.skills.base import (
    ExecutorIO,
    MavAction,
    PlanContext,
    SkillFailedError,
    SkillPlan,
    SkillStatus,
    TickContext,
)
from sd_schemas.executor import ExecutorState, ResumePoint
from sd_schemas.skills import SkillParams
from sd_schemas.vehicle import VehicleState

__all__ = ["ReturnHomeHandler"]


@register_handler
class ReturnHomeHandler:
    name: ClassVar[str] = "return_home"
    TRANSITIONS_EXTRA: ClassVar[dict[Any, Any]] = {}

    def plan(self, params: SkillParams, ctx: PlanContext) -> SkillPlan:
        return SkillPlan(
            path=[],
            target_state=ExecutorState.RTL,
            speed_ms=None,
            expected_duration_s=0.0,
            mav_actions=[MavAction(kind="set_mode", params={"mode": "SMART_RTL"})],
        )

    async def start(self, plan: SkillPlan, io: ExecutorIO) -> None:
        if not await io.set_mode(CopterMode.SMART_RTL) and not await io.set_mode(CopterMode.RTL):
            raise SkillFailedError("ONB_MODE_CHANGE_TIMEOUT", "Flygkontrollern flög inte hem.")
        await asyncio.Event().wait()  # done at LANDED, which the executor sees

    def on_tick(self, ctx: TickContext) -> SkillStatus:
        return SkillStatus(stage="running")

    async def cancel(self, io: ExecutorIO) -> None:
        return None  # flying home is never cancelled into LOITER by the executor

    def resume_point(self, vehicle: VehicleState) -> ResumePoint | None:
        return None
