"""``capture_stills``: take ``count`` stills ``interval_s`` apart while hovering.

No movement: the order runs in HOLD (LOITER) and leaves the drone there. Every still goes
through perception as usual (``ExecutorIO.capture_still``); its report carries the
order's ``command_id``. ``inspect_detection`` takes its stills the same way
(:func:`take_stills`). Failed only when not a single still came.
"""

from __future__ import annotations

from typing import Any, ClassVar, Final
from uuid import UUID

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
from sd_schemas.base import Id
from sd_schemas.executor import ExecutorState, ResumePoint
from sd_schemas.skills import CaptureStillsParams, SkillParams
from sd_schemas.vehicle import VehicleState

__all__ = ["CaptureStillsHandler", "take_stills"]

NO_STILLS: Final = "Kameran gav inga bilder."


async def take_stills(io: ExecutorIO, command_id: Id, count: int, interval_s: float) -> list[UUID]:
    """``count`` stills, one every ``interval_s`` (the first at once); the ids that came."""
    taken: list[UUID] = []
    for i in range(count):
        started = io.clock.monotonic()
        still_id = await io.capture_still(command_id)
        if still_id is not None:
            taken.append(still_id)
        if i < count - 1:
            await io.sleep(max(0.0, interval_s - (io.clock.monotonic() - started)))
    return taken


@register_handler
class CaptureStillsHandler:
    name: ClassVar[str] = "capture_stills"
    TRANSITIONS_EXTRA: ClassVar[dict[Any, Any]] = {}

    def __init__(self) -> None:
        self.command_id: Id | None = None
        self.count = 1
        self.interval_s = 1.0
        self.taken = 0

    def plan(self, params: SkillParams, ctx: PlanContext) -> SkillPlan:
        if not isinstance(params, CaptureStillsParams):
            raise TypeError("expected CaptureStillsParams")
        self.command_id = ctx.command_id
        self.count, self.interval_s = params.count, params.interval_s
        return SkillPlan(
            path=[],
            target_state=ExecutorState.HOLD,
            speed_ms=None,
            expected_duration_s=(params.count - 1) * params.interval_s,
            mav_actions=[MavAction(kind="capture_still", params={"count": params.count})],
        )

    async def start(self, plan: SkillPlan, io: ExecutorIO) -> None:
        if self.command_id is None:
            raise RuntimeError("planned before running")
        taken = await take_stills(io, self.command_id, self.count, self.interval_s)
        self.taken = len(taken)
        io.emit_event(
            "stills_captured",
            {"command_id": str(self.command_id), "asked": self.count, "taken": len(taken)},
        )
        if not taken:
            raise SkillFailedError("ONB_CAMERA_UNAVAILABLE", NO_STILLS)

    def on_tick(self, ctx: TickContext) -> SkillStatus:
        return SkillStatus(stage="running", progress_pct=100.0 * self.taken / self.count)

    async def cancel(self, io: ExecutorIO) -> None:
        return None  # it never left LOITER

    def resume_point(self, vehicle: VehicleState) -> ResumePoint | None:
        return None
