"""``inspect_detection``: a closer look at a detection, camera straight down.

The point, height and heading are ``InspectDetectionParams.geometry`` (the same
function the ground's gate used): on the edge of the target's keep-out cylinder + 10 m,
on the drone's side, high enough that the target is at most 25 degrees from straight
down, the target along the picture's long side. ``ONB_INSPECT_GEOMETRY`` when no
height in the charter's band (or under the order's ``alt_m``) gets within 28 degrees.

The drone flies there in GUIDED at most ``INSPECT_SPEED_MS`` (``goto``), waits
``SETTLE_S``, takes ``INSPECT_STILLS`` stills ``INSPECT_STILL_INTERVAL_S`` apart, and
hovers (HOLD). An interrupted inspection is not resumed.
"""

from __future__ import annotations

from typing import Any, ClassVar, Final

from sd_mav.modes import CopterMode
from sd_onboard.skills import register_handler
from sd_onboard.skills.base import (
    ExecutorIO,
    PlanContext,
    PlanError,
    SkillFailedError,
    SkillPlan,
    SkillStatus,
    TickContext,
)
from sd_onboard.skills.capture_stills import NO_STILLS, take_stills
from sd_onboard.skills.goto import GotoHandler, goto_plan
from sd_schemas.base import Id
from sd_schemas.executor import ResumePoint
from sd_schemas.skills import InspectDetectionParams, SkillParams, TargetContext
from sd_schemas.vehicle import VehicleState

__all__ = [
    "INSPECT_STILLS",
    "INSPECT_STILL_INTERVAL_S",
    "SETTLE_S",
    "InspectHandler",
]

SETTLE_S: Final = 2.0
INSPECT_STILLS: Final = 5
INSPECT_STILL_INTERVAL_S: Final = 1.0


@register_handler
class InspectHandler:
    name: ClassVar[str] = "inspect_detection"
    TRANSITIONS_EXTRA: ClassVar[dict[Any, Any]] = {}

    def __init__(self) -> None:
        self.goto = GotoHandler()
        self.command_id: Id | None = None
        self.detection_id: Id | None = None
        self.taken = 0
        self._at_point = False

    def plan(self, params: SkillParams, ctx: PlanContext) -> SkillPlan:
        if not isinstance(params, InspectDetectionParams):
            raise TypeError("expected InspectDetectionParams")
        if params.lat is None or params.lon is None or params.target_radius_m is None:
            raise PlanError("SCHEMA_INVALID", "målets position saknas")
        found = params.geometry(
            TargetContext(
                charter=ctx.charter,
                position=ctx.here(),
                home=ctx.charter.home,
                keepouts=list(ctx.keepouts),
            )
        )
        if found is None:
            raise PlanError("ONB_INSPECT_GEOMETRY")
        point, alt, yaw, speed = found
        self.command_id, self.detection_id = ctx.command_id, params.detection_id
        self.goto.prepare(ctx, point.lat, point.lon, alt, speed, yaw_deg=yaw)
        plan = goto_plan(ctx, point.lat, point.lon, alt, speed)
        extra = SETTLE_S + (INSPECT_STILLS - 1) * INSPECT_STILL_INTERVAL_S
        return plan.model_copy(update={"expected_duration_s": plan.expected_duration_s + extra})

    async def start(self, plan: SkillPlan, io: ExecutorIO) -> None:
        if self.command_id is None:
            raise RuntimeError("planned before running")
        await self.goto.start(plan, io)  # ends in LOITER at the point, facing the heading
        self._at_point = True
        await io.sleep(SETTLE_S)
        taken = await take_stills(io, self.command_id, INSPECT_STILLS, INSPECT_STILL_INTERVAL_S)
        self.taken = len(taken)
        io.emit_event(
            "inspect_done",
            {
                "command_id": str(self.command_id),
                "detection_id": str(self.detection_id),
                "still_ids": [str(s) for s in taken],
            },
        )
        if not taken:
            raise SkillFailedError("ONB_CAMERA_UNAVAILABLE", NO_STILLS)

    def on_tick(self, ctx: TickContext) -> SkillStatus:
        if not self._at_point:
            flown = self.goto.on_tick(ctx).progress_pct or 0.0
            return SkillStatus(stage="running", progress_pct=0.8 * flown)
        return SkillStatus(stage="running", progress_pct=80.0 + 20.0 * self.taken / INSPECT_STILLS)

    async def cancel(self, io: ExecutorIO) -> None:
        await io.set_mode(CopterMode.LOITER)

    def resume_point(self, vehicle: VehicleState) -> ResumePoint | None:
        return None  # a closer look is ordered again, not resumed
