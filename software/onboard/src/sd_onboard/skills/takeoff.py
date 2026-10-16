"""``takeoff``: GUIDED, arm, NAV_TAKEOFF to ``alt_m`` above home, then LOITER.

The preconditions (preflight, approved charter, a fresh clearance, standing disarmed on
the ground, a profile that may arm) are checked by the executor before this runs. Done
when the height has been within 1 m of ``alt_m`` for 1 s. Not in the air 15 s after
arming, or not at the height after 60 s: TAKEOFF_TIMEOUT.
"""

from __future__ import annotations

from typing import Any, ClassVar, Final

from sd_mav.modes import CopterMode
from sd_mav.vehicle import MavResult
from sd_onboard.executor.guards import is_airborne
from sd_onboard.skills import register_handler
from sd_onboard.skills.base import (
    ExecutorIO,
    MavAction,
    PlanContext,
    SkillFailedError,
    SkillPlan,
    SkillStatus,
    TakeoffTimeoutError,
    TickContext,
    mode_failed,
)
from sd_schemas.base import LatLonAlt
from sd_schemas.executor import ExecutorState, ResumePoint
from sd_schemas.skills import SkillParams, TakeoffParams
from sd_schemas.vehicle import VehicleState

__all__ = ["TakeoffHandler"]

AIRBORNE_WITHIN_S: Final = 15.0
REACHED_WITHIN_S: Final = 60.0
REACHED_TOL_M: Final = 1.0
REACHED_FOR_S: Final = 1.0
CLIMB_MS: Final = 2.5  # for the expected duration only
TICK_S: Final = 0.1


@register_handler
class TakeoffHandler:
    name: ClassVar[str] = "takeoff"
    TRANSITIONS_EXTRA: ClassVar[dict[Any, Any]] = {}

    def __init__(self, delay_after_arm_s: float = 0.0) -> None:
        self.alt_m = 0.0
        self.delay_after_arm_s = delay_after_arm_s
        self._progress = 0.0

    def plan(self, params: SkillParams, ctx: PlanContext) -> SkillPlan:
        if not isinstance(params, TakeoffParams):
            raise TypeError("expected TakeoffParams")
        self.alt_m = params.alt_m if params.alt_m is not None else ctx.charter.default_alt_agl_m
        home = ctx.charter.home
        # The gate checks the top of the climb straight above home.
        top = LatLonAlt(lat=home.lat, lon=home.lon, alt_amsl_m=ctx.alt_amsl(self.alt_m))
        return SkillPlan(
            path=[top],
            target_state=ExecutorState.TAKEOFF,
            speed_ms=None,
            expected_duration_s=self.alt_m / CLIMB_MS,
            mav_actions=[
                MavAction(kind="set_mode", params={"mode": "GUIDED"}),
                MavAction(kind="arm"),
                MavAction(kind="takeoff", params={"alt_m": self.alt_m}),
                MavAction(kind="set_mode", params={"mode": "LOITER"}),
            ],
        )

    async def start(self, plan: SkillPlan, io: ExecutorIO) -> None:
        if not await io.set_mode(CopterMode.GUIDED):
            raise mode_failed(CopterMode.GUIDED)
        armed = await io.arm()
        if armed is not MavResult.ACCEPTED:
            raise SkillFailedError("ONB_ARM_REFUSED", io.last_statustext() or armed.name)
        io.emit_event("armed", {})
        armed_at = io.clock.monotonic()
        if self.delay_after_arm_s > 0:
            await io.sleep(self.delay_after_arm_s)
        result = await io.takeoff(self.alt_m)
        if result is not MavResult.ACCEPTED:
            raise SkillFailedError("ONB_ARM_REFUSED", io.last_statustext() or result.name)
        reached_since: float | None = None
        while True:
            now = io.clock.monotonic()
            v = io.vehicle
            self._progress = max(0.0, min(100.0, 100 * v.alt_rel_home_m / self.alt_m))
            if v.alt_rel_home_m >= self.alt_m - REACHED_TOL_M:
                reached_since = now if reached_since is None else reached_since
                if now - reached_since >= REACHED_FOR_S:
                    break
            else:
                reached_since = None
            if not is_airborne(v) and now - armed_at > AIRBORNE_WITHIN_S:
                raise TakeoffTimeoutError("Drönaren lyfte inte.")
            if now - armed_at > REACHED_WITHIN_S:
                raise TakeoffTimeoutError("Drönaren nådde inte höjden i tid.")
            await io.sleep(TICK_S)
        if not await io.set_mode(CopterMode.LOITER):
            raise mode_failed(CopterMode.LOITER)
        self._progress = 100.0
        io.emit_event("takeoff_done", {"alt_m": round(io.vehicle.alt_rel_home_m, 1)})

    def on_tick(self, ctx: TickContext) -> SkillStatus:
        return SkillStatus(stage="running", progress_pct=self._progress)

    async def cancel(self, io: ExecutorIO) -> None:
        if is_airborne(io.vehicle):
            await io.set_mode(CopterMode.LOITER)
        # On the ground nothing is sent: ArduPilot disarms by itself (DISARM_DELAY).

    def resume_point(self, vehicle: VehicleState) -> ResumePoint | None:
        return None
