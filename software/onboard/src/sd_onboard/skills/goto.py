"""``goto``: fly to a point in GUIDED and hover there (also a pure height change).

Above a 5 m height difference the vehicle first climbs or sinks where it is, then flies
straight. The target is sent again every 2 s, and at once when it changes
(:meth:`GotoHandler.retarget`). Arrived = within 2 m, height within 1.5 m and slower
than 0.5 m/s for 2 s; then LOITER.
"""

from __future__ import annotations

import math
from typing import Any, ClassVar, Final

from sd_mav.modes import CopterMode
from sd_onboard.skills import register_handler
from sd_onboard.skills._common import horizontal_m, still
from sd_onboard.skills.base import (
    ExecutorIO,
    MavAction,
    PlanContext,
    SkillFailedError,
    SkillPlan,
    SkillStatus,
    TickContext,
    mode_failed,
)
from sd_safety.constants import GOTO_ALT_STEP_TOL_M, GOTO_ARRIVE_RADIUS_M
from sd_safety.targets import Plane
from sd_schemas.base import Id, LatLonAlt
from sd_schemas.executor import ExecutorState, GotoResumePoint, ResumePoint
from sd_schemas.skills import GotoParams, SkillParams
from sd_schemas.vehicle import VehicleState

__all__ = ["GotoHandler", "goto_plan"]

ALT_STEP_FIRST_M: Final = 5.0  # a larger height change is flown on the spot first
RESEND_S: Final = 2.0
ARRIVED_FOR_S: Final = 2.0
DEFAULT_SPEED_MS: Final = 10.0
CLIMB_MS: Final = 2.0  # for the expected duration only
TICK_S: Final = 0.1


def goto_plan(
    ctx: PlanContext,
    lat: float,
    lon: float,
    alt_m: float,
    speed_ms: float,
) -> SkillPlan:
    here = ctx.here()
    v = ctx.vehicle
    path: list[LatLonAlt] = [here]
    step = abs(v.alt_rel_home_m - alt_m) > ALT_STEP_FIRST_M
    if step:
        path.append(ctx.here(alt_m))
    path.append(LatLonAlt(lat=lat, lon=lon, alt_amsl_m=ctx.alt_amsl(alt_m)))
    dist = math.dist(ctx.plane.xy(v.lat, v.lon), ctx.plane.xy(lat, lon))
    duration = dist / speed_ms + abs(v.alt_rel_home_m - alt_m) / CLIMB_MS
    return SkillPlan(
        path=path,
        target_state=ExecutorState.TRANSIT,
        speed_ms=speed_ms,
        expected_duration_s=duration,
        mav_actions=[
            MavAction(kind="set_mode", params={"mode": "GUIDED"}),
            MavAction(kind="change_speed", params={"speed_ms": speed_ms}),
            MavAction(kind="set_position_target", params={"lat": lat, "lon": lon, "alt_m": alt_m}),
            MavAction(kind="set_mode", params={"mode": "LOITER"}),
        ],
    )


@register_handler
class GotoHandler:
    name: ClassVar[str] = "goto"
    TRANSITIONS_EXTRA: ClassVar[dict[Any, Any]] = {}

    def __init__(self) -> None:
        self.command_id: Id | None = None
        self.target: tuple[float, float, float] | None = None
        self.speed_ms = DEFAULT_SPEED_MS
        self.yaw_deg: float | None = None  # None: the flight controller chooses the heading
        self._plane: Plane | None = None
        self._start_dist = 0.0
        self._progress = 0.0
        self._resend = False  # the target changed: send it on the next tick
        self._deadline = math.inf  # monotonic seconds: not there by then, the order fails

    def plan(self, params: SkillParams, ctx: PlanContext) -> SkillPlan:
        if not isinstance(params, GotoParams):
            raise TypeError("expected GotoParams")
        speed = params.speed_ms or min(DEFAULT_SPEED_MS, ctx.charter.max_speed_ms)
        self.prepare(ctx, params.lat, params.lon, params.alt_m, speed)
        return goto_plan(ctx, params.lat, params.lon, params.alt_m, speed)

    def prepare(
        self,
        ctx: PlanContext,
        lat: float,
        lon: float,
        alt_m: float,
        speed_ms: float,
        *,
        yaw_deg: float | None = None,
    ) -> None:
        self.command_id = ctx.command_id
        self.target = (lat, lon, alt_m)
        self.speed_ms = speed_ms
        self.yaw_deg = yaw_deg
        self._plane = ctx.plane
        self._start_dist = horizontal_m(ctx.plane, ctx.vehicle, lat, lon)

    def retarget(
        self,
        ctx: PlanContext,
        lat: float,
        lon: float,
        alt_m: float,
        speed_ms: float,
        *,
        yaw_deg: float | None = None,
    ) -> None:
        """A new end for the leg being flown, from ``ctx``: sent on the next tick."""
        self.prepare(ctx, lat, lon, alt_m, speed_ms, yaw_deg=yaw_deg)
        self._resend = True
        self._deadline = self._deadline_for(
            ctx.clock.monotonic(), goto_plan(ctx, lat, lon, alt_m, speed_ms)
        )

    async def start(self, plan: SkillPlan, io: ExecutorIO) -> None:
        if self.target is None or self._plane is None:
            raise RuntimeError("planned before running")
        if io.vehicle.fc_mode != CopterMode.GUIDED.name and not await io.set_mode(
            CopterMode.GUIDED
        ):
            raise mode_failed(CopterMode.GUIDED)
        await io.change_speed(self.speed_ms)
        self._deadline = self._deadline_for(io.clock.monotonic(), plan)
        if abs(io.vehicle.alt_rel_home_m - self.target[2]) > ALT_STEP_FIRST_M:
            # Climb or sink on the spot first.
            v = io.vehicle
            spot = (v.lat, v.lon)
            last = -math.inf
            while abs(io.vehicle.alt_rel_home_m - self.target[2]) > GOTO_ALT_STEP_TOL_M:
                if self._resend or io.clock.monotonic() - last >= RESEND_S:
                    self._resend = False
                    await io.set_position_target(spot[0], spot[1], self.target[2], self.yaw_deg)
                    last = io.clock.monotonic()
                self._timeout(io)
                await io.sleep(TICK_S)
        last = -math.inf
        arrived_since: float | None = None
        while True:
            now = io.clock.monotonic()
            lat, lon, alt = self.target
            if self._resend:
                arrived_since = None  # a new end: arrived there afresh
            if self._resend or now - last >= RESEND_S:
                self._resend = False
                await io.set_position_target(lat, lon, alt, self.yaw_deg)
                last = now
            v = io.vehicle
            dist = horizontal_m(self._plane, v, lat, lon)
            if self._start_dist > 0:
                self._progress = max(0.0, min(100.0, 100 * (1 - dist / self._start_dist)))
            here = (
                dist <= GOTO_ARRIVE_RADIUS_M and abs(v.alt_rel_home_m - alt) <= GOTO_ALT_STEP_TOL_M
            )
            if here and still(v):
                arrived_since = now if arrived_since is None else arrived_since
                if now - arrived_since >= ARRIVED_FOR_S:
                    break
            else:
                arrived_since = None
            self._timeout(io)
            await io.sleep(TICK_S)
        self._progress = 100.0
        if not await io.set_mode(CopterMode.LOITER):
            raise mode_failed(CopterMode.LOITER)

    @staticmethod
    def _deadline_for(now: float, plan: SkillPlan) -> float:
        return now + plan.expected_duration_s * 2 + 30

    def _timeout(self, io: ExecutorIO) -> None:
        if io.clock.monotonic() > self._deadline:
            raise SkillFailedError("ONB_SKILL_TIMEOUT", "Drönaren kom inte fram i tid.")

    def on_tick(self, ctx: TickContext) -> SkillStatus:
        return SkillStatus(stage="running", progress_pct=self._progress)

    async def cancel(self, io: ExecutorIO) -> None:
        await io.set_mode(CopterMode.LOITER)

    def resume_point(self, vehicle: VehicleState) -> ResumePoint | None:
        if self.command_id is None or self.target is None:
            return None
        lat, lon, alt = self.target
        return GotoResumePoint(
            command_id=self.command_id, lat=lat, lon=lon, alt_m=alt, speed_ms=self.speed_ms
        )
