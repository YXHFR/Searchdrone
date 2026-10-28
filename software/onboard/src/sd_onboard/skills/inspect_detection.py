"""``inspect_detection``: a closer look at a detection, camera straight down.

The point, height and heading are ``InspectDetectionParams.geometry`` (the same
function the ground's gate used): on the edge of the target's keep-out cylinder + 10 m,
on the drone's side, high enough that the target is at most 25 degrees from straight
down, the target along the picture's long side. ``ONB_INSPECT_GEOMETRY`` when no
height in the charter's band (or under the order's ``alt_m``) gets within 28 degrees.

The drone flies there in GUIDED at most ``INSPECT_SPEED_MS`` (``goto``), waits
``SETTLE_S``, takes ``INSPECT_STILLS`` stills ``INSPECT_STILL_INTERVAL_S`` apart, and
hovers (HOLD). An interrupted inspection is not resumed.

A drone inside cylinders first leaves them along a straight leg that goes deeper into
none and enters no other (``sd_safety.keepout.reroute``), ``EXIT_CLEAR_M`` past the last
edge, and stops there. Only then is the point worked out and the leg there checked, from
where the drone is and against the cylinders in force at that moment, through the
executor's whole gate (``ExecutorIO.leg_refusal``); at most ``EXITS_MAX`` times, then the
order fails (``ONB_INSPECT_GEOMETRY``). The keep-out reflex plans the leg being flown
again the same way (:meth:`InspectHandler.replan`).
"""

from __future__ import annotations

import math
from collections.abc import Callable, Sequence
from dataclasses import dataclass, replace
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
from sd_safety.keepout import circles, reroute
from sd_safety.targets import Polygon, Xy, polygon_from_geojson
from sd_schemas.base import Id, LatLonAlt
from sd_schemas.executor import ResumePoint
from sd_schemas.keepout import INSPECT_SPEED_MS, leg_enters
from sd_schemas.skills import InspectDetectionParams, SkillParams, TargetContext
from sd_schemas.vehicle import VehicleState

__all__ = [
    "EXITS_MAX",
    "EXIT_CLEAR_M",
    "INSPECT_STILLS",
    "INSPECT_STILL_INTERVAL_S",
    "SETTLE_S",
    "InspectHandler",
]

SETTLE_S: Final = 2.0
INSPECT_STILLS: Final = 5
INSPECT_STILL_INTERVAL_S: Final = 1.0
EXIT_CLEAR_M: Final = 5.0  # a way out ends this far past the last edge: goto stops within 2 m
EXITS_MAX: Final = 3  # ways out in a row (cylinders that keep coming): then the order fails


@dataclass(frozen=True, slots=True)
class _Leg:
    """One straight leg of an inspection: out of the cylinders, or to the point."""

    lat: float
    lon: float
    alt_m: float
    speed_ms: float
    yaw_deg: float | None
    exit: bool


def _way_out(here: Xy, cyl: Sequence[tuple[Xy, float]], area: Polygon) -> Xy | None:
    """Out of the cylinders ``here`` lies in, ``EXIT_CLEAR_M`` past the last edge along
    the same straight leg (``sd_safety.keepout.reroute``'s way out); None when there is
    none, or no room for the extra metres before another cylinder or the charter's edge."""
    way = reroute([here], cyl, area=area)
    if way is None or len(way) < 2:
        return None
    out = way[1]
    d = math.dist(here, out)
    k = (d + EXIT_CLEAR_M) / d
    far = (here[0] + k * (out[0] - here[0]), here[1] + k * (out[1] - here[1]))
    if any(leg_enters(here, far, c, r) for c, r in cyl) or area.leg_outside_by(here, far) > 0.0:
        return None
    return far


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
        self._params: InspectDetectionParams | None = None
        self._ctx: PlanContext | None = None
        self._leg: _Leg | None = None
        self._leg_plan: SkillPlan | None = None
        self._exits = 0

    def plan(self, params: SkillParams, ctx: PlanContext) -> SkillPlan:
        if not isinstance(params, InspectDetectionParams):
            raise TypeError("expected InspectDetectionParams")
        if params.lat is None or params.lon is None or params.target_radius_m is None:
            raise PlanError("SCHEMA_INVALID", "målets position saknas")
        self._params = params
        leg = self._next(ctx)
        if leg is None:
            raise PlanError("ONB_INSPECT_GEOMETRY")
        path = self._path(ctx, leg)
        duration = self._plan_of(ctx, leg).expected_duration_s
        if leg.exit:
            # The point as it looks from the way out: the whole route, for the gate. The
            # leg there is worked out again from where the drone stops.
            out = replace(
                ctx, vehicle=ctx.vehicle.model_copy(update={"lat": leg.lat, "lon": leg.lon})
            )
            then = self._next(out)
            if then is None:
                raise PlanError("ONB_INSPECT_GEOMETRY")
            path += self._path(out, then)[1:]
            duration += self._plan_of(out, then).expected_duration_s
        self.command_id, self.detection_id = ctx.command_id, params.detection_id
        self._fly(ctx, leg, retarget=False)
        extra = SETTLE_S + (INSPECT_STILLS - 1) * INSPECT_STILL_INTERVAL_S
        return self._plan_of(ctx, leg).model_copy(
            update={"path": path, "expected_duration_s": duration + extra}
        )

    async def start(self, plan: SkillPlan, io: ExecutorIO) -> None:
        if self.command_id is None or self._ctx is None:
            raise RuntimeError("planned before running")
        while True:
            leg, leg_plan = self._leg, self._leg_plan
            if leg is None or leg_plan is None:
                raise RuntimeError("planned before running")
            await self.goto.start(leg_plan, io)  # ends in LOITER, still, at the leg's end
            leg = self._leg
            if leg is None or self.goto.target != (leg.lat, leg.lon, leg.alt_m):
                continue  # the reflex planned the leg again as it ended: fly that one
            if not leg.exit:
                break
            # Out of the cylinders and stopped: the next leg from here, against the
            # cylinders in force now, through the whole gate.
            ctx = replace(self._ctx, vehicle=io.vehicle, keepouts=io.keepouts)
            nxt = self._next(ctx)
            if nxt is None or io.leg_refusal(self._path(ctx, nxt)):
                raise SkillFailedError(
                    "ONB_INSPECT_GEOMETRY", "Ingen säker väg till inspektionspunkten."
                )
            self._fly(ctx, nxt, retarget=False)
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

    def leg_end(self) -> tuple[float, float] | None:
        leg = self._leg
        return None if leg is None or self._at_point else (leg.lat, leg.lon)

    def exiting(self) -> bool:
        return self._leg is not None and self._leg.exit

    def replan(self, ctx: PlanContext, allowed: Callable[[list[LatLonAlt]], bool]) -> bool:
        """The leg again from ``ctx`` (a way out when inside a cylinder, else the point),
        flown from the next tick if ``allowed`` takes its path."""
        if self._leg is None or self._at_point:
            return False
        leg = self._next(ctx)
        if leg is None or not allowed(self._path(ctx, leg)):
            return False
        self._fly(ctx, leg, retarget=True)
        return True

    def _next(self, ctx: PlanContext) -> _Leg | None:
        """The next leg from the drone's position in ``ctx``: out of the cylinders it is
        in, else to the point (``InspectDetectionParams.geometry``); None when neither."""
        params = self._params
        if params is None:
            return None
        v, plane = ctx.vehicle, ctx.plane
        here = plane.xy(v.lat, v.lon)
        cyl = circles(ctx.keepouts, plane)
        if any(math.dist(here, c) < r for c, r in cyl):
            if self._exits >= EXITS_MAX:
                return None
            out = _way_out(here, cyl, polygon_from_geojson(ctx.charter.area, plane))
            if out is None:
                return None
            lat, lon = plane.latlon(*out)
            speed = min(INSPECT_SPEED_MS, ctx.charter.max_speed_ms)
            return _Leg(lat, lon, v.alt_rel_home_m, speed, None, exit=True)
        found = params.geometry(
            TargetContext(
                charter=ctx.charter,
                position=ctx.here(),
                home=ctx.charter.home,
                keepouts=list(ctx.keepouts),
            )
        )
        if found is None:
            return None
        point, alt, yaw, speed = found
        return _Leg(point.lat, point.lon, alt, speed, yaw, exit=False)

    @staticmethod
    def _plan_of(ctx: PlanContext, leg: _Leg) -> SkillPlan:
        return goto_plan(ctx, leg.lat, leg.lon, leg.alt_m, leg.speed_ms)

    def _path(self, ctx: PlanContext, leg: _Leg) -> list[LatLonAlt]:
        return list(self._plan_of(ctx, leg).path)

    def _fly(self, ctx: PlanContext, leg: _Leg, *, retarget: bool) -> None:
        self._ctx, self._leg, self._leg_plan = ctx, leg, self._plan_of(ctx, leg)
        if leg.exit:
            self._exits += 1
        set_leg = self.goto.retarget if retarget else self.goto.prepare
        set_leg(ctx, leg.lat, leg.lon, leg.alt_m, leg.speed_ms, yaw_deg=leg.yaw_deg)

    def on_tick(self, ctx: TickContext) -> SkillStatus:
        if not self._at_point:
            flown = self.goto.on_tick(ctx).progress_pct or 0.0
            return SkillStatus(stage="running", progress_pct=0.8 * flown)
        return SkillStatus(stage="running", progress_pct=80.0 + 20.0 * self.taken / INSPECT_STILLS)

    async def cancel(self, io: ExecutorIO) -> None:
        await io.set_mode(CopterMode.LOITER)

    def resume_point(self, vehicle: VehicleState) -> ResumePoint | None:
        return None  # a closer look is ordered again, not resumed
