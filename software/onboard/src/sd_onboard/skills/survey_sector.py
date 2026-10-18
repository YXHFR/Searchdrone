"""``survey_sector``: search a polygon in a lawnmower pattern as an AUTO mission.

The mission is home, speed, the search points, a 30 s loiter at the last point and
return-to-launch, so it ends safely even if the onboard computer dies. When the last
search point is reached the executor switches to LOITER (HOLD). ``resume`` flies a new
mission from the point where the search was interrupted.
"""

from __future__ import annotations

import math
from itertools import pairwise
from typing import Any, ClassVar, Final

from shapely import Polygon as ShapelyPolygon

from sd_geo import PatternError, lawnmower
from sd_mav.mission import last_nav_seq, survey_mission
from sd_mav.modes import CopterMode
from sd_onboard.skills import register_handler
from sd_onboard.skills.base import (
    ExecutorIO,
    MavAction,
    PlanContext,
    PlanError,
    SkillFailedError,
    SkillPlan,
    SkillStatus,
    TickContext,
    mode_failed,
)
from sd_safety.constants import DEFAULT_SURVEY_SPACING_M, MISSION_MAX_ITEMS, SURVEY_MAX_AREA_M2
from sd_safety.targets import Plane
from sd_schemas.base import Id, LatLon, LatLonAlt
from sd_schemas.executor import ExecutorState, ResumePoint, SurveyResumePoint
from sd_schemas.skills import SkillParams, SurveySectorParams
from sd_schemas.vehicle import VehicleState

__all__ = ["SurveyHandler", "SurveyRun", "next_index", "progress_pct"]

DEFAULT_SPEED_MS: Final = 7.0
TICK_S: Final = 0.1
FIRST_SEARCH_SEQ: Final = 2  # seq 0 = home, seq 1 = speed


def next_index(seq: int, base_index: int, seq_offset: int) -> int:
    """The search point the vehicle is heading for, from ``MISSION_CURRENT.seq``."""
    return max(base_index, seq - FIRST_SEARCH_SEQ - seq_offset + base_index)


def progress_pct(next_idx: int, n_waypoints: int) -> float:
    if n_waypoints <= 1:
        return 100.0 if next_idx >= n_waypoints else 0.0
    done = max(0, next_idx - 1)
    return min(100.0, 100.0 * done / (n_waypoints - 1))


class SurveyRun:
    """One AUTO mission over ``waypoints[base_index:]`` (after ``interrupted_at`` when
    resumed). Shared by ``survey_sector``, ``follow_path`` and ``resume``."""

    def __init__(
        self,
        command_id: Id,
        waypoints: list[LatLon],
        alt_m: float,
        speed_ms: float,
        *,
        base_index: int = 0,
        interrupted_at: LatLon | None = None,
        target_state: ExecutorState = ExecutorState.SURVEY,
    ) -> None:
        self.command_id = command_id
        self.waypoints = waypoints
        self.alt_m = alt_m
        self.speed_ms = speed_ms
        self.base_index = base_index
        self.interrupted_at = interrupted_at
        self.seq_offset = 0 if interrupted_at is None else 1
        self.next_index = base_index
        self.mission_hash: str | None = None
        self.target_state = target_state

    @property
    def flown_points(self) -> list[LatLon]:
        lead = [] if self.interrupted_at is None else [self.interrupted_at]
        return [*lead, *self.waypoints[self.base_index :]]

    @property
    def last_nav_seq(self) -> int:
        return FIRST_SEARCH_SEQ + self.seq_offset + (len(self.waypoints) - self.base_index) - 1

    def plan(self, ctx: PlanContext) -> SkillPlan:
        if len(self.flown_points) + 4 > MISSION_MAX_ITEMS:
            raise PlanError("ONB_SURVEY_TOO_LARGE", f"{len(self.flown_points)} punkter")
        here = ctx.here()
        alt = ctx.alt_amsl(self.alt_m)
        path = [here, *(LatLonAlt(lat=p.lat, lon=p.lon, alt_amsl_m=alt) for p in self.flown_points)]
        length = sum(
            math.dist(ctx.plane.xy(a.lat, a.lon), ctx.plane.xy(b.lat, b.lon))
            for a, b in pairwise(path)
        )
        return SkillPlan(
            path=path,
            target_state=self.target_state,
            speed_ms=self.speed_ms,
            expected_duration_s=length / self.speed_ms
            + abs(ctx.vehicle.alt_rel_home_m - self.alt_m) / 2,
            mav_actions=[
                MavAction(kind="upload_mission", params={"points": len(self.flown_points)}),
                MavAction(kind="set_mission_current", params={"seq": 1}),
                MavAction(kind="set_mode", params={"mode": "AUTO"}),
                MavAction(kind="set_mode", params={"mode": "LOITER"}),
            ],
        )

    async def fly(self, plan: SkillPlan, io: ExecutorIO, home: LatLonAlt) -> None:
        items = survey_mission(
            home, [(p.lat, p.lon) for p in self.flown_points], self.alt_m, self.speed_ms
        )
        if last_nav_seq(items) != self.last_nav_seq:
            raise RuntimeError("the mission layout does not match the resume arithmetic")
        self.mission_hash = await io.upload_mission(items)
        io.emit_event("mission_uploaded", {"mission_hash": self.mission_hash, "items": len(items)})
        await io.set_mission_current(1)  # the speed item, so AUTO flies at speed_ms
        if not await io.set_mode(CopterMode.AUTO):
            raise mode_failed(CopterMode.AUTO)
        deadline = io.clock.monotonic() + plan.expected_duration_s * 2 + 30
        last = self.last_nav_seq
        while True:
            seq = io.vehicle.mission_seq_current
            if seq is not None and seq >= FIRST_SEARCH_SEQ:
                self.next_index = min(
                    len(self.waypoints) - 1, next_index(seq, self.base_index, self.seq_offset)
                )
                if seq > last:
                    break  # already past the last search point (on the loiter item)
            reached = await io.wait_message(
                lambda m: m.get_type() == "MISSION_ITEM_REACHED" and int(m.seq) >= last, TICK_S
            )
            if reached is not None:
                break
            if io.clock.monotonic() > deadline:
                raise SkillFailedError("ONB_SKILL_TIMEOUT", "Sökningen tog för lång tid.")
        self.next_index = len(self.waypoints)
        if not await io.set_mode(CopterMode.LOITER):
            raise mode_failed(CopterMode.LOITER)

    def progress(self) -> float:
        return progress_pct(self.next_index, len(self.waypoints))

    def resume_point(self, vehicle: VehicleState) -> SurveyResumePoint | None:
        if self.next_index >= len(self.waypoints):
            return None
        return SurveyResumePoint(
            command_id=self.command_id,
            waypoints=self.waypoints,
            alt_m=self.alt_m,
            speed_ms=self.speed_ms,
            base_index=self.base_index,
            seq_offset=self.seq_offset,
            next_index=self.next_index,
            interrupted_at=LatLon(lat=vehicle.lat, lon=vehicle.lon),
            path=self.target_state is ExecutorState.TRANSIT,
        )


def survey_polygon(params: SurveySectorParams, ctx: PlanContext) -> dict[str, Any]:
    if params.polygon is not None:
        return params.polygon
    sector = next((s for s in ctx.charter.sectors if s.sector_id == params.sector_id), None)
    if sector is None:
        raise PlanError("ONB_UNKNOWN_SECTOR", str(params.sector_id))
    return sector.area


def plan_waypoints(
    polygon: dict[str, Any], plane: Plane, spacing_m: float, heading: float | None, start: LatLon
) -> list[LatLon]:
    corners = [plane.xy(lat, lon) for lon, lat in polygon["coordinates"][0][:-1]]
    if ShapelyPolygon(corners).area > SURVEY_MAX_AREA_M2:
        raise PlanError("ONB_SURVEY_TOO_LARGE", "området är för stort")
    try:
        points = lawnmower(corners, spacing_m, heading, plane.xy(start.lat, start.lon))
    except PatternError as exc:
        raise PlanError("SCHEMA_INVALID", str(exc)) from exc
    out = []
    for x, y in points:
        lat, lon = plane.latlon(x, y)
        out.append(LatLon(lat=lat, lon=lon))
    return out


@register_handler
class SurveyHandler:
    name: ClassVar[str] = "survey_sector"
    TRANSITIONS_EXTRA: ClassVar[dict[Any, Any]] = {}

    def __init__(self) -> None:
        self.run: SurveyRun | None = None
        self._home: LatLonAlt | None = None

    def plan(self, params: SkillParams, ctx: PlanContext) -> SkillPlan:
        if not isinstance(params, SurveySectorParams):
            raise TypeError("expected SurveySectorParams")
        if params.pattern != "lawnmower":
            raise PlanError("SKILL_NOT_IMPLEMENTED", f"mönstret {params.pattern[:32]}")
        polygon = survey_polygon(params, ctx)
        spacing = params.spacing_m or DEFAULT_SURVEY_SPACING_M
        alt = params.alt_m if params.alt_m is not None else ctx.charter.default_alt_agl_m
        speed = params.speed_ms or min(DEFAULT_SPEED_MS, ctx.charter.max_speed_ms)
        here = LatLon(lat=ctx.vehicle.lat, lon=ctx.vehicle.lon)
        waypoints = plan_waypoints(polygon, ctx.plane, spacing, params.heading_deg, here)
        self.run = SurveyRun(ctx.command_id, waypoints, alt, speed)
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
        return self.run.resume_point(vehicle) if self.run else None
