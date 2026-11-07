"""``survey_sector``: search a polygon in a lawnmower pattern as an AUTO mission.

The mission is home, speed, the search points and a hover without end at the last point
(``sd_mav.mission.survey_mission``): it never flies home by itself, so if the onboard
computer stops steering the drone hovers there until someone decides (the ground raises
MISSION_FALLBACK). When the last search point is reached the executor switches to
LOITER (HOLD). ``resume`` flies a new
mission from the point where the search was interrupted.

Every mission goes around the keep-out cylinders in force when it is planned
(``sd_safety.keepout.reroute``); when a new cylinder comes during the flight the
executor plans the rest again the same way (the reflex). A cylinder that comes closer
than the drone can turn away from at its speed: the drone first stops in LOITER, and the
rest is planned again from where it stopped, through the whole gate. Under a charter that
allows overflight a search route flies straight through the cylinders of unconfirmed
persons (``sd_safety.keepout.Passage``), never when its first leg goes down.
"""

from __future__ import annotations

import logging
import math
from collections.abc import Collection, Sequence
from dataclasses import replace
from itertools import pairwise
from typing import Any, ClassVar, Final

from shapely import Polygon as ShapelyPolygon

from sd_geo import PatternError, lawnmower
from sd_mav.mission import SURVEY_EXTRA_ITEMS, last_nav_seq, survey_mission
from sd_mav.modes import CopterMode
from sd_onboard.skills import register_handler
from sd_onboard.skills._common import STOP_WAIT_S, still
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
from sd_safety.constants import (
    DEFAULT_SURVEY_SPACING_M,
    GOTO_ALT_STEP_TOL_M,
    MISSION_MAX_ITEMS,
    SURVEY_MAX_AREA_M2,
)
from sd_safety.keepout import (
    Circle,
    Passage,
    chord_m,
    circles,
    passage_for,
    reroute,
    spent_for,
    split,
    unconfirmed,
    xy_route_violation,
)
from sd_safety.policy import KEEPOUT_PASS_SKILLS
from sd_safety.targets import Plane, Xy, polygon_from_geojson
from sd_schemas.base import Id, LatLon, LatLonAlt
from sd_schemas.executor import ExecutorState, ResumePoint, SurveyResumePoint
from sd_schemas.keepout import Keepout
from sd_schemas.skills import SkillParams, SurveySectorParams
from sd_schemas.vehicle import VehicleState

__all__ = ["SurveyHandler", "SurveyRun", "next_index", "progress_pct"]

DEFAULT_SPEED_MS: Final = 7.0
TICK_S: Final = 0.1
# A reroute that stops first waits at most this long for LOITER to stop the drone (from
# 15 m/s at ArduCopter's braking: about 6 s), then plans from wherever it is.
FIRST_SEARCH_SEQ: Final = 2  # seq 0 = home, seq 1 = speed
# A way around an unconfirmed person's cylinder that leaves out at least this much of the
# search inside it blocks the search (``SurveyRun.blocked``): less is a corner cut.
BLOCKED_MIN_M: Final = 5.0

log = logging.getLogger("sd_onboard.survey")


def _inside_m(route: list[Xy], c: Circle) -> float:
    """How much of the route lies inside the circle."""
    return sum(chord_m(a, b, c[0], c[1]) for a, b in pairwise(route))


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
    resumed). Shared by ``survey_sector``, ``follow_path`` and ``resume``.

    ``pattern`` is the search as planned, before any way around a cylinder, and
    ``origin`` says which of its points each waypoint is (None: a corner of a way around).
    The rest is planned around the cylinders from the pattern (:meth:`pattern_rest`), not
    from the last way around, so a way around a cylinder that is gone is not flown on."""

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
        skill: str | None = None,
    ) -> None:
        self.command_id = command_id
        # The order it flies (``survey_sector``, ``follow_path``, ``resume``): a search route
        # may pass through unconfirmed cylinders (``passage``). None: never (an orbit).
        self.skill = skill
        self.waypoints = waypoints
        self.alt_m = alt_m
        self.speed_ms = speed_ms
        self.base_index = base_index
        self.interrupted_at = interrupted_at
        self.seq_offset = 0 if interrupted_at is None else 1
        self.next_index = base_index
        self.mission_hash: str | None = None
        self.target_state = target_state
        self.pattern: list[LatLon] = list(waypoints)
        self.origin: list[int | None] = list(range(len(waypoints)))
        # The unconfirmed persons' cylinders the last plan went around instead of searching
        # through, and how many metres of the search inside each it left out
        # (:meth:`around_keepouts`); a search route only. A person nobody has looked at
        # yet blocks the search there until someone reviews the candidate.
        self.blocked: list[tuple[Keepout, float]] = []

    @property
    def flown_points(self) -> list[LatLon]:
        lead = [] if self.interrupted_at is None else [self.interrupted_at]
        return [*lead, *self.waypoints[self.base_index :]]

    @property
    def last_nav_seq(self) -> int:
        return FIRST_SEARCH_SEQ + self.seq_offset + (len(self.waypoints) - self.base_index) - 1

    def pattern_rest(self) -> list[LatLon]:
        """The pattern's points after the last one reached: what the search still has to
        cover, whatever ways around the cylinders the waypoints take now. Points a way
        around left out (inside a cylinder) are in it again."""
        done = [o for o in self.origin[: self.next_index] if o is not None]
        return self.pattern[max(done, default=-1) + 1 :]

    def detours(self) -> bool:
        """The waypoints still to fly are not the pattern's rest (a way around)."""
        rest = self.origin[self.next_index :]
        return None in rest or len(rest) != len(self.pattern_rest())

    def passage(self, ctx: PlanContext) -> Passage | None:
        """How the route may cross unconfirmed cylinders from where the drone is now
        (``sd_safety.keepout.passage_for`` at the run's speed); None, every cylinder kept
        out of, when the first leg goes down (no descent inside a cylinder)."""
        if ctx.vehicle.alt_rel_home_m > self.alt_m + GOTO_ALT_STEP_TOL_M:
            return None
        return passage_for(self.skill or "", ctx.charter, self.speed_ms)

    def around_keepouts(self, ctx: PlanContext) -> None:
        """Replace the points still to fly with a way around the cylinders in force, if
        the straight way enters one (``PlanError`` when there is none in the charter).
        With a passage (:meth:`passage`) the route flies straight through unconfirmed
        cylinders where it may."""
        self.blocked = []
        if not ctx.keepouts:
            return
        plane = ctx.plane
        passage = self.passage(ctx)
        hard, soft = split(ctx.keepouts) if passage is not None else (list(ctx.keepouts), [])
        cyl, scyl = circles(hard, plane), circles(soft, plane)
        # The pass the drone is on counts the time it spent inside already.
        spent = spent_for(soft, passage, ctx.keepout_inside_s) if passage is not None else []
        here = plane.xy(ctx.vehicle.lat, ctx.vehicle.lon)
        route = [here, *(plane.point(p) for p in self.flown_points)]
        if xy_route_violation(route, cyl, scyl, passage, spent_m=spent) is None:
            return
        area = polygon_from_geojson(ctx.charter.area, plane)
        out = reroute(route, cyl, area=area, soft=scyl, passage=passage, spent_m=spent)
        if out is None or len(out) < 2:
            raise PlanError("KEEPOUT_VIOLATION", "ingen väg runt förbudscylindrarna")
        try:  # an observation for the ground only: it never stops the way around
            self.blocked = self._blocked(ctx.keepouts, plane, route, out)
        except Exception:
            log.exception("survey: blocked cylinders not worked out")
            self.blocked = []
        points = [LatLon(lat=lat, lon=lon) for lat, lon in (plane.latlon(*p) for p in out[1:])]
        lead: list[int | None] = [] if self.interrupted_at is None else [None]
        origins = dict(zip(route[1:], [*lead, *self.origin[self.base_index :]], strict=True))
        self.origin = [origins.get(p) for p in out[1:]]  # the route's own points, kept as they are
        self.waypoints, self.base_index, self.next_index = points, 0, 0
        self.interrupted_at, self.seq_offset = None, 0

    def _blocked(
        self, keepouts: Sequence[Keepout], plane: Plane, route: list[Xy], out: list[Xy]
    ) -> list[tuple[Keepout, float]]:
        """The unconfirmed cylinders ``out`` searches less of than ``route`` did, by at
        least ``BLOCKED_MIN_M`` (gone around: no passage, or none allowed), with the metres
        left out; none for a route that is not a search."""
        if self.skill not in KEEPOUT_PASS_SKILLS:
            return []
        found: list[tuple[Keepout, float]] = []
        for k in keepouts:
            if not unconfirmed(k):
                continue
            c = (plane.xy(k.lat, k.lon), k.radius_m)
            left_out = _inside_m(route, c) - _inside_m(out, c)
            if left_out >= BLOCKED_MIN_M:
                found.append((k, left_out))
        return found

    def regained_m(
        self, in_force: Collection[Id], plane: Plane, old: list[Xy], new: list[Xy]
    ) -> float:
        """How much more of the search inside the cylinders the last plan went around
        (``blocked``) and that are no longer ``in_force`` the route ``new`` flies than
        ``old`` does: a rejected candidate's area searched again. A cylinder handed over
        under another id is gone around by ``new`` too, and regains nothing."""
        regained = 0.0
        for k, _ in self.blocked:
            if k.keepout_id in in_force:
                continue
            c = (plane.xy(k.lat, k.lon), k.radius_m)
            regained += max(0.0, _inside_m(new, c) - _inside_m(old, c))
        return regained

    def plan(self, ctx: PlanContext) -> SkillPlan:
        self.around_keepouts(ctx)
        if len(self.flown_points) + SURVEY_EXTRA_ITEMS > MISSION_MAX_ITEMS:
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

    def blocked_event(self) -> dict[str, Any]:
        """``executor.survey_blocked``: the cylinders of unconfirmed persons the route now
        flown goes around, so the ground can put their candidates first in the queue."""
        return {
            "command_id": str(self.command_id),
            "skill": self.skill,
            "blocked": [
                {
                    "keepout_id": str(k.keepout_id),
                    "source": k.source,
                    "source_id": str(k.source_id),
                    "skipped_m": round(m, 1),
                }
                for k, m in self.blocked
            ],
        }

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
        if self.blocked:  # once the route flies; an observation, never a reason to stop
            try:
                io.emit_event("survey_blocked", self.blocked_event())
            except Exception:
                log.exception("survey: survey_blocked not sent")
        deadline = io.clock.monotonic() + plan.expected_duration_s * 2 + 30
        last = self.last_nav_seq
        while True:
            seq = io.vehicle.mission_seq_current
            if seq is not None and seq >= FIRST_SEARCH_SEQ:
                self.next_index = min(
                    len(self.waypoints) - 1, next_index(seq, self.base_index, self.seq_offset)
                )
                if seq > last:
                    break  # already past the last search point (on the hover at the end)
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
        """Where a resume goes on. On a way around a cylinder it goes on from the pattern
        (:meth:`pattern_rest`): the resume plans its own way around the cylinders in force
        then, and a way around one lifted meanwhile is not flown."""
        if self.next_index >= len(self.waypoints):
            return None
        waypoints, base, nxt = self.waypoints, self.base_index, self.next_index
        rest = self.pattern_rest()
        if self.detours() and rest:
            waypoints = self.pattern
            base = nxt = len(self.pattern) - len(rest)
        return SurveyResumePoint(
            command_id=self.command_id,
            waypoints=waypoints,
            alt_m=self.alt_m,
            speed_ms=self.speed_ms,
            base_index=base,
            seq_offset=self.seq_offset,
            next_index=nxt,
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
        self._brake = False  # a reroute: stop in LOITER before the new mission goes up
        # A reroute that stops first: the rest as it was before the way around, and the
        # context it is planned again in, from where the drone stopped.
        self._stop_first: tuple[list[LatLon], PlanContext] | None = None

    @classmethod
    def rerouted(
        cls,
        run: SurveyRun,
        home: LatLonAlt,
        *,
        stop_first: tuple[list[LatLon], PlanContext] | None = None,
    ) -> SurveyHandler:
        """The keep-out reflex's handler: the rest of a running order (``survey_sector``,
        ``follow_path`` or ``resume``) as an already planned ``run``. With ``stop_first``
        (the rest still to fly and the context it was planned in) the drone stops before
        the rest is planned again from there (:meth:`_from_stop`)."""
        handler = cls()
        handler.run, handler._home, handler._brake = run, home, True
        handler._stop_first = stop_first
        return handler

    @property
    def stopping(self) -> bool:
        """A reroute that stops first and has not planned its rest from the stop yet: its
        plan there takes the cylinders in force then, so nothing else replans it before.
        Also ends when the drone is not still within ``STOP_WAIT_S`` (it plans from where
        it is then), or when the plan from the stop fails."""
        return self._stop_first is not None

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
        self.run = SurveyRun(ctx.command_id, waypoints, alt, speed, skill=self.name)
        self._home = ctx.charter.home
        return self.run.plan(ctx)

    async def start(self, plan: SkillPlan, io: ExecutorIO) -> None:
        if self.run is None or self._home is None:
            raise RuntimeError("planned before running")
        if self._brake and not await io.set_mode(CopterMode.LOITER):
            raise mode_failed(CopterMode.LOITER)
        if self._stop_first is not None:
            try:
                plan = await self._from_stop(io, *self._stop_first)
            finally:
                self._stop_first = None
        await self.run.fly(plan, io, self._home)

    async def _from_stop(self, io: ExecutorIO, rest: list[LatLon], ctx: PlanContext) -> SkillPlan:
        """Wait in LOITER until the drone stands still, then plan ``rest`` again from
        there against the cylinders in force now, through the whole gate. The way around
        planned while it was moving started where it no longer is."""
        old = self.run
        if old is None:
            raise RuntimeError("planned before running")
        deadline = io.clock.monotonic() + STOP_WAIT_S
        while not still(io.vehicle) and io.clock.monotonic() < deadline:
            await io.sleep(TICK_S)
        run = SurveyRun(
            old.command_id,
            list(rest),
            old.alt_m,
            old.speed_ms,
            target_state=old.target_state,
            skill=old.skill,
        )
        try:
            plan = run.plan(replace(ctx, vehicle=io.vehicle, keepouts=io.keepouts))
        except PlanError as exc:
            no_way = exc.code == "KEEPOUT_VIOLATION"
            detail = "Ingen väg runt förbudscylindrarna." if no_way else exc.detail
            raise SkillFailedError(exc.code, detail) from exc
        if refusal := io.leg_refusal(plan.path):  # the gate's own reason, keep-out or not
            raise SkillFailedError(refusal[0], "Vägen runt förbudscylindrarna nekades.")
        self.run = run
        io.emit_event(
            "keepout_reroute_stopped",
            {"command_id": str(run.command_id), "points": len(plan.path) - 1},
        )
        return plan

    def on_tick(self, ctx: TickContext) -> SkillStatus:
        return SkillStatus(stage="running", progress_pct=self.run.progress() if self.run else None)

    async def cancel(self, io: ExecutorIO) -> None:
        await io.set_mode(CopterMode.LOITER)

    def resume_point(self, vehicle: VehicleState) -> ResumePoint | None:
        return self.run.resume_point(vehicle) if self.run else None
