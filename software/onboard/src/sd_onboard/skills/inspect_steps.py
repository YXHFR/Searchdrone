"""The closer look straight over a candidate, step by step (``sd_schemas.inspect``).

Used by ``InspectHandler`` when the charter allows flying over the candidate
(``MissionCharter.overflight_allowed``: ``sim``, ``exercise``, ``live`` with the permit).

* **Step 0**: straight over the order's position at the current height. Below the height
  floor anywhere on the way, the drone first climbs on the spot to the floor + 2 m.
* **Each step**: hover ``STEP_SETTLE_S``, then ``STEP_ROUNDS`` rounds of stills (one from
  every camera, ``ExecutorIO.capture_round``). In every still at most one person detection
  is the target: the one nearest the order's position (the anchor) within the look radius
  (:func:`look_radius_m`). Seen sure enough, the target's estimate moves (a weighted mean
  of the anchor and the matches) and the drone moves over it before the next step.
* **Down**: ``STEP_DOWN_M`` per step, never below the floor in force (``sd_safety.floor``):
  the automatic floor, or the lowest floor after the pilot's yes (``InspectLower``). The
  descent is fed to the flight controller as a target moving down at ``DESCENT_MAX_MS``
  (``DESCENT_SLOW_MS`` below the map's floor without the laser). Every leg goes through
  the executor's whole gate first (``ExecutorIO.leg_refusal``).
* **The end** (``InspectEnd``): a clear picture, the floor reached, the pilot said no or
  did not answer in ``LOWER_OFFER_S``, ``STEPS_MAX`` steps, ``INSPECT_MAX_S``, a cylinder
  in the way (``keepout``), or the laser lost below the map's floor without it (the drone
  climbs back to that floor first). The drone stays at the lowest height reached; the
  executor then hovers for the pilot's decision.
* **Someone else's cylinder** that comes around the drone (a ground team, a person seen
  outside the look radius) while it hovers or goes down: the drone stops and climbs
  straight up over it (``climb_alt``: its top and margin, never above the charter's
  height less ``_CEILING_SPARE_M``), and the steps end (``keepout``). On a leg the
  executor's reflex does the same through :meth:`StepLook.climb`.

Nothing here ever goes below a floor: every goal is clipped to the floor in force, and a
floor that rose above the drone (the laser saw a tree) is climbed to at once. The laser
and the cylinders are watched on every tick of a descent, of the settle and of the
rounds of stills: a laser lost while the step's goal lies below the map's floor without
it stops the descent at once, no lower than that floor.
"""

from __future__ import annotations

import asyncio
import math
from collections.abc import Callable
from dataclasses import dataclass, field
from typing import Final
from uuid import UUID

from sd_mav.modes import CopterMode
from sd_onboard.skills._common import climb_alt, over_a_person, still
from sd_onboard.skills.base import (
    ExecutorIO,
    InspectShot,
    PlanContext,
    PlanError,
    SkillFailedError,
    SkillPlan,
    mode_failed,
)
from sd_onboard.skills.capture_stills import NO_STILLS
from sd_onboard.skills.goto import GotoHandler, goto_plan
from sd_safety.constants import GOTO_ALT_STEP_TOL_M, GOTO_ARRIVE_RADIUS_M
from sd_safety.floor import FloorResult, floor_along, floor_at
from sd_safety.keepout import circles
from sd_schemas.api import KeepoutHoldField
from sd_schemas.base import Id, LatLon, LatLonAlt
from sd_schemas.inspect import (
    CLEAR_PX,
    CLEAR_SCORE,
    DESCENT_MAX_MS,
    DESCENT_SLOW_MS,
    EVENT_LASER_LOST,
    EVENT_LOWER_OFFER,
    INSPECT_MAX_S,
    INSPECT_SPEED_MS,
    LOWER_OFFER_S,
    STEP_DOWN_M,
    STEP_MIN_M,
    STEP_ROUND_INTERVAL_S,
    STEP_ROUNDS,
    STEP_SETTLE_S,
    STEPS_MAX,
    FloorReason,
    InspectEnd,
    InspectLaserLost,
    InspectLowerOffer,
    InspectPhase,
    match_radius_m,
)
from sd_schemas.link import GeoDetectionMsg
from sd_schemas.rangefinder import LaserState
from sd_schemas.skills import InspectDetectionParams

__all__ = [
    "APPROACH_ABOVE_FLOOR_M",
    "LOOK_RADIUS_MAX_M",
    "LOWER_GAIN_MIN_M",
    "Floors",
    "Match",
    "StepLook",
    "estimate",
    "is_clear",
    "look_radius_m",
    "pick",
]

APPROACH_ABOVE_FLOOR_M: Final = 2.0  # below the floor on the way: climb to it + this first
LOOK_RADIUS_MAX_M: Final = 30.0  # the look radius never grows past this (TN-8 review)
LOWER_GAIN_MIN_M: Final = 2.0  # the lowest floor must lie this much lower to offer it
_ANCHOR_SIGMA_MIN_M: Final = 2.0  # the anchor's weight: 1 / max(sigma95, this)^2
_MATCH_SIGMA_MIN_M: Final = 0.5  # a match's weight: score / max(sigma95, this)^2
_RECENTRE_M: Final = GOTO_ARRIVE_RADIUS_M  # the estimate moved further: move over it
_TICK_S: Final = 0.1
_SEND_S: Final = 0.5  # the descent's moving target is sent this often
_DESCENT_SPARE_S: Final = 20.0  # a descent not done by its time + this ends the steps
_FLOOR_TOL_M: Final = 0.5  # a floor this much above the drone is climbed to
_DESCENT_TOL_M: Final = 0.7  # a step down is done this near its height (no creeping up)
# The climb over someone else's cylinder stays this far under the charter's height, as
# the executor's keep-out climb does (the altitude limit's margin).
_CEILING_SPARE_M: Final = 5.0


def look_radius_m(sigma95_m: float | None) -> float:
    """How near the order's position a person detection must be to count as the target
    (and within which the target's own detections make no new cylinder):
    ``match_radius_m`` capped at ``LOOK_RADIUS_MAX_M``."""
    return min(match_radius_m(sigma95_m or 0.0), LOOK_RADIUS_MAX_M)


@dataclass(frozen=True, slots=True)
class Match:
    """The target in one still."""

    still_id: UUID
    camera: str
    lat: float
    lon: float
    score: float
    sigma95_m: float
    px_size: float


def pick(
    shot: InspectShot, anchor: LatLon, radius_m: float, dist: Callable[[float, float], float]
) -> Match | None:
    """The target in ``shot``: the detection nearest the anchor within ``radius_m``
    (``dist(lat, lon)`` is the distance to the anchor); None when none is that near."""
    best: tuple[float, GeoDetectionMsg] | None = None
    for d in shot.detections:
        r = dist(d.lat, d.lon)
        if r <= radius_m and (best is None or r < best[0]):
            best = (r, d)
    if best is None:
        return None
    d = best[1]
    return Match(shot.still_id, shot.camera, d.lat, d.lon, d.score, d.sigma95_m, d.px_size)


def estimate(anchor: LatLon, sigma95_m: float | None, matches: list[Match]) -> LatLon:
    """The target's position: the weighted mean of the anchor (weight 1 / sigma^2) and the
    matches at least ``CLEAR_SCORE`` sure (weight score / sigma^2, as ``fusion.refit``)."""
    s = max(sigma95_m or 0.0, _ANCHOR_SIGMA_MIN_M)
    w = 1.0 / (s * s)
    lat, lon = anchor.lat * w, anchor.lon * w
    for m in matches:
        if m.score < CLEAR_SCORE:
            continue
        sm = max(m.sigma95_m, _MATCH_SIGMA_MIN_M)
        wm = m.score / (sm * sm)
        lat, lon, w = lat + m.lat * wm, lon + m.lon * wm, w + wm
    return LatLon(lat=lat / w, lon=lon / w)


def is_clear(matches: list[Match]) -> bool:
    """A clear picture: the target at least ``CLEAR_SCORE`` sure and ``CLEAR_PX`` big in at
    least two stills of one step (any camera)."""
    good = {m.still_id for m in matches if m.score >= CLEAR_SCORE and m.px_size >= CLEAR_PX}
    return len(good) >= 2


class _End(Exception):  # noqa: N818 - control flow, not an error
    """The steps are over: ``end`` says why."""

    def __init__(self, end: InspectEnd) -> None:
        super().__init__(end)
        self.end: InspectEnd = end


@dataclass
class Floors:
    """The floor over a point now, in metres above home, and why."""

    auto_m: float
    lowest_m: float
    ground_m: float | None  # the ground above home; None without a map
    reason: FloorReason
    laser_confirmed: bool


@dataclass
class StepLook:
    """One closer look straight over the target (see the module's docstring)."""

    params: InspectDetectionParams
    command_id: Id
    start_alt_m: float  # above home, where the look started (the search height)
    anchor: LatLon
    radius_m: float
    step: int = 0
    phase: InspectPhase = "approach"
    end: InspectEnd | None = None
    clear: bool = False
    still_ids: list[UUID] = field(default_factory=list[UUID])
    matches: list[Match] = field(default_factory=list[Match])
    lowest_m: float | None = None  # the lowest height reached, above home
    floors: Floors | None = None  # the floors at the last check (for the progress)
    goto: GotoHandler = field(default_factory=GotoHandler)
    _ctx: PlanContext | None = None
    _legs: list[tuple[float, float, float]] = field(default_factory=list)
    _leg_plan: SkillPlan | None = None
    _moving: tuple[float, float] | None = None  # the end of the leg flown, (lat, lon)
    _stop: bool = False  # the keep-out reflex: stop where the drone is (``keepout``)
    _climb_to: float | None = None  # ... or climb straight up to this, then stop
    _climbing: bool = False
    _lower_yes: asyncio.Event | None = None
    _lower_asked: bool = False
    _pilot_yes: bool = False
    laser_seen: bool = False
    _paused_s: float = 0.0  # the pilot's answer time: not counted in INSPECT_MAX_S
    _t0: float = 0.0

    # --- the plan (no I/O) ----------------------------------------------------------------

    def plan(self, ctx: PlanContext) -> SkillPlan:
        """Step 0: straight over the anchor at the current height, after a climb on the
        spot when the drone is below the automatic floor + 2 m anywhere on the way."""
        self._ctx = ctx
        v = ctx.vehicle
        home = ctx.charter.home.alt_amsl_m
        path = [LatLon(lat=v.lat, lon=v.lon), self.anchor]
        found = floor_along(
            ctx.floor_pack,
            path,
            flight_date=ctx.clock.utc_now().date(),
            laser=None,
            start_alt_amsl_m=home + self.start_alt_m,
        )
        need = found.floor_auto_amsl_m - home + APPROACH_ABOVE_FLOOR_M
        if found.reason == "no_map":
            need = v.alt_rel_home_m  # no map: the look stays at its height
        alt = max(v.alt_rel_home_m, need)
        if alt > ctx.charter.alt_max_agl_m:
            raise PlanError("FLOOR_ABOVE_CEILING")
        speed = min(INSPECT_SPEED_MS, ctx.charter.max_speed_ms)
        self._legs = []
        if alt > v.alt_rel_home_m + GOTO_ALT_STEP_TOL_M:
            self._legs.append((v.lat, v.lon, alt))  # up on the spot first
        self._legs.append((self.anchor.lat, self.anchor.lon, alt))
        lat, lon, a = self._legs[0]
        self.goto.prepare(ctx, lat, lon, a, speed)
        self._leg_plan = goto_plan(ctx, lat, lon, a, speed)
        whole = [
            ctx.here(),
            *(LatLonAlt(lat=p, lon=q, alt_amsl_m=home + h) for p, q, h in self._legs),
        ]
        extra = (STEPS_MAX + 1) * (STEP_SETTLE_S + STEP_ROUNDS * STEP_ROUND_INTERVAL_S)
        dist = math.dist(ctx.plane.xy(v.lat, v.lon), ctx.plane.point(self.anchor))
        return self._leg_plan.model_copy(
            update={
                "path": whole,
                "expected_duration_s": dist / speed + (alt - v.alt_rel_home_m) / 2.0 + extra,
            }
        )

    # --- the run ------------------------------------------------------------------------

    async def run(self, io: ExecutorIO) -> None:
        """The whole look; returns with ``end`` set (the drone hovers in LOITER)."""
        self._t0 = io.clock.monotonic()
        self._lower_yes = asyncio.Event()
        if self._lower_asked:
            self._lower_yes.set()  # a yes that came before the run
        try:
            await self._approach(io)
            while True:
                self.phase = "step"
                await self._look(io)
                if self.clear:
                    raise _End("clear")
                if self.step >= STEPS_MAX:
                    raise _End("steps_max")
                await self._next_step(io)
                self.step += 1
        except _End as done:
            self.end = done.end
        except _LaserLost as lost:
            self._moving = None
            await self.laser_climb(io, lost.floor_m)
        except _InsideCylinder as inside:
            self._moving = None
            await self._climb_out(io, inside.alt_m)
        self._moving = None
        self._note_height(io)

    def lower(self) -> None:
        """The pilot's yes (``InspectLower``): down to the lowest floor."""
        self._lower_asked = True
        if self._lower_yes is not None:
            self._lower_yes.set()

    async def _approach(self, io: ExecutorIO) -> None:
        ctx = self._ctx
        if ctx is None or self._leg_plan is None:
            raise RuntimeError("planned before running")
        speed = min(INSPECT_SPEED_MS, ctx.charter.max_speed_ms)
        for i, (lat, lon, alt) in enumerate(self._legs):
            if i > 0:
                here = _ctx_now(ctx, io)
                path = [here.here(), LatLonAlt(lat=lat, lon=lon, alt_amsl_m=here.alt_amsl(alt))]
                if io.leg_refusal(path):
                    raise SkillFailedError(
                        "ONB_INSPECT_GEOMETRY", "Ingen säker väg till platsen ovanför målet."
                    )
                self.goto.prepare(here, lat, lon, alt, speed)
                self._leg_plan = goto_plan(here, lat, lon, alt, speed)
            await self._fly(io, self._leg_plan, (lat, lon))

    async def _fly(self, io: ExecutorIO, plan: SkillPlan, end: tuple[float, float]) -> None:
        """One goto leg; the reflex may stop it (``keepout``) or turn it into a climb."""
        self._moving = end
        await self.goto.start(plan, io)
        self._moving = None
        self._after_reflex()
        self._check_laser(io)

    def _after_reflex(self) -> None:
        if self._climbing or self._stop:
            raise _End("keepout")

    async def _look(self, io: ExecutorIO) -> None:
        """Settle, then the rounds of stills; the target's matches and estimate."""
        self._note_height(io)
        self._check_time(io)
        await self._wait(io, STEP_SETTLE_S)
        here: list[Match] = []
        shots_seen = 0
        for r in range(STEP_ROUNDS):
            if r > 0:
                await self._wait(io, STEP_ROUND_INTERVAL_S)
            shots = await self._capture(io)
            shots_seen += len(shots)
            for shot in shots:
                if len(self.still_ids) < 200:
                    self.still_ids.append(shot.still_id)
                m = pick(shot, self.anchor, self.radius_m, self._dist_to_anchor)
                if m is not None:
                    here.append(m)
            self._watch(io)
        if shots_seen == 0:
            raise SkillFailedError("ONB_CAMERA_UNAVAILABLE", NO_STILLS)
        self.matches += here
        self.clear = is_clear(here)

    async def _next_step(self, io: ExecutorIO) -> None:
        """The next step's height from the floor in force, the pilot's click below the
        automatic floor, and the leg there through the gate."""
        ctx = self._ctx
        if ctx is None:
            raise RuntimeError("planned before running")
        self._check_time(io)
        self._watch(io)  # a cylinder around the drone: up over it, no deeper into it
        target = estimate(self.anchor, self.params.sigma95_m, self.matches)
        while True:
            here = _ctx_now(ctx, io)
            v = here.vehicle
            fl = self._floors(io, here, target)
            in_force = fl.lowest_m if self._pilot_yes else fl.auto_m
            if in_force > v.alt_rel_home_m + _FLOOR_TOL_M:
                await self._climb(io, in_force)  # the floor rose above the drone
                raise _End("floor")
            goal = max(v.alt_rel_home_m - STEP_DOWN_M, in_force)
            if v.alt_rel_home_m - goal >= STEP_MIN_M:
                break
            lower = max(v.alt_rel_home_m - STEP_DOWN_M, fl.lowest_m)
            if (
                self._pilot_yes
                or fl.lowest_m > fl.auto_m - LOWER_GAIN_MIN_M
                or v.alt_rel_home_m - lower < STEP_MIN_M
            ):
                raise _End("floor")
            await self._offer_lower(io, fl, v.alt_rel_home_m)
        await self._step_to(io, here, target, goal, fl)

    async def _offer_lower(self, io: ExecutorIO, fl: Floors, alt_m: float) -> None:
        """Hover and ask the pilot (``inspect.lower_offer``); no or no answer in
        ``LOWER_OFFER_S`` ends the steps (``lower_declined``)."""
        if self._lower_yes is None:
            raise RuntimeError("planned before running")
        self.phase = "lower_offer"
        ground = fl.ground_m if fl.ground_m is not None else 0.0
        io.emit_event(
            EVENT_LOWER_OFFER,
            InspectLowerOffer(
                command_id=self.command_id,
                now_above_ground_m=_above(alt_m - ground),
                lowest_above_ground_m=_above(fl.lowest_m - ground),
                reason=fl.reason,
            ).model_dump(mode="json"),
        )
        asked = io.clock.monotonic()
        while not self._lower_yes.is_set():
            if io.clock.monotonic() - asked >= LOWER_OFFER_S:
                self._paused_s += io.clock.monotonic() - asked
                raise _End("lower_declined")
            self._watch(io)
            await io.sleep(_TICK_S)
        self._paused_s += io.clock.monotonic() - asked
        self._pilot_yes = True
        self.phase = "step"

    async def _step_to(
        self, io: ExecutorIO, ctx: PlanContext, target: LatLon, goal_m: float, fl: Floors
    ) -> None:
        """Over the estimate at the current height (when it moved), then down to
        ``goal_m``; both legs through the gate first."""
        v = ctx.vehicle
        speed = min(INSPECT_SPEED_MS, ctx.charter.max_speed_ms)
        moved = math.dist(ctx.plane.xy(v.lat, v.lon), ctx.plane.point(target)) > _RECENTRE_M
        lat, lon = (target.lat, target.lon) if moved else (v.lat, v.lon)
        over = LatLonAlt(lat=lat, lon=lon, alt_amsl_m=ctx.alt_amsl(v.alt_rel_home_m))
        down = LatLonAlt(lat=lat, lon=lon, alt_amsl_m=ctx.alt_amsl(goal_m))
        codes = io.leg_refusal([ctx.here(), over, down])
        if codes:
            raise _End("keepout" if "KEEPOUT_VIOLATION" in codes else "floor")
        if moved:
            self.goto.prepare(ctx, lat, lon, v.alt_rel_home_m, speed)
            await self._fly(io, goto_plan(ctx, lat, lon, v.alt_rel_home_m, speed), (lat, lon))
        # Below the map's floor without the laser (only ever with the laser confirmed):
        # slower, so the laser keeps up.
        slow = goal_m < self._map_floor_m(io, ctx, lat, lon) - _FLOOR_TOL_M
        await self._descend(io, lat, lon, goal_m, DESCENT_SLOW_MS if slow else DESCENT_MAX_MS)

    async def _descend(
        self, io: ExecutorIO, lat: float, lon: float, goal_m: float, rate: float
    ) -> None:
        """Down on the spot to ``goal_m``: a target moving down at ``rate``, never below
        ``goal_m``. Stopped by the reflex, a cylinder around the drone or the time limit
        where it is; by the laser lost (with ``goal_m`` below the map's floor without it)
        where it is, or at that floor when it is below it already."""
        if io.vehicle.fc_mode != CopterMode.GUIDED.name and not await io.set_mode(
            CopterMode.GUIDED
        ):
            raise mode_failed(CopterMode.GUIDED)
        self._moving = (lat, lon)
        start = io.clock.monotonic()
        top = io.vehicle.alt_rel_home_m
        deadline = start + max(0.0, top - goal_m) / rate + _DESCENT_SPARE_S
        sent = -math.inf
        try:
            while True:
                now = io.clock.monotonic()
                carrot = max(goal_m, top - rate * (now - start))
                if self._stop or self._climb_to is not None:
                    v = io.vehicle
                    await io.set_position_target(lat, lon, v.alt_rel_home_m)
                    break
                try:  # before the next target goes out: not one more down
                    self._check_laser(io, goal_m=goal_m)
                    self._check_cylinder(io)
                except _LaserLost as lost:
                    hold_m = max(io.vehicle.alt_rel_home_m, lost.floor_m)
                    await io.set_position_target(lat, lon, hold_m)
                    raise
                except _InsideCylinder:
                    await io.set_position_target(lat, lon, io.vehicle.alt_rel_home_m)
                    raise
                if now - sent >= _SEND_S:
                    await io.set_position_target(lat, lon, carrot)
                    sent = now
                v = io.vehicle
                there = abs(v.alt_rel_home_m - goal_m) <= _DESCENT_TOL_M
                if carrot <= goal_m and there and still(v):
                    break
                if now > deadline:
                    await io.set_position_target(lat, lon, v.alt_rel_home_m)
                    self._moving = None
                    raise _End("floor")  # not down in time: no further down
                if self._over_time(io):
                    await io.set_position_target(lat, lon, v.alt_rel_home_m)
                    raise _End("timeout")  # stops where it is
                await io.sleep(_TICK_S)
        finally:
            self._moving = None
        if self._climb_to is not None:
            await self._climb(io, self._climb_to)
        if not await io.set_mode(CopterMode.LOITER):
            raise mode_failed(CopterMode.LOITER)
        self._note_height(io)
        self._after_reflex()

    async def _climb(self, io: ExecutorIO, alt_m: float) -> None:
        """Straight up on the spot to ``alt_m`` (above home), never above the ceiling."""
        ctx = self._ctx
        if ctx is None:
            raise RuntimeError("planned before running")
        alt = min(alt_m, ctx.charter.alt_max_agl_m)
        v = io.vehicle
        if alt <= v.alt_rel_home_m:
            return
        here = _ctx_now(ctx, io)
        speed = min(INSPECT_SPEED_MS, ctx.charter.max_speed_ms)
        self._climbing = True
        self.goto.prepare(here, v.lat, v.lon, alt, speed, from_stop=True)
        self._moving = (v.lat, v.lon)
        await self.goto.start(goto_plan(here, v.lat, v.lon, alt, speed), io)
        self._moving = None

    # --- the reflex (``executor.keepout_reflex.Replans``) ----------------------------------

    def leg_end(self) -> tuple[float, float] | None:
        return self._moving

    def climbing(self) -> bool:
        return self._climbing

    def replan(self, ctx: PlanContext) -> bool:
        """A cylinder on the leg: the steps end where the drone stops (``keepout``). False
        inside a cylinder: the executor then has it climb (:meth:`climb`)."""
        if self._moving is None or self._climbing:
            return False
        v = ctx.vehicle
        here = ctx.plane.xy(v.lat, v.lon)
        if any(math.dist(here, c) < r for c, r in circles(ctx.keepouts, ctx.plane)):
            return False
        self._stop = True
        speed = min(INSPECT_SPEED_MS, ctx.charter.max_speed_ms)
        self.goto.retarget(ctx, v.lat, v.lon, v.alt_rel_home_m, speed, from_stop=True)
        return True

    def climb(self, ctx: PlanContext, alt_m: float) -> bool:
        """Over a person with no way out: straight up to ``alt_m``, then the steps end."""
        if self._moving is None or self._climbing:
            return False
        v = ctx.vehicle
        speed = min(INSPECT_SPEED_MS, ctx.charter.max_speed_ms)
        self._climb_to = alt_m
        self._climbing = True
        self.goto.retarget(ctx, v.lat, v.lon, alt_m, speed, renew_deadline=True, from_stop=True)
        return True

    # --- the floor, the laser and the time ---------------------------------------------------

    def _floors(self, io: ExecutorIO, ctx: PlanContext, target: LatLon) -> Floors:
        v = ctx.vehicle
        laser = io.laser_state()
        lost = self.laser_seen and not (laser is not None and laser.confirmed)
        found = floor_along(
            io.floor_pack,
            [LatLon(lat=v.lat, lon=v.lon), target],
            flight_date=io.clock.utc_now().date(),
            laser=laser,
            start_alt_amsl_m=ctx.alt_amsl(self.start_alt_m),
            laser_lost=lost,
        )
        self.floors = _from(found, ctx.charter.home.alt_amsl_m)
        return self.floors

    def _map_floor_m(self, io: ExecutorIO, ctx: PlanContext, lat: float, lon: float) -> float:
        """The map's floor without the laser over (lat, lon), above home."""
        found = floor_at(
            io.floor_pack,
            lat,
            lon,
            flight_date=io.clock.utc_now().date(),
            laser=None,
            start_alt_amsl_m=ctx.alt_amsl(self.start_alt_m),
            laser_lost=True,
        )
        return found.floor_lowest_amsl_m - ctx.charter.home.alt_amsl_m

    def _check_laser(self, io: ExecutorIO, *, goal_m: float | None = None) -> None:
        """The laser confirmed once in this look and not now, with the drone (or the
        descent's ``goal_m``) below the map's floor without it: stop, climb to that floor
        when below it, and end (``laser_lost``). Raised as ``_LaserLost`` so the climb
        runs outside any leg."""
        laser: LaserState | None = io.laser_state()
        if laser is not None and laser.confirmed:
            self.laser_seen = True
            return
        if not self.laser_seen or self._ctx is None:
            return
        v = io.vehicle
        floor_m = self._map_floor_m(io, self._ctx, v.lat, v.lon)
        lowest = v.alt_rel_home_m if goal_m is None else min(v.alt_rel_home_m, goal_m)
        if lowest >= floor_m - _FLOOR_TOL_M:
            return
        raise _LaserLost(floor_m)

    def _check_cylinder(self, io: ExecutorIO) -> None:
        """Inside a cylinder in force (someone else's: the target's own is suspended):
        raised as ``_InsideCylinder`` with how high to climb over it (None: it cannot)."""
        ctx = self._ctx
        if ctx is None:
            return
        v = io.vehicle
        here = ctx.plane.xy(v.lat, v.lon)
        cyl = circles(io.keepouts, ctx.plane)
        if not over_a_person(here, cyl):
            return
        top = ctx.charter.alt_max_agl_m - _CEILING_SPARE_M
        raise _InsideCylinder(climb_alt(here, cyl, v.alt_rel_home_m, top))

    def _watch(self, io: ExecutorIO) -> None:
        """While the drone hovers: the laser and the cylinders (see the checks)."""
        self._check_laser(io)
        self._check_cylinder(io)

    async def _wait(self, io: ExecutorIO, seconds: float) -> None:
        """Hover ``seconds``, watching every tick (:meth:`_watch`)."""
        until = io.clock.monotonic() + seconds
        while True:
            self._watch(io)
            left = until - io.clock.monotonic()
            if left <= 0.0:
                return
            await io.sleep(min(_TICK_S, left))

    async def _capture(self, io: ExecutorIO) -> list[InspectShot]:
        """One round of stills, watching every tick while the cameras work (up to
        2 x 10 s); the round is dropped when the watch ends the steps."""
        task = asyncio.ensure_future(io.capture_round(self.command_id, self.step))
        try:
            while not task.done():
                self._watch(io)
                await io.sleep(_TICK_S)
            return task.result()
        finally:
            if not task.done():
                task.cancel()

    async def _climb_out(self, io: ExecutorIO, alt_m: float | None) -> None:
        """Someone else's cylinder around the drone: straight up over it (when it can),
        the steps end (``keepout``); the operator is told as at the reflex's climb."""
        data: dict[str, object] = {
            "command_id": str(self.command_id),
            "skill": "inspect_detection",
            str(KeepoutHoldField.INSIDE): True,
        }
        up = alt_m if alt_m is not None and alt_m > io.vehicle.alt_rel_home_m else None
        if up is not None:
            self._climbing = True  # from now: a decision waits for the climb
            data[str(KeepoutHoldField.CLIMB_TO_M)] = round(up, 1)
        io.emit_event("keepout_hold", data)
        if up is not None:
            if io.vehicle.fc_mode != CopterMode.GUIDED.name:
                await io.set_mode(CopterMode.GUIDED)
            await self._climb(io, up)
        if not await io.set_mode(CopterMode.LOITER):
            raise mode_failed(CopterMode.LOITER)
        self.end = "keepout"

    def _over_time(self, io: ExecutorIO) -> bool:
        return io.clock.monotonic() - self._t0 - self._paused_s >= INSPECT_MAX_S

    def _check_time(self, io: ExecutorIO) -> None:
        if self._over_time(io):
            raise _End("timeout")

    def _note_height(self, io: ExecutorIO) -> None:
        alt = io.vehicle.alt_rel_home_m
        self.lowest_m = alt if self.lowest_m is None else min(self.lowest_m, alt)

    def _dist_to_anchor(self, lat: float, lon: float) -> float:
        ctx = self._ctx
        if ctx is None:
            raise RuntimeError("planned before running")
        return math.dist(ctx.plane.xy(lat, lon), ctx.plane.point(self.anchor))

    async def laser_climb(self, io: ExecutorIO, floor_m: float) -> None:
        """The laser lost below the map's floor without it: up there, the steps end."""
        ctx = self._ctx
        v = io.vehicle
        self._climbing = True  # from now: a decision waits for the climb (``climbing``)
        io.emit_event(
            EVENT_LASER_LOST,
            InspectLaserLost(
                command_id=self.command_id,
                above_ground_m=None
                if ctx is None or self.floors is None or self.floors.ground_m is None
                else _above(v.alt_rel_home_m - self.floors.ground_m),
            ).model_dump(mode="json"),
        )
        if io.vehicle.fc_mode != CopterMode.GUIDED.name:
            await io.set_mode(CopterMode.GUIDED)
        await self._climb(io, floor_m)
        if not await io.set_mode(CopterMode.LOITER):
            raise mode_failed(CopterMode.LOITER)
        self.end = "laser_lost"


class _LaserLost(Exception):  # noqa: N818 - control flow, not an error
    def __init__(self, floor_m: float) -> None:
        super().__init__(floor_m)
        self.floor_m = floor_m


class _InsideCylinder(Exception):  # noqa: N818 - control flow, not an error
    def __init__(self, alt_m: float | None) -> None:
        super().__init__(alt_m)
        self.alt_m = alt_m


def _from(found: FloorResult, home_amsl_m: float) -> Floors:
    return Floors(
        auto_m=found.floor_auto_amsl_m - home_amsl_m,
        lowest_m=found.floor_lowest_amsl_m - home_amsl_m,
        ground_m=None if found.ground_amsl_m is None else found.ground_amsl_m - home_amsl_m,
        reason=found.reason,
        laser_confirmed=found.laser_confirmed,
    )


def _ctx_now(ctx: PlanContext, io: ExecutorIO) -> PlanContext:
    """``ctx`` with the vehicle and the cylinders as they are now."""
    from dataclasses import replace

    return replace(ctx, vehicle=io.vehicle, keepouts=io.keepouts)


def _above(m: float) -> float:
    return round(min(1000.0, max(0.0, m)), 1)
