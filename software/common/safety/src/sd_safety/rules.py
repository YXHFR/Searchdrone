"""The gate's rules, one function each, in the order of the rule table.

Every rule looks at :class:`Facts` (the request, the context and what was worked out
from them once) and returns the findings it has: a reason code plus details. A rule never
raises and never decides on its own; :func:`sd_safety.evaluate` collects every finding.
Where the context is incomplete or contradicts itself, a rule takes the stricter reading.
"""

from __future__ import annotations

import math
from collections.abc import Callable
from dataclasses import dataclass, field
from itertools import pairwise
from typing import Any, Final, Literal

from sd_safety.constants import (
    AIRBORNE_MIN_ALT_M,
    BRAIN_BATTERY_MARGIN_PCT,
    CLEARANCE_VALID_S,
    EXECUTOR_DEFAULT_SPEED_MS,
    MISSION_MAX_ITEMS,
    RATE_ACTION_MIN_GAP_MS,
    RATE_ACTIONS_PER_WINDOW,
    RATE_REPORTS_PER_WINDOW,
    RATE_WINDOW_MS,
    STATE_MAX_AGE_MS,
    TAKEOFF_BATTERY_MARGIN_PCT,
)
from sd_safety.limits import area_reach_from_home_m, fc_area_max_from_home_m
from sd_safety.policy import (
    ALLOWED_OUTSIDE_CHARTER_SKILLS,
    ALWAYS_REQUIRE_APPROVAL,
    CLEARANCE_EXEMPT,
    GROUND_UNCHECKED_TARGETS,
    MODE_A_EXEMPT,
    OPERATOR_ALLOWED_UNDER_LATCH,
    UNDER_OVERRIDE_ALLOWED,
)
from sd_safety.targets import Plane, distance_m, polygon_from_geojson
from sd_schemas.base import LatLon
from sd_schemas.charter import AutonomyMode
from sd_schemas.executor import ExecutorState
from sd_schemas.gate import GateContext, GateRequest
from sd_schemas.mission import MissionState
from sd_schemas.skills import GateTargets, SkillParams, SkillSpec

__all__ = ["CLEAR_AROUND_DRONE", "RULES", "Facts", "Finding", "Rule", "clearance_problem"]

CLEAR_AROUND_DRONE: Final = "clear_around_drone"

Issuer = Literal["operator", "brain", "system"]
_ALL: Final[frozenset[Issuer]] = frozenset({"operator", "brain", "system"})
_BRAIN: Final[frozenset[Issuer]] = frozenset({"brain"})
_OPERATOR: Final[frozenset[Issuer]] = frozenset({"operator"})


@dataclass(frozen=True, slots=True)
class Finding:
    code: str
    details: dict[str, Any] = field(default_factory=dict[str, Any])


@dataclass(frozen=True, slots=True)
class Facts:
    """What every rule may read: the request, the context and facts derived from them."""

    req: GateRequest
    ctx: GateContext
    spec: SkillSpec | None  # None: unknown or forbidden skill
    params: SkillParams | None  # None: the parameters did not validate
    targets: GateTargets | None  # None: no parameters, or working out the targets failed
    plane: Plane | None  # around the charter's home; None without a charter

    @property
    def onboard(self) -> bool:
        return self.ctx.side == "onboard"

    @property
    def moves(self) -> bool:
        """Unknown skills count as moving the vehicle."""
        return self.spec is None or self.spec.moves_vehicle

    @property
    def restricted(self) -> bool:
        """Blocked while a person or the flight controller is in charge."""
        return self.spec is None or self.spec.moves_vehicle or not self.spec.allowed_under_latch

    @property
    def runs_onboard(self) -> bool:
        return self.spec is None or self.spec.runs_on == "onboard"

    @property
    def airborne(self) -> bool:
        v = self.ctx.vehicle
        return (
            v is not None
            and v.armed
            and v.landed_state == "in_air"
            and v.alt_rel_home_m > AIRBORNE_MIN_ALT_M
        )

    @property
    def battery_pct(self) -> float | None:
        return self.ctx.vehicle.battery_pct if self.ctx.vehicle is not None else None

    @property
    def rtl_pct(self) -> float:
        """The charter's level, but never below the mission limit."""
        charter = self.ctx.charter
        limit = self.ctx.limits.battery_rtl_pct
        return limit if charter is None else max(charter.battery_rtl_pct, limit)

    @property
    def mode_a(self) -> bool:
        """Approve each step if the context or the charter says so."""
        modes = {self.ctx.autonomy_mode}
        if self.ctx.charter is not None:
            modes.add(self.ctx.charter.autonomy_mode)
        return AutonomyMode.APPROVE_EACH in modes


RuleFn = Callable[[Facts], list[Finding]]


@dataclass(frozen=True, slots=True)
class Rule:
    rule_id: str
    issuers: frozenset[Issuer]
    ground_only: bool
    check: RuleFn


def _one(condition: bool, code: str, **details: Any) -> list[Finding]:
    return [Finding(code, details)] if condition else []


# --- schema, registry and who may ask -------------------------------------------------------


def _skill_unknown(f: Facts) -> list[Finding]:
    return _one(f.spec is None, "SKILL_UNKNOWN", skill=f.req.skill)


def _schema_invalid(f: Facts) -> list[Finding]:
    if f.spec is None:
        return []
    if f.params is None:
        return [Finding("SCHEMA_INVALID", {"params": "invalid"})]
    if f.req.planned_path is not None and len(f.req.planned_path) > MISSION_MAX_ITEMS:
        return [Finding("SCHEMA_INVALID", {"planned_path": "too_long"})]
    t = f.targets
    if t is None:
        return [Finding("SCHEMA_INVALID", {"targets": "error"})]
    if f.spec.moves_vehicle and not t.resolved and not _targets_checked_elsewhere(f):
        return [Finding("SCHEMA_INVALID", {"targets": "unresolved"})]
    for area in t.areas:
        lon, lat = area["coordinates"][0][0]
        plane = f.plane or Plane(lat, lon)
        if not polygon_from_geojson(area, plane).is_simple():
            return [Finding("SCHEMA_INVALID", {"area": "self_intersecting"})]
    return []


def _targets_checked_elsewhere(f: Facts) -> bool:
    """The route exists only onboard: the ground leaves it, onboard checks the handler's path."""
    if f.onboard:
        return bool(f.req.planned_path)  # an empty path proves nothing
    return f.req.skill in GROUND_UNCHECKED_TARGETS


def _not_implemented(f: Facts) -> list[Finding]:
    return _one(f.spec is not None and f.spec.status == "stub", "SKILL_NOT_IMPLEMENTED")


def _forbidden_action(f: Facts) -> list[Finding]:
    return _one(f.spec is not None and not f.spec.ai_allowed, "FORBIDDEN_ACTION")


def _autonomy_change(f: Facts) -> list[Finding]:
    return _one("autonomy_mode" in f.req.params, "AUTONOMY_CHANGE_FORBIDDEN")


def _not_allowed(f: Facts) -> list[Finding]:
    charter = f.ctx.charter
    return _one(
        f.spec is not None
        and charter is not None
        and f.req.skill not in charter.allowed_skills
        and f.req.skill not in ALLOWED_OUTSIDE_CHARTER_SKILLS,
        "SKILL_NOT_ALLOWED",
    )


def _ai_off(f: Facts) -> list[Finding]:
    return _one(not f.ctx.ai_enabled and f.req.skill != "report", "AI_DISABLED")


_GROUND_ONLY_STATES: Final = frozenset(
    {MissionState.DRAFT, MissionState.CHARTER_PROPOSED, MissionState.LANDED}
)
_NO_FLIGHT_SKILLS: Final = frozenset({"report", "mark_location", "propose_charter_change"})


def _mission_match(f: Facts) -> list[Finding]:
    """The order must be for the charter's own mission and vehicle."""
    charter = f.ctx.charter
    other = charter is not None and (
        f.req.mission_id != charter.mission_id or f.req.vehicle_id != charter.vehicle_id
    )
    return _one(other, "MISSION_STATE_INVALID", mission="other")


def _mission_state(f: Facts) -> list[Finding]:
    state = f.ctx.mission_state
    skill = f.req.skill
    if state is None or state == MissionState.CLOSED:
        bad = True
    elif state in _GROUND_ONLY_STATES:
        bad = skill not in _NO_FLIGHT_SKILLS
    elif state == MissionState.READY:
        bad = skill not in _NO_FLIGHT_SKILLS and skill != "takeoff"
    else:
        bad = False
    return _one(bad, "MISSION_STATE_INVALID", state=state)


# --- latches: a person or the flight controller is in charge --------------------------------


def _operator_latch(f: Facts) -> list[Finding]:
    latched = f.ctx.latch is not None or f.ctx.stop_latch
    return _one(latched and f.req.skill not in OPERATOR_ALLOWED_UNDER_LATCH, "RELEASE_REQUIRED")


_LATCH_CODES: Final = {
    "stop": "RESUME_NEEDS_OPERATOR",
    "watchdog_stop": "RESUME_NEEDS_OPERATOR",
    "return_home": "OPERATOR_OVERRIDE_ACTIVE",
    "watchdog_rtl": "OPERATOR_OVERRIDE_ACTIVE",
    "battery_rtl": "FAILSAFE_ACTIVE",
}


def _brain_latch(f: Facts) -> list[Finding]:
    if not f.restricted:
        return []
    latch = f.ctx.latch
    if f.onboard or latch is None:
        return _one(f.ctx.stop_latch, "RESUME_NEEDS_OPERATOR")
    return [Finding(_LATCH_CODES[latch.action], {"latch": latch.action})]


def _paused_by_person(f: Facts) -> list[Finding]:
    return _one(
        f.ctx.mission_state == MissionState.PAUSED and f.ctx.paused_by != "brain" and f.restricted,
        "RESUME_NEEDS_OPERATOR",
    )


def _returning_by_person(f: Facts) -> list[Finding]:
    return _one(
        f.ctx.mission_state == MissionState.RETURNING
        and f.ctx.returning_cause not in ("brain", "battery", "failsafe")
        and f.req.skill not in UNDER_OVERRIDE_ALLOWED,
        "OPERATOR_OVERRIDE_ACTIVE",
    )


def _failsafe(f: Facts) -> list[Finding]:
    ctx = f.ctx
    active = (
        (
            ctx.mission_state == MissionState.RETURNING
            and ctx.returning_cause in ("battery", "failsafe")
        )
        or ctx.mission_state == MissionState.FAILSAFE
        or ctx.executor_state == ExecutorState.FAILSAFE
    )
    return _one(active and f.req.skill not in UNDER_OVERRIDE_ALLOWED, "FAILSAFE_ACTIVE")


def _held(f: Facts) -> list[Finding]:
    held = f.ctx.executor_state == ExecutorState.HELD or (
        not f.onboard and f.ctx.mission_state == MissionState.HELD
    )
    return _one(held and f.req.skill not in UNDER_OVERRIDE_ALLOWED, "CONTROL_HELD")


def _manual(f: Facts) -> list[Finding]:
    return _one(f.ctx.manual_control and f.restricted, "MANUAL_CONTROL_ACTIVE")


# --- charter, fence, link and state picture -------------------------------------------------


def _charter_approved(f: Facts) -> list[Finding]:
    charter = f.ctx.charter
    unapproved = charter is None or charter.approved_at is None or charter.approved_by is None
    return _one(f.moves and unapproved, "CHARTER_NOT_APPROVED")


def _charter_installed(f: Facts) -> list[Finding]:
    return _one(f.moves and not f.ctx.charter_installed, "CHARTER_NOT_INSTALLED")


def _charter_version(f: Facts) -> list[Finding]:
    charter = f.ctx.charter
    asked = f.req.charter_version
    mismatch = asked is not None if charter is None else asked != charter.version
    return _one(mismatch, "CHARTER_VERSION_MISMATCH", charter_version=asked)


def _fence_verified(f: Facts) -> list[Finding]:
    return _one(f.req.skill == "takeoff" and not f.ctx.fence_verified, "FENCE_NOT_VERIFIED")


def _beyond_fc_fence(f: Facts) -> list[Finding]:
    charter = f.ctx.charter
    if not f.moves or charter is None or f.req.skill == "return_home":
        return []  # flying home is always allowed geometrically
    limit = fc_area_max_from_home_m(f.ctx.limits)
    worst = area_reach_from_home_m(charter)
    return _one(
        worst > limit,
        "AREA_BEYOND_FC_FENCE",
        fc_fence_distance_m=_finite(worst),
        fc_fence_limit_m=limit,
    )


def _link(f: Facts) -> list[Finding]:
    return _one(f.runs_onboard and not f.ctx.sdlink_up, "LINK_DOWN")


def _stale(f: Facts) -> list[Finding]:
    if not f.runs_onboard:
        return []
    ctx = f.ctx
    stale = (
        ctx.vehicle is None
        or ctx.vehicle_age_ms is None
        or ctx.vehicle_age_ms > STATE_MAX_AGE_MS
        or ctx.snapshot_age_ms is None
        or ctx.snapshot_age_ms > STATE_MAX_AGE_MS
    )
    return _one(stale, "STATE_STALE", vehicle_age_ms=ctx.vehicle_age_ms)


def _based_on(f: Facts) -> list[Finding]:
    if not f.moves:
        return []
    ref, ctx = f.req.based_on, f.ctx
    changed = (
        ref is None
        or ctx.state_epoch is None
        or ctx.last_transition_state_seq is None
        or ref.state_epoch != ctx.state_epoch
        or ref.state_seq < ctx.last_transition_state_seq
    )
    return _one(changed, "STATE_CHANGED")


def _when(f: Facts) -> list[Finding]:
    when = f.spec.allowed_when if f.spec is not None else "any"
    bad = (when == "on_ground" and f.airborne) or (when == "airborne" and not f.airborne)
    return _one(bad, "INVALID_TRANSITION", airborne=f.airborne)


# --- clearance and takeoff ------------------------------------------------------------------


def clearance_problem(f: Facts) -> str | None:
    """Why there is no valid "clear around the drone" for this order, or None.

    Public so that the ground station's clearance service can apply the same test."""
    c, req, ctx = f.ctx.clearance, f.req, f.ctx
    if c is None or c.consumed or c.origin == "remote":
        return "CLEARANCE_REQUIRED"
    charter = ctx.charter
    if (
        c.mission_id != req.mission_id
        or c.vehicle_id != req.vehicle_id
        or charter is None
        or c.charter_id != charter.charter_id
        or c.charter_version != charter.version
    ):
        return "CLEARANCE_REQUIRED"
    age = ctx.now_mono_ms - c.received_mono_ms
    if age < 0:
        return "CLEARANCE_REQUIRED"  # a clock that went backwards proves nothing
    if age > min(c.valid_for_ms, CLEARANCE_VALID_S * 1000):
        return "CLEARANCE_EXPIRED"
    return None


def _clearance_needed(f: Facts) -> bool:
    return not f.airborne and f.req.skill not in CLEARANCE_EXEMPT


def _clearance_reject(f: Facts) -> list[Finding]:
    """The AI in mode B, and every order onboard, needs a clearance that is already there."""
    if not _clearance_needed(f):
        return []
    if not f.onboard and (f.req.issuer_kind != "brain" or f.mode_a):
        return []
    problem = clearance_problem(f)
    return [] if problem is None else [Finding(problem, {"skill": f.req.skill})]


def _clearance_card(f: Facts) -> list[Finding]:
    """The operator, and the AI in mode A, may give it with a tick on the approval card."""
    if f.onboard or not _clearance_needed(f):
        return []
    if f.req.issuer_kind == "brain" and not f.mode_a:
        return []
    if f.req.phase == "approve" and CLEAR_AROUND_DRONE in f.req.approval_confirmations:
        return []
    return _one(clearance_problem(f) is not None, "NEEDS_CLEARANCE")


def _takeoff(f: Facts) -> list[Finding]:
    if f.req.skill != "takeoff":
        return []
    v = f.ctx.vehicle
    why: list[str] = []
    if v is None:
        why.append("no_vehicle_state")
    else:
        if v.armed:
            why.append("armed")
        if v.landed_state != "on_ground":
            why.append("not_on_ground")
        if v.battery_pct is None:
            why.append("battery_unknown")
        elif v.battery_pct < f.rtl_pct + TAKEOFF_BATTERY_MARGIN_PCT:
            why.append("battery_low")
    if f.ctx.preflight_ok is not True:
        why.append("preflight_not_ok")
    return _one(bool(why), "TAKEOFF_PRECONDITION", takeoff=why)


# --- battery --------------------------------------------------------------------------------


def _battery_floor(f: Facts) -> list[Finding]:
    pct = f.battery_pct
    exempt = f.spec is not None and f.spec.exempt_from_battery_rule
    return _one(
        not exempt and pct is not None and pct <= f.rtl_pct,
        "BATTERY_RESERVE",
        battery_pct=pct,
        rtl_pct=f.rtl_pct,
    )


def _brain_battery(f: Facts) -> list[Finding]:
    exempt = f.spec is not None and f.spec.exempt_from_battery_rule
    if not f.moves or exempt:
        return []
    pct = f.battery_pct
    if pct is None:
        return [Finding("BATTERY_UNKNOWN")]
    margin = f.rtl_pct + BRAIN_BATTERY_MARGIN_PCT
    return _one(pct <= margin, "BATTERY_RESERVE", battery_pct=pct, rtl_pct=f.rtl_pct)


# --- where the order goes -------------------------------------------------------------------


def _planned_alts(f: Facts) -> list[float]:
    """Heights above home of the onboard handler's path (needs the charter's home)."""
    path, charter = f.req.planned_path, f.ctx.charter
    if path is None or charter is None:
        return []
    return [p.alt_amsl_m - charter.home.alt_amsl_m for p in path]


def _altitude(f: Facts) -> list[Finding]:
    alts = [*(f.targets.alts_m if f.targets is not None else []), *_planned_alts(f)]
    if not alts:
        return []
    out: list[Finding] = []
    top, bottom = max(alts), min(alts)
    limit = f.ctx.limits.max_alt_agl_m
    out += _one(top > limit, "ALT_ABOVE_LIMIT", alt_m=top, alt_limit_m=limit)
    charter = f.ctx.charter
    if charter is not None:
        out += _one(top > charter.alt_max_agl_m, "ALT_ABOVE_CHARTER", alt_m=top)
        out += _one(bottom < charter.alt_min_agl_m, "ALT_BELOW_CHARTER", alt_m=bottom)
    return out


def _route_points(f: Facts) -> tuple[list[LatLon], list[list[LatLon]], list[dict[str, Any]]]:
    """Single points, routes (each a list of corners) and areas the order touches."""
    t = f.targets
    points = list(t.points) if t is not None else []
    routes = [list(t.path)] if t is not None and t.path else []
    areas = list(t.areas) if t is not None else []
    if f.req.planned_path:
        routes.append([LatLon(lat=p.lat, lon=p.lon) for p in f.req.planned_path])
    return points, routes, areas


def _outside_charter(f: Facts) -> list[Finding]:
    charter, plane = f.ctx.charter, f.plane
    points, routes, areas = _route_points(f)
    if charter is None or plane is None or not (points or routes or areas):
        return []
    fence = polygon_from_geojson(charter.area, plane)
    if not fence.is_simple():
        return [Finding("OUTSIDE_CHARTER", {"charter_area": "invalid"})]
    # Stops at the first piece found outside: its distance is a lower bound for the detail.
    for p in [*points, *(r[0] for r in routes)]:
        xy = plane.point(p)
        if not fence.contains(xy):
            return _outside(fence.boundary_distance(xy))
    for route in routes:
        for a, b in pairwise(route):
            by = fence.leg_outside_by(plane.point(a), plane.point(b))
            if by > 0.0:
                return _outside(by)
    for area in areas:
        by = fence.outside_by(polygon_from_geojson(area, plane))
        if by > 0.0:
            return _outside(by)
    return []


def _outside(by: float) -> list[Finding]:
    return [Finding("OUTSIDE_CHARTER", {"outside_by_m": round(by, 1)})]


def _finite(value: float) -> float | None:
    """Details are stored as JSON, which has no infinity."""
    return value if math.isfinite(value) else None


def _pilot_distance(f: Facts) -> list[Finding]:
    """The farthest point of a straight leg or a polygon from a point is one of its corners."""
    points, routes, areas = _route_points(f)
    corners = [*points, *(p for r in routes for p in r)]
    for area in areas:
        corners += [LatLon(lat=lat, lon=lon) for lon, lat in area["coordinates"][0]]
    charter, vehicle = f.ctx.charter, f.ctx.vehicle
    pilots: list[LatLon] = []
    limit = f.ctx.limits.max_range_from_pilot_m
    if charter is not None:
        pilots.append(charter.pilot_position)
        limit = min(limit, charter.max_range_from_pilot_m)
    if vehicle is not None:
        pilots.append(vehicle.pilot_position)
    if not corners or not pilots:
        return []
    worst = max(distance_m(p.lat, p.lon, c.lat, c.lon) for p in pilots for c in corners)
    return _one(
        worst > limit,
        "PILOT_DISTANCE",
        pilot_distance_m=_finite(round(worst, 1)),
        pilot_limit_m=limit,
    )


def _speed(f: Facts) -> list[Finding]:
    t = f.targets
    if t is None or not (t.points or t.path or t.areas):
        return []
    # No speed given: the executor flies at its default speed, which must fit as well.
    speed = t.speed_ms if t.speed_ms is not None else EXECUTOR_DEFAULT_SPEED_MS
    out = _one(speed > f.ctx.limits.max_speed_ms, "SPEED_ABOVE_LIMIT", speed_ms=speed)
    charter = f.ctx.charter
    if charter is not None:
        out += _one(speed > charter.max_speed_ms, "SPEED_ABOVE_CHARTER", speed_ms=speed)
    return out


# --- the AI's manners -----------------------------------------------------------------------


def _absent(f: Facts) -> list[Finding]:
    return _one(f.moves and not f.ctx.operator_present, "OPERATOR_ABSENT")


def _pending(f: Facts) -> list[Finding]:
    """At the click the card being approved is itself pending: already limited at submit."""
    return _one(f.req.phase != "approve" and f.ctx.brain_pending_approvals >= 1, "PENDING_LIMIT")


def _rate(f: Facts) -> list[Finding]:
    if f.req.phase == "approve":
        return []  # the order was counted when it was submitted
    now = f.ctx.now_mono_ms
    rate_class = f.spec.rate_class if f.spec is not None else "action"

    def recent(stamps: tuple[int, ...], window: int) -> int:
        return sum(1 for t in stamps if now - t < window)  # future stamps count too

    if rate_class == "action":
        actions = f.ctx.recent_brain_actions_mono_ms
        busy = (
            recent(actions, RATE_ACTION_MIN_GAP_MS) >= 1
            or recent(actions, RATE_WINDOW_MS) >= RATE_ACTIONS_PER_WINDOW
        )
    elif rate_class == "report":
        busy = recent(f.ctx.recent_brain_reports_mono_ms, RATE_WINDOW_MS) >= RATE_REPORTS_PER_WINDOW
    else:
        busy = False
    return _one(busy, "RATE_LIMIT")


def _approval_mode_a(f: Facts) -> list[Finding]:
    return _one(f.mode_a and f.req.skill not in MODE_A_EXEMPT, "NEEDS_APPROVAL_MODE_A")


def _approval_skill(f: Facts) -> list[Finding]:
    needs = f.req.skill in ALWAYS_REQUIRE_APPROVAL or (
        f.spec is not None and f.spec.requires_approval_in_b
    )
    return _one(needs, "NEEDS_APPROVAL_SKILL")


def _approval_rule(fn: RuleFn) -> RuleFn:
    """Approval codes are met by the click itself, and never become a card onboard (the
    executor checks the approval's hash instead)."""

    def check(f: Facts) -> list[Finding]:
        if f.onboard or f.req.phase == "approve":
            return []
        return fn(f)

    return check


RULES: Final[tuple[Rule, ...]] = (
    Rule("R-SCHEMA", _ALL, False, _skill_unknown),
    Rule("R-SCHEMA", _ALL, False, _schema_invalid),
    Rule("R-IMPL", _ALL, False, _not_implemented),
    Rule("R-AI", _BRAIN, False, _forbidden_action),
    Rule("R-AUTONOMY", _BRAIN, False, _autonomy_change),
    Rule("R-ALLOWED", _BRAIN, False, _not_allowed),
    Rule("R-AI-OFF", _BRAIN, True, _ai_off),
    Rule("R-MISSION", _ALL, False, _mission_match),
    Rule("R-MISSION", _ALL, True, _mission_state),
    Rule("R-LATCH", _OPERATOR, True, _operator_latch),
    Rule("R-LATCH-SPARR", _BRAIN, False, _brain_latch),
    Rule("R-LATCH", _BRAIN, True, _paused_by_person),
    Rule("R-LATCH", _BRAIN, True, _returning_by_person),
    Rule("R-LATCH", _BRAIN, False, _failsafe),
    Rule("R-LATCH", _BRAIN, False, _held),
    Rule("R-MANUAL", _BRAIN, False, _manual),
    Rule("R-CHARTER", _ALL, False, _charter_approved),
    Rule("R-CHARTER", _ALL, True, _charter_installed),
    Rule("R-CHARTER", _BRAIN, False, _charter_version),
    Rule("R-FENCE", _ALL, False, _fence_verified),
    Rule("R-FC-FENCE", _ALL, False, _beyond_fc_fence),
    Rule("R-LINK", _ALL, True, _link),
    Rule("R-STALE", _ALL, False, _stale),
    Rule("R-BASED-ON", _BRAIN, False, _based_on),
    Rule("R-WHEN", _ALL, False, _when),
    Rule("R-CLEAR", _ALL, False, _clearance_reject),
    Rule("R-CLEAR", _ALL, False, _clearance_card),
    Rule("R-TAKEOFF", _ALL, False, _takeoff),
    Rule("R-BATT", _ALL, False, _battery_floor),
    Rule("R-BATT", _BRAIN, False, _brain_battery),
    Rule("R-ALT", _ALL, False, _altitude),
    Rule("R-AREA", _ALL, False, _outside_charter),
    Rule("R-PILOT", _ALL, False, _pilot_distance),
    Rule("R-SPEED", _ALL, False, _speed),
    Rule("R-ABSENT", _BRAIN, True, _absent),
    Rule("R-PENDING", _BRAIN, True, _pending),
    Rule("R-RATE", _BRAIN, True, _rate),
    Rule("R-APPROVE", _BRAIN, False, _approval_rule(_approval_mode_a)),
    Rule("R-APPROVE", _BRAIN, False, _approval_rule(_approval_skill)),
)
