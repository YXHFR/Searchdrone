"""Reading the ground station's ``Situation``: blocks, limits and the start rule."""

from __future__ import annotations

import datetime as dt
from dataclasses import dataclass
from typing import Literal

from sd_brain.memory import PendingRequest, pending_request
from sd_schemas.commander import MissionMemory, Situation

__all__ = [
    "Limits",
    "MotionBlock",
    "current_limits",
    "fresh_request",
    "motion_block",
    "takeoff_ok",
]

MotionBlock = Literal["stop", "return", "battery", "failsafe", "held", "manual"]


def motion_block(s: Situation) -> MotionBlock | None:
    """A human decision or an emergency holds: the AI may only report and mark locations."""
    if s.override is not None:
        if s.override.action in ("stop", "watchdog_stop"):
            return "stop"
        if s.override.action == "battery_rtl":
            return "battery"
        return "return"  # return_home, watchdog_rtl
    if s.vehicle.manual_control:
        return "manual"  # the pilot steers by hand (Ta över)
    if s.mission_status == "PAUSED" and s.paused_by in ("operator", "system"):
        return "stop"
    if s.mission_status == "RETURNING":
        if s.returning_cause == "battery":
            return "battery"
        if s.returning_cause in ("operator", "system"):
            return "return"
        if s.returning_cause == "failsafe":
            return "failsafe"
    if s.mission_status == "FAILSAFE":
        return "failsafe"
    if s.mission_status == "HELD":
        return "held"
    return None


@dataclass(frozen=True)
class Limits:
    alt_min_agl_m: float
    alt_max_agl_m: float
    default_alt_agl_m: float
    max_speed_ms: float
    max_range_from_pilot_m: float
    battery_rtl_pct: float
    mission_alt_max_agl_m: float  # the mission's own limits (the highest allowed)
    mission_max_speed_ms: float


def current_limits(s: Situation) -> Limits:
    """The charter's values when a charter is approved, otherwise the mission's limits."""
    lim = s.limits
    c = s.charter
    if c is not None:
        return Limits(
            alt_min_agl_m=c.alt_min_agl_m,
            alt_max_agl_m=min(c.alt_max_agl_m, lim.max_alt_agl_m),
            default_alt_agl_m=c.default_alt_agl_m,
            max_speed_ms=min(c.max_speed_ms, lim.max_speed_ms),
            max_range_from_pilot_m=min(c.max_range_from_pilot_m, lim.max_range_from_pilot_m),
            battery_rtl_pct=max(c.battery_rtl_pct, lim.battery_rtl_pct),
            mission_alt_max_agl_m=lim.max_alt_agl_m,
            mission_max_speed_ms=lim.max_speed_ms,
        )
    return Limits(
        alt_min_agl_m=15.0,
        alt_max_agl_m=lim.max_alt_agl_m,
        default_alt_agl_m=min(40.0, lim.max_alt_agl_m),
        max_speed_ms=lim.max_speed_ms,
        max_range_from_pilot_m=lim.max_range_from_pilot_m,
        battery_rtl_pct=lim.battery_rtl_pct,
        mission_alt_max_agl_m=lim.max_alt_agl_m,
        mission_max_speed_ms=lim.max_speed_ms,
    )


def fresh_request(
    s: Situation, mem: MissionMemory, now: dt.datetime, ttl_s: float
) -> PendingRequest | None:
    """The operator's start request, if one is waiting and not older than ``ttl_s``.

    Age counts from the later of the request and the charter's approval, so it does not
    expire while the operator reads the charter card.
    """
    req = pending_request(mem)
    if req is None:
        return None
    since = req.requested_at
    if s.charter is not None and s.charter.approved_at is not None:
        since = max(since, s.charter.approved_at)
    return req if (now - since).total_seconds() <= ttl_s else None


def takeoff_ok(
    s: Situation,
    mem: MissionMemory,
    now: dt.datetime,
    *,
    policy: Literal["request_then_clearance", "clearance_only"],
    ttl_s: float,
) -> bool:
    """Whether ``takeoff`` may be offered to the model in this turn.

    Approved and installed charter with ``takeoff`` allowed, a valid "clear around the
    drone" from a person, the vehicle on the ground, and (by default) a fresh start
    request from the operator. The gate checks all of it again.
    """
    if s.mission_status != "READY" or s.charter is None:
        return False
    if "takeoff" not in s.charter.allowed_skills:
        return False
    until = s.takeoff_clearance_valid_until
    if not s.brain_may_takeoff or until is None or until <= now:
        return False
    v = s.vehicle
    if not (v.landed and not v.armed and v.connected):
        return False
    if policy == "request_then_clearance":
        return fresh_request(s, mem, now, ttl_s) is not None
    return True
