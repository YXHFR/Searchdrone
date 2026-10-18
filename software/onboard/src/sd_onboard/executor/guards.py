"""The guards that ``Transition.guard`` names: each picks the target state or refuses.

A guard is a pure function of :class:`GuardInput`. It returns the target state, or a
:class:`Refusal` with the reason code (``None`` = ignore the trigger without an answer).
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from typing import Final

from sd_mav.modes import CopterMode
from sd_onboard.skills._common import is_airborne
from sd_safety.targets import Plane, polygon_from_geojson
from sd_schemas.charter import MissionCharter
from sd_schemas.executor import ExecutorState, ResumePoint
from sd_schemas.vehicle import VehicleState

__all__ = ["GUARDS", "GuardInput", "Refusal", "is_airborne", "run_guard"]

S = ExecutorState
_HOLD_MODES: Final = frozenset({CopterMode.BRAKE.name, CopterMode.LOITER.name})
_HOME_MODES: Final = frozenset({CopterMode.SMART_RTL.name, CopterMode.RTL.name})


@dataclass(frozen=True, slots=True)
class Refusal:
    code: str | None  # None: ignore the trigger quietly


@dataclass(frozen=True, slots=True)
class GuardInput:
    state: ExecutorState
    targets: frozenset[ExecutorState]
    vehicle: VehicleState | None = None
    resume_point: ResumePoint | None = None
    held_by: str | None = None
    in_classification_window: bool = False
    battery_rtl_active: bool = False
    failsafe_cause_gone: bool = False
    old_charter: MissionCharter | None = None
    new_charter: MissionCharter | None = None


Outcome = ExecutorState | Refusal


def _only(g: GuardInput) -> Outcome:
    if len(g.targets) != 1:
        raise ValueError("'always' needs exactly one target")
    return next(iter(g.targets))


def always(g: GuardInput) -> Outcome:
    return g.state if g.state in g.targets else _only(g)


def charter_inflight_compatible(g: GuardInput) -> Outcome:
    """A new charter in the air may only be as strict or stricter (and the same fence)."""
    old, new = g.old_charter, g.new_charter
    if old is None or new is None or not inflight_compatible(old, new):
        return Refusal("ONB_CHARTER_IN_FLIGHT")
    return g.state


def inflight_compatible(old: MissionCharter, new: MissionCharter) -> bool:
    plane = Plane(old.home.lat, old.home.lon)
    old_area = polygon_from_geojson(old.area, plane)
    new_area = polygon_from_geojson(new.area, plane)
    return (
        old_area.is_simple()
        and new_area.is_simple()
        and old_area.outside_by(new_area) == 0.0
        and new.fence_hash == old.fence_hash
        and new.geofence_buffer_m == old.geofence_buffer_m
        and new.alt_max_agl_m <= old.alt_max_agl_m
        and new.max_speed_ms <= old.max_speed_ms
        and new.max_range_from_pilot_m <= old.max_range_from_pilot_m
        and new.alt_min_agl_m >= old.alt_min_agl_m
        and new.battery_rtl_pct >= old.battery_rtl_pct
    )


def airborne_or_ground(g: GuardInput) -> Outcome:
    return S.HOLD if is_airborne(g.vehicle) else S.PREFLIGHT


def resume_target(g: GuardInput) -> Outcome:
    rp = g.resume_point
    if rp is None:
        return Refusal("ONB_NOTHING_TO_RESUME")
    return S.TRANSIT if rp.kind == "goto" or rp.path else S.SURVEY


def within_classification_window(g: GuardInput) -> Outcome:
    if g.held_by == "external" and g.in_classification_window:
        return S.FAILSAFE
    return Refusal(None)


def release_target(g: GuardInput) -> Outcome:
    if g.battery_rtl_active:
        return Refusal("ONB_BATTERY_RTL_ACTIVE")
    v = g.vehicle
    if v is None:
        return Refusal("ONB_RELEASE_MODE_UNSUPPORTED")
    if not v.armed and v.landed_state == "on_ground":
        return S.LANDED
    if is_airborne(v) and v.fc_mode in _HOLD_MODES:
        return S.HOLD
    if is_airborne(v) and v.fc_mode in _HOME_MODES:
        return S.RTL
    return Refusal("ONB_RELEASE_MODE_UNSUPPORTED")


def failsafe_release(g: GuardInput) -> Outcome:
    if not g.failsafe_cause_gone:
        return Refusal("ONB_FAILSAFE_ACTIVE")
    return release_target(g)


def landed_on_ground(g: GuardInput) -> Outcome:
    v = g.vehicle
    if v is not None and v.landed_state == "on_ground" and not v.armed:
        return S.LANDED
    return Refusal(None)


GUARDS: Final[dict[str, Callable[[GuardInput], Outcome]]] = {
    f.__name__: f
    for f in (
        always,
        charter_inflight_compatible,
        airborne_or_ground,
        resume_target,
        within_classification_window,
        release_target,
        failsafe_release,
        landed_on_ground,
    )
}


def run_guard(name: str, g: GuardInput) -> Outcome:
    """Run the guard and make sure it picked one of the row's targets."""
    outcome = GUARDS[name](g)
    if isinstance(outcome, ExecutorState) and outcome not in g.targets:
        raise ValueError(f"guard {name} picked {outcome}, not one of {sorted(g.targets)}")
    return outcome
