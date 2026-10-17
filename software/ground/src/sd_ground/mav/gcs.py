"""What STOP and FLY HOME send to the flight controller (as 250/190).

``run_override`` picks the case from the flight phase (``sd_mav.modes.override_phase``):

=========  =====================  =====================================================
Button     Phase                  Sent
=========  =====================  =====================================================
STOP       airborne               ``STOP_STEPS``: BRAKE, then LOITER, then LAND
STOP       armed on the ground    disarm (no force); refused, still armed after 1 s, or
                                  airborne meanwhile: ``STOP_STEPS`` at once
STOP       disarmed               nothing (``already_active``)
FLY HOME   airborne               ``HOME_STEPS``: SMART_RTL, then RTL, then LAND
FLY HOME   armed on the ground    disarm as for STOP, then ``HOME_STEPS``
FLY HOME   disarmed               nothing (``already_active``)
=========  =====================  =====================================================

A chain that needs its LAND step raises the alarm ``STOP_PATH_FAILED`` or
``HOME_PATH_FAILED``. The phase is worked out again before every resend. Telemetry that
is not fresh counts as airborne, so a stale picture never skips the mode steps.

``guard_override`` watches for ``OVERRIDE_MAX_S`` after the click: whenever the HEARTBEAT
shows a late order (a mode outside the target set that the onboard computer can
command, ``sd_mav.chain.late_order``) the target mode is sent again. A mode only a
person with an RC transmitter or the flight controller can choose, or any mode once the
RC mode switch has moved, ends the watch: the RC transmitter goes before everything.
For STOP a failsafe that began within ``OVERRIDE_NOTICE_WINDOW_S`` also stands and ends
the watch.
"""

from __future__ import annotations

from collections.abc import Callable, Sequence
from dataclasses import dataclass
from typing import Final, Literal

from pymavlink.dialects.v20 import ardupilotmega as mavlink

from sd_mav.allowlist import FC_COMPID
from sd_mav.chain import (
    POLL_S,
    STEP_CONFIRM_S,
    failsafe_began_recently,
    heartbeat_fresh,
    late_order,
    mode_shown,
    rc_switch_moved,
    set_mode_with_fallback,
)
from sd_mav.modes import (
    HOME_MODES,
    HOME_STEPS,
    STOP_MODES,
    STOP_STEPS,
    CopterMode,
    OverridePhase,
    mode_name,
    override_phase,
)
from sd_mav.vehicle import (
    MODE_FRESH_S,
    ControlNotAllowedError,
    ForbiddenMavlinkError,
    MavResult,
    SendOrigin,
    Vehicle,
)
from sd_safety.constants import OVERRIDE_MAX_S, OVERRIDE_RESEND_MS

__all__ = [
    "PATH_FAILED_ALARM",
    "OverrideAction",
    "OverrideResult",
    "guard_override",
    "phase_of",
    "run_override",
    "set_mode_with_fallback",
    "steps_for",
    "target_modes",
]

OverrideAction = Literal["stop", "return_home"]
PATH_FAILED_ALARM: Final = {"stop": "STOP_PATH_FAILED", "return_home": "HOME_PATH_FAILED"}
_RESEND_S: Final = OVERRIDE_RESEND_MS / 1000


@dataclass(frozen=True, slots=True)
class OverrideResult:
    action: OverrideAction
    result: Literal["acked", "already_active", "failed", "aborted"]
    mode_requested: str | None  # e.g. "BRAKE", or "DISARM"
    mode_confirmed: str | None  # the mode the HEARTBEAT showed, or "DISARMED"
    alarm: str | None  # STOP_PATH_FAILED / HOME_PATH_FAILED when the chain needed LAND


def steps_for(action: OverrideAction) -> Sequence[CopterMode]:
    return STOP_STEPS if action == "stop" else HOME_STEPS


def target_modes(action: OverrideAction) -> frozenset[CopterMode]:
    return STOP_MODES if action == "stop" else HOME_MODES


def phase_of(vehicle: Vehicle) -> OverridePhase:
    """The flight phase from fresh telemetry; anything stale counts as airborne."""
    t = vehicle.telemetry
    now = vehicle.clock.monotonic_ms()
    fresh_ms = MODE_FRESH_S * 1000
    beat = heartbeat_fresh(vehicle)
    fresh = (
        beat
        and t.landed_mono_ms is not None
        and now - t.landed_mono_ms <= fresh_ms
        and t.position_mono_ms is not None
        and now - t.position_mono_ms <= fresh_ms
    )
    if not fresh:
        if beat and not t.armed:
            return OverridePhase.ON_GROUND_DISARMED
        return OverridePhase.AIRBORNE
    return override_phase(t.armed, t.landed_state, t.alt_rel_home_m)


def _armed_shown(vehicle: Vehicle) -> bool | None:
    return vehicle.telemetry.armed if heartbeat_fresh(vehicle) else None


def _disarm_msg(vehicle: Vehicle) -> object:
    return mavlink.MAVLink_command_long_message(
        vehicle.fc_sysid, FC_COMPID, mavlink.MAV_CMD_COMPONENT_ARM_DISARM, 0, 0, 0, 0, 0, 0, 0, 0
    )


async def _disarm(
    vehicle: Vehicle, aborted: Callable[[], bool]
) -> Literal["disarmed", "refused", "aborted"]:
    """Disarm without force; ``refused`` when the flight controller says no, the vehicle
    is still armed ``STEP_CONFIRM_S`` after the first send, or it is airborne by now."""
    clock = vehicle.clock
    start = clock.monotonic()
    while clock.monotonic() - start < STEP_CONFIRM_S:
        if aborted():
            return "aborted"
        if _armed_shown(vehicle) is False:
            return "disarmed"
        if phase_of(vehicle) is OverridePhase.AIRBORNE:
            return "refused"
        sent_at = clock.monotonic()
        try:
            result = await vehicle.send(
                _disarm_msg(vehicle),
                origin=SendOrigin.GROUND,
                attempts=1,
                ack_timeout_s=_RESEND_S,
            )
        except (ForbiddenMavlinkError, ControlNotAllowedError):
            return "refused"  # no longer standing on the ground
        if result not in (MavResult.ACCEPTED, MavResult.TIMEOUT):
            return "refused"
        while clock.monotonic() - sent_at < _RESEND_S:
            if _armed_shown(vehicle) is False or aborted():
                break
            await clock.sleep(POLL_S)
    return "disarmed" if _armed_shown(vehicle) is False else "refused"


async def run_override(
    vehicle: Vehicle,
    action: OverrideAction,
    *,
    aborted: Callable[[], bool] = lambda: False,
    on_step: Callable[[str], None] | None = None,
) -> OverrideResult:
    """Send STOP or FLY HOME now; the result once the HEARTBEAT shows it (or not)."""
    phase = phase_of(vehicle)
    if phase is OverridePhase.ON_GROUND_DISARMED:
        return OverrideResult(action, "already_active", None, "DISARMED", None)
    if phase is OverridePhase.ON_GROUND_ARMED:
        if on_step is not None:
            on_step("DISARM")
        outcome = await _disarm(vehicle, aborted)
        if outcome == "aborted":
            return OverrideResult(action, "aborted", "DISARM", None, None)
        if outcome == "disarmed":
            return OverrideResult(action, "acked", "DISARM", "DISARMED", None)
        # Refused, or taking off meanwhile: the mode steps at once.
    chain = await set_mode_with_fallback(
        vehicle,
        steps_for(action),
        origin=SendOrigin.GROUND,
        aborted=aborted,
        on_step=None if on_step is None else (lambda m: on_step(m.name)),
    )
    requested = chain.requested.name if chain.requested is not None else None
    if chain.result == "aborted":
        return OverrideResult(action, "aborted", requested, None, None)
    alarm = PATH_FAILED_ALARM[action] if chain.steps_used >= len(steps_for(action)) else None
    if chain.result == "failed":
        return OverrideResult(action, "failed", requested, None, PATH_FAILED_ALARM[action])
    confirmed = chain.confirmed.name if chain.confirmed is not None else None
    return OverrideResult(action, "acked", requested, confirmed, alarm)


async def guard_override(
    vehicle: Vehicle,
    action: OverrideAction,
    confirmed: CopterMode,
    *,
    clicked_mono: float,
    aborted: Callable[[], bool] = lambda: False,
    on_resend: Callable[[str], None] | None = None,
) -> int:
    """Keep the target for ``OVERRIDE_MAX_S`` after the click; returns how often it resent."""
    clock = vehicle.clock
    targets = target_modes(action)
    rc_before = vehicle.telemetry.rc_mode_pwm  # where the RC mode switch stood
    resent = 0
    while clock.monotonic() - clicked_mono < OVERRIDE_MAX_S:
        if aborted():
            return resent
        if _armed_shown(vehicle) is False:
            return resent  # landed and disarmed: nothing left to hold
        shown = mode_shown(vehicle)
        if shown is not None and shown not in targets:
            if not late_order(shown, targets) or rc_switch_moved(vehicle, rc_before):
                return resent  # a person (RC transmitter) or the flight controller chose it
            if action == "stop" and failsafe_began_recently(vehicle):
                return resent  # the flight controller's own failsafe decided: it stands
            if on_resend is not None:
                on_resend(mode_name(shown))
            resent += 1
            await set_mode_with_fallback(
                vehicle, [confirmed], origin=SendOrigin.GROUND, aborted=aborted
            )
            continue
        await clock.sleep(POLL_S)
    return resent
