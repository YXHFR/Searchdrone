"""The mode steps of STOP and FLY HOME: try each step until the HEARTBEAT shows it.

Used by the ground station's emergency path (as 250/190) and by the onboard computer's
fallback when the ground's MAVLink does not reach the flight controller (as 1/191 with
the fallback origin), so both sides follow exactly the same rules:

* the first ``DO_SET_MODE`` of a step goes out at once and is sent again every
  ``OVERRIDE_RESEND_MS`` until the HEARTBEAT shows the mode;
* a step has failed when the flight controller answers anything but ``ACCEPTED``
  (also ``TEMPORARILY_REJECTED``), or when the HEARTBEAT does not show the mode
  ``STEP_CONFIRM_S`` after the step's first send; then the next step goes at once;
* ``aborted()`` is asked before every single send: a newer action for the vehicle stops
  this chain, so an older action never sends anything late.

After the steps, a mode outside the target set is undone only when it is a **late
order** (``late_order``): a mode the onboard computer itself can command. Any other mode
(ALT_HOLD, STABILIZE, POSHOLD, ...) can only come from a person with an RC transmitter
or from the flight controller, and the RC transmitter goes before everything, so it is
left alone. For STOP a failsafe that began within ``OVERRIDE_NOTICE_WINDOW_S`` also
stands (``failsafe_began_recently``).
"""

from __future__ import annotations

from collections.abc import Callable, Sequence
from dataclasses import dataclass
from typing import Final, Literal

from pymavlink.dialects.v20 import ardupilotmega as mavlink

from sd_mav.allowlist import FC_COMPID
from sd_mav.modes import ONBOARD_MODES, CopterMode
from sd_mav.vehicle import (
    MODE_FRESH_S,
    ControlNotAllowedError,
    ForbiddenMavlinkError,
    MavResult,
    SendOrigin,
    Vehicle,
)
from sd_safety.constants import OVERRIDE_NOTICE_WINDOW_S, OVERRIDE_RESEND_MS

__all__ = [
    "POLL_S",
    "STEP_CONFIRM_S",
    "ChainResult",
    "failsafe_began_recently",
    "heartbeat_fresh",
    "late_order",
    "mode_shown",
    "set_mode_with_fallback",
]

STEP_CONFIRM_S: Final = 1.0  # a step whose mode the HEARTBEAT does not show by then failed
POLL_S: Final = 0.02
_RESEND_S: Final = OVERRIDE_RESEND_MS / 1000


@dataclass(frozen=True, slots=True)
class ChainResult:
    result: Literal["acked", "failed", "aborted"]
    requested: CopterMode | None  # the last step sent
    confirmed: CopterMode | None  # the mode the HEARTBEAT showed when acked
    steps_used: int  # 1 = the first step was enough


def heartbeat_fresh(vehicle: Vehicle) -> bool:
    """A HEARTBEAT from the flight controller within ``MODE_FRESH_S``: its mode and its
    armed flag can be trusted."""
    t = vehicle.telemetry
    seen = t.fc_heartbeat_mono_ms
    return seen is not None and vehicle.clock.monotonic_ms() - seen <= MODE_FRESH_S * 1000


def late_order(mode: int, targets: frozenset[CopterMode]) -> bool:
    """``mode`` is outside ``targets`` and one the onboard computer can command."""
    return mode not in targets and mode in ONBOARD_MODES


def failsafe_began_recently(vehicle: Vehicle) -> bool:
    """A failsafe signal that was quiet before began within ``OVERRIDE_NOTICE_WINDOW_S``
    (a failsafe that has been repeating for a while does not count)."""
    onset = vehicle.telemetry.failsafe_onset_mono_ms
    return (
        onset is not None
        and vehicle.clock.monotonic_ms() - onset <= OVERRIDE_NOTICE_WINDOW_S * 1000
    )


def mode_shown(vehicle: Vehicle) -> int | None:
    """The flight controller's mode in a fresh HEARTBEAT, else ``None``."""
    return vehicle.telemetry.fc_custom_mode if heartbeat_fresh(vehicle) else None


def _set_mode_msg(vehicle: Vehicle, mode: CopterMode) -> object:
    return mavlink.MAVLink_command_long_message(
        vehicle.fc_sysid,
        FC_COMPID,
        mavlink.MAV_CMD_DO_SET_MODE,
        0,
        mavlink.MAV_MODE_FLAG_CUSTOM_MODE_ENABLED,
        int(mode),
        0,
        0,
        0,
        0,
        0,
    )


async def _step(
    vehicle: Vehicle, mode: CopterMode, origin: SendOrigin, aborted: Callable[[], bool]
) -> Literal["acked", "failed", "aborted"]:
    clock = vehicle.clock
    start = clock.monotonic()
    while True:
        if aborted():
            return "aborted"
        if mode_shown(vehicle) == int(mode):
            return "acked"
        if clock.monotonic() - start >= STEP_CONFIRM_S:
            return "failed"
        sent_at = clock.monotonic()
        wait = max(POLL_S, min(_RESEND_S, STEP_CONFIRM_S - (sent_at - start)))
        try:
            result = await vehicle.send(
                _set_mode_msg(vehicle, mode), origin=origin, attempts=1, ack_timeout_s=wait
            )
        except (ForbiddenMavlinkError, ControlNotAllowedError):
            return "failed"
        if result is not None and result not in (MavResult.ACCEPTED, MavResult.TIMEOUT):
            return "failed"  # refused: the next step goes at once
        # Wait for the HEARTBEAT until the next resend is due (or the step's time is up).
        while clock.monotonic() - sent_at < _RESEND_S:
            if mode_shown(vehicle) == int(mode) or aborted():
                break
            if clock.monotonic() - start >= STEP_CONFIRM_S:
                break
            await clock.sleep(POLL_S)


async def set_mode_with_fallback(
    vehicle: Vehicle,
    steps: Sequence[CopterMode],
    *,
    origin: SendOrigin,
    aborted: Callable[[], bool] = lambda: False,
    on_step: Callable[[CopterMode], None] | None = None,
) -> ChainResult:
    """Try ``steps`` in order; ``acked`` with the mode the HEARTBEAT shows, else ``failed``."""
    if not steps:
        raise ValueError("no steps")
    requested: CopterMode | None = None
    for index, mode in enumerate(steps):
        if aborted():
            return ChainResult("aborted", requested, None, index)
        requested = mode
        if on_step is not None:
            on_step(mode)
        outcome = await _step(vehicle, mode, origin, aborted)
        if outcome == "acked":
            return ChainResult("acked", mode, mode, index + 1)
        if outcome == "aborted":
            return ChainResult("aborted", mode, None, index + 1)
    return ChainResult("failed", requested, None, len(steps))
