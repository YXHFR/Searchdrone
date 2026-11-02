"""Who steers: compare-and-set on the flight mode, and telling a person from a failsafe.

``expected_mode`` is the mode we last asked for and saw confirmed in a HEARTBEAT. A
message that steers is only sent when every one of these holds:

1. the executor is not HELD or FAILSAFE and control is not suspended;
2. no OperatorOverrideNotice arrived in the last ``OVERRIDE_NOTICE_WINDOW_S`` (an
   accepted Resume ends that window at once, ``released``);
3. with an expected mode: the flight controller's mode in a HEARTBEAT at most 1.5 s old
   is the expected one (or the one we are changing to);
4. no other mode change of ours waits for its confirmation (``set_mode`` waits).

A switch to AUTO of ours that the flight controller accepted and that was cancelled
before a HEARTBEAT showed it (a reroute that replaces the one switching to AUTO) is still
ours until that switch's own confirm deadline (``LateMode``): the flight controller may
have taken it, and a HEARTBEAT showing AUTO then is our own doing, not a takeover. It
becomes the expected mode. Only AUTO: the executor never steers by holding a mode, so a
LOITER, BRAKE, RTL or LAND the drone was not asked for is always a person's or a
failsafe's. It ends at once with a notice from the ground, a change from outside, a hold
the executor enters itself, a later mode of ours that a HEARTBEAT shows after the flight
controller accepted it (it handles mode commands in order), or the RC mode switch moving.

A mode change nobody here asked for suspends control at once (HELD, ``external``). A
failsafe signal up to 2 s before or after it makes it FAILSAFE instead; a notice from the
ground in that window names who took over.
"""

from __future__ import annotations

from dataclasses import dataclass, replace
from typing import Final

from sd_mav.chain import RC_SWITCH_MOVED_US
from sd_mav.modes import CopterMode, mode_name
from sd_mav.vehicle import MODE_FRESH_S, FcTelemetry
from sd_safety.constants import MODE_CONFIRM_TIMEOUT_S, OVERRIDE_NOTICE_WINDOW_S

__all__ = ["Authority", "LateMode", "ModeChange"]

_WINDOW_MS: Final = int(OVERRIDE_NOTICE_WINDOW_S * 1000)
_FRESH_MS: Final = int(MODE_FRESH_S * 1000)
_LATE_MS: Final = int(MODE_CONFIRM_TIMEOUT_S * 1000)
LATE_MODES: Final = frozenset({CopterMode.AUTO})  # the only modes a late HEARTBEAT may show


@dataclass(frozen=True, slots=True)
class ModeChange:
    """A mode the flight controller took that we did not ask for."""

    mode: str
    at_mono_ms: int


@dataclass(frozen=True, slots=True)
class LateMode:
    """A switch of ours the flight controller accepted, cancelled before a HEARTBEAT
    showed it."""

    mode: CopterMode
    until_ms: int  # that switch's own confirm deadline (monotonic ms)
    rc_mode_pwm: int | None  # the RC mode switch when it was sent
    # A newer switch of ours the flight controller accepted (mode, monotonic ms): a
    # HEARTBEAT received after that showing it ends this one.
    newer: tuple[CopterMode, int] | None = None


class Authority:
    def __init__(self) -> None:
        self.expected_mode: CopterMode | None = None
        self.pending_mode: CopterMode | None = None
        self.late: LateMode | None = None  # accepted, then cancelled before it showed
        self.notice_mono_ms: int | None = None  # the last OperatorOverrideNotice or takeover
        self.change: ModeChange | None = None  # the change being classified

    # --- compare-and-set --------------------------------------------------------------

    def notice(self, now_ms: int) -> None:
        self.notice_mono_ms = now_ms
        self.pending_mode = None
        self.late = None  # a mode of ours that shows after this is not taken as ours

    def drop_ours(self) -> None:
        """The executor holds (fc_silent, a limit): no mode change of ours is on the way."""
        self.pending_mode = None
        self.late = None

    def sent_and_cancelled(self, mode: CopterMode, until_ms: int, rc_mode_pwm: int | None) -> None:
        """A switch the flight controller accepted, cancelled before a HEARTBEAT showed it:
        it may show until ``until_ms``, that switch's own confirm deadline. Only AUTO
        (``LATE_MODES``); any other mode that shows later is not taken as ours."""
        self.late = LateMode(mode, until_ms, rc_mode_pwm) if mode in LATE_MODES else None

    def _late(self, t: FcTelemetry, now_ms: int) -> CopterMode | None:
        """The late mode while it may still show: before its deadline, the RC mode switch
        where it was (a person who moved it chose the mode, ``sd_mav.chain``)."""
        late = self.late
        if late is None:
            return None
        rc = t.rc_mode_pwm
        moved = rc is not None and (
            late.rc_mode_pwm is None or abs(rc - late.rc_mode_pwm) > RC_SWITCH_MOVED_US
        )
        seen = t.fc_heartbeat_mono_ms
        newer_shown = (
            late.newer is not None
            and seen is not None
            and seen >= late.newer[1]
            and t.fc_custom_mode == int(late.newer[0])
        )
        if now_ms > late.until_ms or moved or newer_shown:
            self.late = None
            return None
        return late.mode

    def _ours(self, t: FcTelemetry, expected: CopterMode, now_ms: int) -> set[int]:
        """The modes the flight controller may be in because we asked for them."""
        modes = {int(expected)}
        if self.pending_mode is not None:
            modes.add(int(self.pending_mode))
        late = self._late(t, now_ms)
        if late is not None:
            modes.add(int(late))
        return modes

    def adopt_late(self, t: FcTelemetry, now_ms: int) -> CopterMode | None:
        """A fresh HEARTBEAT shows a switch of ours that was cancelled after the flight
        controller accepted it: it took it, so it is the expected mode now."""
        late = self._late(t, now_ms)
        fresh = t.fc_heartbeat_mono_ms is not None and now_ms - t.fc_heartbeat_mono_ms <= _FRESH_MS
        if late is None or self.expected_mode is None or not fresh:
            return None
        pending = self.pending_mode
        if t.fc_custom_mode != int(late) or late in (self.expected_mode, pending):
            return None
        self.late = None
        self.expected_mode = late
        return late

    def notice_recent(self, now_ms: int) -> bool:
        return self.notice_mono_ms is not None and now_ms - self.notice_mono_ms <= _WINDOW_MS

    def released(self) -> None:
        """The pilot's accepted Resume ends the notice window and any classification at
        once: the executor may steer again right away. A new notice opens a new window."""
        self.notice_mono_ms = None
        self.change = None

    def mode_problem(self, t: FcTelemetry, now_ms: int) -> str | None:
        """Why the flight controller's mode does not allow steering, or None."""
        if self.expected_mode is None:
            return None  # on the ground, before the first mode of a takeoff
        fresh = t.fc_heartbeat_mono_ms is not None and now_ms - t.fc_heartbeat_mono_ms <= _FRESH_MS
        if not fresh or t.fc_custom_mode is None:
            return "mode_unknown"
        if t.fc_custom_mode not in self._ours(t, self.expected_mode, now_ms):
            return f"mode_{mode_name(t.fc_custom_mode)}"
        return None

    def confirmed(self, mode: CopterMode, *, accepted_ms: int | None, seen_ms: int | None) -> None:
        """The telemetry shows ``mode``, which we asked for and the flight controller
        accepted at ``accepted_ms`` (None: not a switch of ours, a pilot's Resume). A
        HEARTBEAT received after the accept ends a late mode: the flight controller handles
        mode commands in order, so the cancelled one can no longer show. One older than
        that (the telemetry still shows the mode from before) does not; the first
        HEARTBEAT after the accept that shows ``mode`` then does (``newer``)."""
        self.expected_mode = mode
        late = self.late
        if late is None:
            return
        if (
            late.mode == mode
            or accepted_ms is None
            or (seen_ms is not None and seen_ms >= accepted_ms)
        ):
            self.late = None
        else:
            self.late = replace(late, newer=(mode, accepted_ms))
        if self.pending_mode == mode:
            self.pending_mode = None

    def reset(self) -> None:
        """On the ground after landing: nothing is expected any more."""
        self.expected_mode = None
        self.pending_mode = None
        self.late = None
        self.change = None

    # --- classification ---------------------------------------------------------------

    def unexpected_mode(self, t: FcTelemetry, now_ms: int) -> ModeChange | None:
        """A fresh HEARTBEAT shows a mode we neither expect nor are changing to."""
        if self.expected_mode is None or t.fc_custom_mode is None:
            return None
        fresh = t.fc_heartbeat_mono_ms is not None and now_ms - t.fc_heartbeat_mono_ms <= _FRESH_MS
        if not fresh:
            return None
        if t.fc_custom_mode in self._ours(t, self.expected_mode, now_ms):
            return None
        return ModeChange(mode_name(t.fc_custom_mode), now_ms)

    def start_classification(self, change: ModeChange) -> None:
        self.change = change
        self.pending_mode = None
        self.late = None

    def in_window(self, now_ms: int) -> bool:
        return self.change is not None and now_ms - self.change.at_mono_ms <= _WINDOW_MS

    def failsafe_near_change(self, t: FcTelemetry) -> bool:
        """A failsafe signal within the window before or after the mode change."""
        if self.change is None or t.last_failsafe_mono_ms is None:
            return False
        return abs(t.last_failsafe_mono_ms - self.change.at_mono_ms) <= _WINDOW_MS

    def notice_near_change(self) -> bool:
        if self.change is None or self.notice_mono_ms is None:
            return False
        return abs(self.notice_mono_ms - self.change.at_mono_ms) <= _WINDOW_MS
