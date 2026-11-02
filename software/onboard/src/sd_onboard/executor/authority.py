"""Who steers: compare-and-set on the flight mode, and telling a person from a failsafe.

``expected_mode`` is the mode we last asked for and saw confirmed in a HEARTBEAT. A
message that steers is only sent when every one of these holds:

1. the executor is not HELD or FAILSAFE and control is not suspended;
2. no OperatorOverrideNotice arrived in the last ``OVERRIDE_NOTICE_WINDOW_S`` (an
   accepted Resume ends that window at once, ``released``);
3. with an expected mode: the flight controller's mode in a HEARTBEAT at most 1.5 s old
   is the expected one (or the one we are changing to);
4. no other mode change of ours waits for its confirmation (``set_mode`` waits).

A mode change of ours that was sent and then cancelled before a HEARTBEAT showed it (a
reroute that replaces the one switching to AUTO) is still ours for
``MODE_CONFIRM_TIMEOUT_S``: the flight controller may have taken it, and a HEARTBEAT
showing it then is our own doing, not a takeover. It becomes the expected mode.

A mode change nobody here asked for suspends control at once (HELD, ``external``). A
failsafe signal up to 2 s before or after it makes it FAILSAFE instead; a notice from the
ground in that window names who took over.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Final

from sd_mav.modes import CopterMode, mode_name
from sd_mav.vehicle import MODE_FRESH_S, FcTelemetry
from sd_safety.constants import MODE_CONFIRM_TIMEOUT_S, OVERRIDE_NOTICE_WINDOW_S

__all__ = ["Authority", "ModeChange"]

_WINDOW_MS: Final = int(OVERRIDE_NOTICE_WINDOW_S * 1000)
_FRESH_MS: Final = int(MODE_FRESH_S * 1000)
_LATE_MS: Final = int(MODE_CONFIRM_TIMEOUT_S * 1000)


@dataclass(frozen=True, slots=True)
class ModeChange:
    """A mode the flight controller took that we did not ask for."""

    mode: str
    at_mono_ms: int


class Authority:
    def __init__(self) -> None:
        self.expected_mode: CopterMode | None = None
        self.pending_mode: CopterMode | None = None
        # Sent, then cancelled before a HEARTBEAT showed it: ours until then (monotonic ms).
        self.cancelled: dict[CopterMode, int] = {}
        self.notice_mono_ms: int | None = None  # the last OperatorOverrideNotice or takeover
        self.change: ModeChange | None = None  # the change being classified

    # --- compare-and-set --------------------------------------------------------------

    def notice(self, now_ms: int) -> None:
        self.notice_mono_ms = now_ms
        self.pending_mode = None
        self.cancelled.clear()  # a mode of ours that shows after this is not taken as ours

    def sent_and_cancelled(self, mode: CopterMode, now_ms: int) -> None:
        """A mode change sent to the flight controller and cancelled before a HEARTBEAT
        showed it: the flight controller may still take it."""
        self.cancelled[mode] = now_ms + _LATE_MS

    def _ours(self, expected: CopterMode, now_ms: int) -> set[int]:
        """The modes the flight controller may be in because we asked for them."""
        modes = {int(expected)}
        if self.pending_mode is not None:
            modes.add(int(self.pending_mode))
        modes.update(int(m) for m, until in self.cancelled.items() if now_ms <= until)
        return modes

    def adopt_late(self, t: FcTelemetry, now_ms: int) -> CopterMode | None:
        """A fresh HEARTBEAT shows a mode of ours that was cancelled after it went out:
        the flight controller took it, so it is the expected mode now."""
        for mode, until in list(self.cancelled.items()):
            if now_ms > until:
                del self.cancelled[mode]
        fresh = t.fc_heartbeat_mono_ms is not None and now_ms - t.fc_heartbeat_mono_ms <= _FRESH_MS
        if self.expected_mode is None or not fresh or t.fc_custom_mode is None:
            return None
        pending = self.pending_mode
        if t.fc_custom_mode == int(self.expected_mode) or (
            pending is not None and t.fc_custom_mode == int(pending)
        ):
            return None
        late = next((m for m in self.cancelled if int(m) == t.fc_custom_mode), None)
        if late is not None:
            del self.cancelled[late]
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
        if t.fc_custom_mode not in self._ours(self.expected_mode, now_ms):
            return f"mode_{mode_name(t.fc_custom_mode)}"
        return None

    def confirmed(self, mode: CopterMode) -> None:
        self.expected_mode = mode
        self.cancelled.pop(mode, None)
        if self.pending_mode == mode:
            self.pending_mode = None

    def reset(self) -> None:
        """On the ground after landing: nothing is expected any more."""
        self.expected_mode = None
        self.pending_mode = None
        self.cancelled.clear()
        self.change = None

    # --- classification ---------------------------------------------------------------

    def unexpected_mode(self, t: FcTelemetry, now_ms: int) -> ModeChange | None:
        """A fresh HEARTBEAT shows a mode we neither expect nor are changing to."""
        if self.expected_mode is None or t.fc_custom_mode is None:
            return None
        fresh = t.fc_heartbeat_mono_ms is not None and now_ms - t.fc_heartbeat_mono_ms <= _FRESH_MS
        if not fresh:
            return None
        if t.fc_custom_mode in self._ours(self.expected_mode, now_ms):
            return None
        return ModeChange(mode_name(t.fc_custom_mode), now_ms)

    def start_classification(self, change: ModeChange) -> None:
        self.change = change
        self.pending_mode = None
        self.cancelled.clear()

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
