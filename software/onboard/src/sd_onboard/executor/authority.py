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
the executor enters itself, the RC mode switch moving, any HEARTBEAT received after its
own answer that shows another mode (the flight controller was in AUTO when it answered, so
it left it), or any HEARTBEAT received after the answer that settled a newer mode command
of ours: the flight controller handles commands in order, so a HEARTBEAT received after
that answer left after the command was handled, and AUTO in it is from outside. A lost
answer whose mode a HEARTBEAT already shows counts as received with that HEARTBEAT.
"After" is the order the messages were received in, never a time: two messages can share
a millisecond. The vehicle records each answer as it is received and every HEARTBEAT
carries the newest one before it (``FcTelemetry.mode_answer_seq``) and the newest one that
showed another mode (``other_mode_seq``), so it holds before the executor runs again.
After AUTO was taken as ours the RC mode switch is still watched for ``RC_RECHECK_MS``
past that (or to the deadline): a pilot's RC AUTO whose HEARTBEAT came before the next
RC_CHANNELS sample is then a takeover after all.

Assumed: the link delivers the flight controller's messages in the order it sent them
(one serial or radio link to it). Were they reordered, a GUIDED HEARTBEAT sent before the
AUTO answer but received after it would end the late AUTO, and the AUTO that shows next
would be taken as a takeover: a false hold, never a takeover taken as ours.

Known residual: a newer mode command of ours whose answer is lost, then a GCS switch to
AUTO before any HEARTBEAT showed our change. Nothing received yet tells that the late AUTO
is over, so that AUTO is taken as ours. It is bounded by the late record's deadline (at
most ``MODE_CONFIRM_TIMEOUT_S`` after the answer, 2 s) and to AUTO, and an RC takeover is
always caught: the mode switch moving ends the record, and it is watched after adoption.

A mode change nobody here asked for suspends control at once (HELD, ``external``). A
failsafe signal up to 2 s before or after it makes it FAILSAFE instead; a notice from the
ground in that window names who took over.
"""

from __future__ import annotations

from dataclasses import dataclass, replace
from typing import Final

from sd_mav.chain import switch_moved
from sd_mav.modes import CopterMode, mode_name
from sd_mav.vehicle import MODE_FRESH_S, FcTelemetry
from sd_safety.constants import OVERRIDE_NOTICE_WINDOW_S

__all__ = ["Authority", "LateMode", "ModeChange"]

_WINDOW_MS: Final = int(OVERRIDE_NOTICE_WINDOW_S * 1000)
_FRESH_MS: Final = int(MODE_FRESH_S * 1000)
# RC_CHANNELS comes at 4 Hz: after a late AUTO is taken, this long covers several samples.
RC_RECHECK_MS: Final = 1000
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
    # The receive order of the answer that settled it. A HEARTBEAT received after a newer
    # answer (``FcTelemetry.mode_answer_seq`` past this) ends it, whatever mode it shows.
    answer_seq: int


class Authority:
    def __init__(self) -> None:
        self.expected_mode: CopterMode | None = None
        self.pending_mode: CopterMode | None = None
        self.late: LateMode | None = None  # accepted, then cancelled before it showed
        self.adopted: LateMode | None = None  # a late AUTO taken as ours: the RC switch watch
        self.notice_mono_ms: int | None = None  # the last OperatorOverrideNotice or takeover
        self.change: ModeChange | None = None  # the change being classified

    # --- compare-and-set --------------------------------------------------------------

    def notice(self, now_ms: int) -> None:
        self.notice_mono_ms = now_ms
        self.pending_mode = None
        self.late = None  # a mode of ours that shows after this is not taken as ours
        self.adopted = None

    def drop_ours(self) -> None:
        """The executor holds (fc_silent, a limit): no mode change of ours is on the way."""
        self.pending_mode = None
        self.late = None
        self.adopted = None

    def sent_and_cancelled(
        self, mode: CopterMode, until_ms: int, rc_mode_pwm: int | None, answer_seq: int
    ) -> None:
        """A switch the flight controller accepted, cancelled before a HEARTBEAT showed it:
        it may show until ``until_ms``, that switch's own confirm deadline. Only AUTO
        (``LATE_MODES``); any other mode that shows later is not taken as ours.
        ``answer_seq`` is the receive order of the answer that accepted it."""
        late = LateMode(mode, until_ms, rc_mode_pwm, answer_seq)
        self.late = late if mode in LATE_MODES else None

    def _late(self, t: FcTelemetry, now_ms: int) -> CopterMode | None:
        """The late mode while it may still show: before its deadline, the RC mode switch
        where it was (a person who moved it chose the mode, ``sd_mav.chain``), and no
        newer answer of ours received before this HEARTBEAT: the flight controller handles
        mode commands in order, so after a newer one was answered the late one cannot
        show. Nor any HEARTBEAT after its own answer that shows another mode: the flight
        controller left it. "Before" and "after" are the receive order, recorded as
        received (``sd_mav.Vehicle``)."""
        late = self.late
        if late is None:
            return None
        moved = switch_moved(late.rc_mode_pwm, t.rc_mode_pwm)
        # Receive order stands for send order: the link does not reorder (module docstring).
        after_newer = t.mode_answer_seq is not None and t.mode_answer_seq > late.answer_seq
        # The flight controller was in the late mode when it sent that answer: a HEARTBEAT
        # received after it that showed another mode means it left it, so the mode showing
        # later is from outside. This holds even when a newer answer of ours was lost, and
        # also when that HEARTBEAT was followed by a newer one before this check ran.
        other = t.other_mode_seq if t.fc_custom_mode == int(late.mode) else t.fc_heartbeat_seq
        left = other is not None and other > late.answer_seq
        if now_ms > late.until_ms or moved or after_newer or left:
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
        controller accepted it: it took it, so it is the expected mode now. Never while a
        newer mode command of ours is on the way (``pending_mode``): its answer may already
        be on its way too, and the late AUTO stays ours meanwhile (``_ours``), so waiting
        holds nothing."""
        late = self._late(t, now_ms)
        fresh = t.fc_heartbeat_mono_ms is not None and now_ms - t.fc_heartbeat_mono_ms <= _FRESH_MS
        if late is None or self.expected_mode is None or not fresh:
            return None
        if self.pending_mode is not None:
            return None
        if t.fc_custom_mode != int(late) or late == self.expected_mode:
            return None
        record = self.late
        self.late = None
        self.expected_mode = late
        if record is not None:
            # Keep the RC baseline: a pilot's RC AUTO may show before RC_CHANNELS does.
            until = max(record.until_ms, now_ms + RC_RECHECK_MS)
            self.adopted = replace(record, until_ms=until)
        return late

    def _rc_takeover(self, t: FcTelemetry, now_ms: int) -> bool:
        """The RC mode switch moved after a late AUTO was taken as ours while the flight
        controller still flies it: the pilot chose it, a takeover."""
        watch = self.adopted
        if watch is None:
            return False
        if now_ms > watch.until_ms:
            self.adopted = None
            return False
        if t.fc_custom_mode != int(watch.mode) or not switch_moved(
            watch.rc_mode_pwm, t.rc_mode_pwm
        ):
            return False
        self.adopted = None
        return True

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

    def confirmed(self, mode: CopterMode, *, resumed: bool = False) -> None:
        """The telemetry shows ``mode``, which we asked for (``resumed``: a pilot's Resume,
        nothing of ours is on the way). A late AUTO ends once AUTO itself is confirmed;
        otherwise the next HEARTBEAT received after a newer answer ends it (``_late``)."""
        self.expected_mode = mode
        if self.pending_mode == mode:
            self.pending_mode = None
        if resumed or (self.late is not None and self.late.mode == mode):
            self.late = None
        if resumed:
            self.adopted = None

    def reset(self) -> None:
        """On the ground after landing: nothing is expected any more."""
        self.expected_mode = None
        self.pending_mode = None
        self.late = None
        self.adopted = None
        self.change = None

    # --- classification ---------------------------------------------------------------

    def unexpected_mode(self, t: FcTelemetry, now_ms: int) -> ModeChange | None:
        """A fresh HEARTBEAT shows a mode we neither expect nor are changing to."""
        if self.expected_mode is None or t.fc_custom_mode is None:
            return None
        fresh = t.fc_heartbeat_mono_ms is not None and now_ms - t.fc_heartbeat_mono_ms <= _FRESH_MS
        if not fresh:
            return None
        if self._rc_takeover(t, now_ms):
            return ModeChange(mode_name(t.fc_custom_mode), now_ms)
        if t.fc_custom_mode in self._ours(t, self.expected_mode, now_ms):
            return None
        return ModeChange(mode_name(t.fc_custom_mode), now_ms)

    def start_classification(self, change: ModeChange) -> None:
        self.change = change
        self.pending_mode = None
        self.late = None
        self.adopted = None

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
