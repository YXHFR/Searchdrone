"""The MAVLink thread: the ground station's own channel to the flight controller.

``MavThread`` is a daemon thread with its own event loop. It owns the connection
(``sd_mav.Vehicle`` as 250/190), so STOP, FLY HOME, the companion watchdog, the battery
guard and the pilot's sticks keep working while the main loop, the database or the AI
hang. ``GroundMav`` is the part that runs inside that loop; tests drive it directly on
their own loop with a fake flight controller and a fake clock.

* **In:** ``submit(ModeAction)``, ``takeover``, ``handback`` and ``sticks`` are safe to
  call from any thread (``ThreadSafeMav``).
* **Out:** ``on_report`` gets each action's result (``ActionReport``), ``on_event`` the
  alarms and events (``vehicle.onboard_silent``, ``loop_stall``, ...); both are called
  in this thread and must only queue. ``snapshot`` is a thread-safe copy for the
  ingress thread and the main loop.
* **GCS heartbeat** 1 Hz from ``Vehicle``, gated by ``HeartbeatGate``: when the main
  loop or the ingress loop has not beaten for ``LOOP_STALL_S`` no heartbeat goes out,
  so the flight controller's GCS failsafe takes over from a hung ground station. The
  emergency path in this thread is not affected.
* **Sticks** (the pilot's "Ta över"): ``MANUAL_CONTROL`` from 250/190, only in LOITER,
  armed and in the air (the allowlist checks). Without a new value from the panel for
  ``MANUAL_CONTROL_TIMEOUT_MS`` nothing more is sent; the flight controller's
  ``RC_OVERRIDE_TIME`` then ends the override and LOITER holds the position.
"""

from __future__ import annotations

import asyncio
import concurrent.futures
import contextlib
import datetime as dt
import math
import threading
import time
from collections.abc import Awaitable, Callable
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Final, Literal

from sd_ground.liveness import LoopLiveness
from sd_ground.logging import get_logger
from sd_ground.mav import fence as mav_fence
from sd_ground.mav.companion_watchdog import CompanionWatchdog, WatchInput
from sd_ground.mav.gcs import OverrideAction, OverrideResult, guard_override, run_override
from sd_ground.mav.gnss_watch import GnssWatch, alarm, text_alarm
from sd_ground.services.latches import (
    BATTERY_LOW_SV,
    ENERGY_HOME_SV,
    HOME_LATCHES,
    BatteryReason,
    Latches,
)
from sd_ground.services.wind import WIND_PARAMS, FcParams, WindStage, WindWatch, fc_params
from sd_ground.sources import PartState
from sd_mav.chain import mode_shown, rc_switch_moved, set_mode_with_fallback
from sd_mav.failsafe_signals import text_signal
from sd_mav.modes import CopterMode
from sd_mav.vehicle import (
    MODE_FRESH_S,
    ControlNotAllowedError,
    FcTelemetry,
    ForbiddenMavlinkError,
    SendOrigin,
    Vehicle,
)
from sd_obs.clock import Clock
from sd_safety.constants import COMPANION_WATCHDOG_S, FC_HEARTBEAT_TIMEOUT_S
from sd_safety.limits import DEFAULT_LIMITS
from sd_schemas.api import Attitude, Telemetry, WindCheck
from sd_schemas.base import LatLon
from sd_schemas.link import MANUAL_CONTROL_TIMEOUT_MS, StickAxes
from sd_schemas.mission import LatchAction

__all__ = [
    "BATTERY_CONFIRM_S",
    "ENERGY_FRESH_S",
    "TICK_S",
    "WIND_PARAMS_RETRY_S",
    "ActionReport",
    "GroundMav",
    "HeartbeatGate",
    "MavSnapshot",
    "MavThread",
    "ModeAction",
    "ThreadSafeMav",
    "vehicle_events",
]

TICK_S: Final = 0.05
STICK_RESEND_S: Final = 0.1  # 10 Hz while the panel's values are fresh
TAKEOVER_POLL_S: Final = 0.01  # how often a waiting take-over looks for its notice's ack
BATTERY_CONFIRM_S: Final = 0.5  # two BATTERY_STATUS at 2 Hz
# The wind check's ENERGY_HOME (computed at 5 Hz outside this thread) counts this long;
# an older one, from callers that stopped, never starts FLY HOME.
ENERGY_FRESH_S: Final = 2.0
WIND_PARAMS_RETRY_S: Final = 30.0  # the flight controller did not answer: read again
ATTITUDE_FRESH_S: Final = 1.0  # ATTITUDE is streamed at 4 Hz
ATTITUDE_EDGE_DEG: Final = 0.01  # float rounding of +-pi and +-pi/2 radians
_IN_AIR: Final = frozenset({"in_air", "takeoff", "landing"})
_FLYING_HOME: Final = frozenset({CopterMode.RTL, CopterMode.SMART_RTL, CopterMode.LAND})
log = get_logger("sd_ground.mav")

EventSink = Callable[[str, str, dict[str, Any]], None]  # (type, severity, payload)


def chain_of(kind: LatchAction) -> OverrideAction:
    """Which steps a latch kind uses: STOP's or FLY HOME's."""
    return "stop" if kind in ("stop", "watchdog_stop") else "return_home"


@dataclass(frozen=True, slots=True)
class ModeAction:
    action_id: str
    vehicle_id: str
    kind: LatchAction
    action_seq: int


@dataclass(frozen=True, slots=True)
class ActionReport:
    action_id: str
    vehicle_id: str
    kind: LatchAction
    action_seq: int
    result: Literal["acked", "already_active", "failed", "aborted"]
    mode_requested: str | None
    mode_confirmed: str | None
    alarm: str | None


@dataclass(frozen=True, slots=True)
class MavSnapshot:
    """A copy of what the MAVLink thread knows, for the other threads."""

    vehicle_id: str
    connected: bool  # a flight-controller HEARTBEAT within FC_HEARTBEAT_TIMEOUT_S
    telemetry: FcTelemetry
    now_mono_ms: int
    in_flight: bool
    onboard_alive: bool
    airborne_since: dt.datetime | None
    manual: bool
    watchdog_phase: str
    # The take-over the pilot steers with ("Ta över"), when it started (wall clock) and
    # the age of the last stick value; None without a take-over.
    manual_takeover_id: str | None = None
    manual_since: dt.datetime | None = None
    manual_input_age_ms: int | None = None
    # The pilot pressed "Ta över" and LOITER waits for the onboard side's ack of the notice.
    takeover_pending: bool = False

    @property
    def by_hand(self) -> bool:
        """The pilot steers, or has pressed "Ta över" and its LOITER still waits: either
        way the AI may not move the drone. The onboard side may already have ended the
        AI's order in the wait, before its own snapshot shows the take-over."""
        return self.manual or self.takeover_pending

    @property
    def fc_mode(self) -> CopterMode | None:
        mode = self.telemetry.fc_custom_mode
        try:
            return None if mode is None else CopterMode(mode)
        except ValueError:
            return None


def in_flight(t: FcTelemetry) -> bool:
    """The wide definition used by the emergency path (start and landing count)."""
    return t.landed_state in _IN_AIR or (t.armed and t.alt_rel_home_m > 0.5)


_VEHICLE_EVENTS: Final = {
    "mav.forbidden_mavlink_blocked": ("mav.forbidden_mavlink_blocked", "critical"),
    "mav.inbound_command_ignored": ("mav.inbound_command_ignored", "warning"),
    "mav.retry_cancelled": ("mav.retry_cancelled", "warning"),
}


EVENTS_PER_KIND_S: Final = 1.0  # at most one event per kind and second from the link


def vehicle_events(
    sink: EventSink,
    now: Callable[[], float] | None = None,
    utc: Callable[[], dt.datetime] | None = None,
) -> Callable[[str, dict[str, Any]], None]:
    """The ``Vehicle``'s events that the ground station keeps; status texts at warning
    level or worse, and failsafe texts at any level, become ``fc.statustext``. At most one
    per kind and second (the rest are counted in the next one), so a flood on the MAVLink
    channel cannot fill the main loop's queue."""
    clock = now or time.monotonic
    wall = utc or (lambda: dt.datetime.now(dt.UTC))
    last: dict[str, float] = {}
    dropped: dict[str, int] = {}

    def emit(kind: str, severity: str, details: dict[str, Any]) -> None:
        t = clock()
        if t - last.get(kind, -EVENTS_PER_KIND_S) < EVENTS_PER_KIND_S:
            dropped[kind] = dropped.get(kind, 0) + 1
            return
        last[kind] = t
        skipped = dropped.pop(kind, 0)
        sink(kind, severity, {**details, "skipped": skipped} if skipped else details)

    def on_event(kind: str, details: dict[str, Any]) -> None:
        if kind == "mav.statustext":
            text = str(details.get("text", ""))
            gps = text_alarm(text, wall().isoformat())
            if gps is not None:
                emit(*gps)
            # Warnings and worse, and every failsafe text (ArduCopter sends a fence breach
            # as a notice).
            failsafe = text_signal(text) is not None
            if int(details.get("severity", 7)) <= 4 or failsafe:
                emit("fc.statustext", "warning", details)
            return
        mapped = _VEHICLE_EVENTS.get(kind)
        if mapped is not None:
            emit(mapped[0], mapped[1], details)

    return on_event


class HeartbeatGate:
    """Our GCS heartbeat goes out only while both event loops are alive."""

    def __init__(self, liveness: LoopLiveness, on_event: EventSink | None = None) -> None:
        self._liveness = liveness
        self._on_event = on_event
        self.stalled = False

    def __call__(self) -> bool:
        stalled = self._liveness.stalled()
        if stalled and not self.stalled:
            log.critical("loop_stall", loops=stalled)
            if self._on_event is not None:
                self._on_event("system.loop_stall", "critical", {"loops": stalled})
        elif not stalled and self.stalled and self._on_event is not None:
            self._on_event("system.loop_recovered", "warning", {})
        self.stalled = bool(stalled)
        return not stalled


@dataclass
class _PendingTakeover:
    """A take-over whose LOITER waits for the onboard side to know about it."""

    takeover_id: str
    action_seq: int
    rc_at_click: int | None
    acknowledged: bool = False


@dataclass
class _Manual:
    takeover_id: str
    since: dt.datetime
    axes: StickAxes | None = None
    received_mono: float = 0.0
    sent_mono: float = 0.0
    sent_seq: int = -1


@dataclass
class _Tracks:
    tasks: set[asyncio.Task[Any]] = field(default_factory=set[asyncio.Task[Any]])


class GroundMav:
    """The emergency path and the guards, inside the MAVLink thread's event loop."""

    def __init__(
        self,
        vehicle: Vehicle,
        clock: Clock,
        *,
        vehicle_id: str,
        latches: Latches,
        on_report: Callable[[ActionReport], None],
        on_event: EventSink,
        system_action: Callable[[str, LatchAction], object],
        rtl_pct: Callable[[], float] = lambda: DEFAULT_LIMITS.battery_rtl_pct,
    ) -> None:
        self.vehicle = vehicle
        self.clock = clock
        self.vehicle_id = vehicle_id
        self.latches = latches
        self._on_report = on_report
        # Every event names the vehicle, so the event log and the panel can tell them apart.
        self._on_event: EventSink = lambda t, sev, p: on_event(
            t, sev, {"vehicle_id": vehicle_id, **p}
        )
        self._system_action = system_action
        self._rtl_pct = rtl_pct
        self.watchdog = CompanionWatchdog(clock.monotonic_ms())
        self.gnss = GnssWatch(self._on_event)
        self._lock = threading.Lock()
        self._snap: MavSnapshot | None = None
        self._tracks = _Tracks()
        self._tick_task: asyncio.Task[None] | None = None
        self._manual: _Manual | None = None
        self._pending: _PendingTakeover | None = None
        self._airborne_since: dt.datetime | None = None
        self._flown = False
        self._battery_low_since: float | None = None
        self._energy_short_at: float | None = None  # monotonic s of the last ENERGY_HOME
        self._was_connected = False
        self.sticks_sent = 0
        self._closed = False

    # --- lifecycle --------------------------------------------------------------------

    async def start(self) -> None:
        await self.vehicle.start()
        self._publish()
        self._tick_task = asyncio.ensure_future(self._tick_loop())

    async def close(self) -> None:
        if self._closed:
            return
        self._closed = True
        tasks = [*self._tracks.tasks, *([self._tick_task] if self._tick_task else [])]
        for task in tasks:
            task.cancel()
        await asyncio.gather(*tasks, return_exceptions=True)
        await self.vehicle.close()

    def _spawn(self, coro: Any) -> None:
        task = asyncio.ensure_future(coro)
        self._tracks.tasks.add(task)
        task.add_done_callback(self._done)

    def _done(self, task: asyncio.Task[Any]) -> None:
        self._tracks.tasks.discard(task)
        if not task.cancelled() and task.exception() is not None:
            log.error("mav.task_failed", error=type(task.exception()).__name__)

    # --- the picture ------------------------------------------------------------------

    def snapshot(self) -> MavSnapshot | None:
        with self._lock:
            return self._snap

    def _publish(self) -> None:
        t = self.vehicle.telemetry
        now = self.clock.monotonic_ms()
        connected = (
            t.fc_heartbeat_mono_ms is not None
            and now - t.fc_heartbeat_mono_ms <= FC_HEARTBEAT_TIMEOUT_S * 1000
        )
        flying = connected and in_flight(t)
        if flying and self._airborne_since is None:
            self._airborne_since = self.clock.utc_now()
        elif not flying and connected:
            self._airborne_since = None
        onboard = (
            t.onboard_seen_mono_ms is not None
            and now - t.onboard_seen_mono_ms <= COMPANION_WATCHDOG_S * 1000
        )
        m = self._manual
        snap = MavSnapshot(
            vehicle_id=self.vehicle_id,
            connected=connected,
            telemetry=t,
            now_mono_ms=now,
            in_flight=flying,
            onboard_alive=onboard,
            airborne_since=self._airborne_since,
            manual=m is not None,
            watchdog_phase=self.watchdog.phase,
            manual_takeover_id=None if m is None else m.takeover_id,
            manual_since=None if m is None else m.since,
            manual_input_age_ms=(
                None
                if m is None or m.axes is None
                else int((self.clock.monotonic() - m.received_mono) * 1000)
            ),
            takeover_pending=self._pending is not None,
        )
        with self._lock:
            self._snap = snap

    # --- actions ----------------------------------------------------------------------

    def submit(self, action: ModeAction) -> None:
        """Start an action's steps now (loop thread only)."""
        if action.vehicle_id != self.vehicle_id:
            return
        self._manual = None  # STOP and FLY HOME end steering by hand
        self._pending = None  # and a take-over that still waits never starts
        # The 10 s watch counts from here, on this thread's own clock.
        # So does the RC mode switch's position: a person moving it afterwards wins.
        rc_at_click = self.vehicle.telemetry.rc_mode_pwm
        self._spawn(self._run_action(action, self.clock.monotonic(), rc_at_click))

    def _superseded(self, action: ModeAction) -> bool:
        """A newer STOP, FLY HOME, system action or Resume stops this chain (a takeover
        or hand-back does not)."""
        return self.latches.override_seq(self.vehicle_id) != action.action_seq

    async def _run_action(
        self, action: ModeAction, clicked_mono: float, rc_at_click: int | None
    ) -> None:
        which = chain_of(action.kind)
        if not self.vehicle.telemetry.fc_connected:
            # Still try: the link may be back any moment. The result says what happened.
            log.warning("mav.override_without_fc", kind=action.kind)

        def stop_sending() -> bool:  # a newer action, or a person flipped the RC switch
            return self._superseded(action) or rc_switch_moved(self.vehicle, rc_at_click)

        result = await run_override(self.vehicle, which, aborted=stop_sending)
        self._report(action, result)
        if result.result == "acked" and result.mode_confirmed not in (None, "DISARMED"):
            confirmed = CopterMode[result.mode_confirmed]
            await guard_override(
                self.vehicle,
                which,
                confirmed,
                clicked_mono=clicked_mono,
                rc_at_click=rc_at_click,
                aborted=lambda: self._superseded(action),
                on_resend=lambda mode: self._on_event(
                    "vehicle.override_reasserted",
                    "warning",
                    {"kind": action.kind, "seen_mode": mode, "action_seq": action.action_seq},
                ),
            )

    def _report(self, action: ModeAction, result: OverrideResult) -> None:
        if result.alarm is not None:
            self._on_event(
                f"alarm.{result.alarm}",
                "critical",
                {"kind": action.kind, "action_seq": action.action_seq},
            )
        self._on_report(
            ActionReport(
                action.action_id,
                action.vehicle_id,
                action.kind,
                action.action_seq,
                result.result,
                result.mode_requested,
                result.mode_confirmed,
                result.alarm,
            )
        )

    # --- steering by hand -------------------------------------------------------------

    def takeover(self, takeover_id: str, action_seq: int, wait_s: float = 0.0) -> None:
        """Steer by hand from here: LOITER, then the panel's sticks.

        With ``wait_s`` (SD-link up), LOITER waits until the onboard side has acknowledged
        the take-over notice (``takeover_acknowledged``) or ``wait_s`` has passed, counted
        here on this thread's own clock: a hung main loop or SD-link cannot hold it up
        longer. A newer action in the meantime (STOP, FLY HOME, a system action, a
        hand-back, another take-over) or a move of the RC mode switch drops it."""
        pending = _PendingTakeover(takeover_id, action_seq, self.vehicle.telemetry.rc_mode_pwm)
        self._pending = pending
        self._publish()  # the picture shows it now, not at the next tick
        if wait_s <= 0:
            self._start_takeover(pending)
            return

        async def after_the_notice() -> None:
            deadline = self.clock.monotonic() + wait_s
            while not pending.acknowledged and self.clock.monotonic() < deadline:
                await self.clock.sleep(TAKEOVER_POLL_S)
            self._start_takeover(pending)

        self._spawn(after_the_notice())

    def takeover_acknowledged(self, takeover_id: str) -> None:
        """The onboard side has the take-over notice: LOITER may go now."""
        pending = self._pending
        if pending is not None and pending.takeover_id == takeover_id:
            pending.acknowledged = True

    def _start_takeover(self, pending: _PendingTakeover) -> None:
        if self._pending is not pending:
            return  # STOP, FLY HOME, a hand-back or a newer take-over came first
        self._pending = None
        seq = pending.action_seq
        if self.latches.latest_seq(self.vehicle_id) != seq:
            # A newer action that never comes here (Resume) took a number while it waited:
            # say so, the pilot pressed Ta över and no LOITER follows. STOP, FLY HOME and
            # system actions clear the take-over in submit(), with events of their own.
            self._on_event(
                "vehicle.takeover_dropped",
                "warning",
                {"takeover_id": pending.takeover_id, "reason": "newer_action"},
            )
            return
        if rc_switch_moved(self.vehicle, pending.rc_at_click):
            # A person at the RC transmitter chose a mode meanwhile: it goes first.
            self._on_event(
                "alarm.TAKEOVER_FAILED",
                "critical",
                {"takeover_id": pending.takeover_id, "reason": "rc_switch"},
            )
            return
        self._manual = _Manual(pending.takeover_id, self.clock.utc_now())
        takeover_id = pending.takeover_id

        async def loiter() -> None:
            result = await set_mode_with_fallback(
                self.vehicle,
                [CopterMode.LOITER],
                origin=SendOrigin.GROUND,
                aborted=lambda: self.latches.latest_seq(self.vehicle_id) != seq,
            )
            if result.result == "failed":  # the pilot must see the takeover did not work
                self._on_event("alarm.TAKEOVER_FAILED", "critical", {"takeover_id": takeover_id})

        self._spawn(loiter())

    def handback(self) -> None:
        self._manual = None
        self._pending = None

    def sticks(self, axes: StickAxes) -> None:
        m = self._manual
        if m is None or str(axes.takeover_id) != m.takeover_id:
            return
        if m.axes is not None and axes.input_seq <= m.axes.input_seq:
            return  # older or repeated values are dropped
        m.axes = axes
        m.received_mono = self.clock.monotonic()

    async def _send_sticks(self) -> None:
        m = self._manual
        if m is None or m.axes is None:
            return
        now = self.clock.monotonic()
        if now - m.received_mono > MANUAL_CONTROL_TIMEOUT_MS / 1000:
            return  # the 1 s watchdog: no fresh value from the panel, nothing is sent
        fresh_value = m.axes.input_seq != m.sent_seq
        if not fresh_value and now - m.sent_mono < STICK_RESEND_S:
            return
        if mode_shown(self.vehicle) != CopterMode.LOITER:
            return
        m.sent_mono, m.sent_seq = now, m.axes.input_seq
        with contextlib.suppress(ForbiddenMavlinkError, ControlNotAllowedError):
            await self.vehicle.manual_control(m.axes, origin=SendOrigin.MANUAL)
            self.sticks_sent += 1

    # --- the guards -------------------------------------------------------------------

    async def _tick_loop(self) -> None:
        while True:
            try:
                await self._tick()
            except Exception as exc:  # the guards must keep running
                log.error("mav.tick_failed", error=type(exc).__name__)
            await self.clock.sleep(TICK_S)

    async def _tick(self) -> None:
        self._publish()
        snap = self.snapshot()
        if snap is None:
            return
        if snap.connected != self._was_connected:
            self._was_connected = snap.connected
            self._on_event(
                "vehicle.mavlink_up" if snap.connected else "alarm.MAVLINK_LOST",
                "info" if snap.connected else "critical",
                {},
            )
        # Landed: it has flown since the last landing and now stands disarmed.
        if snap.connected and snap.in_flight:
            self._flown = True
        if snap.connected and self._flown and not snap.in_flight and not snap.telemetry.armed:
            self._flown = False
            self._landed()
        self._watchdog(snap)
        self._battery(snap)
        if snap.connected:
            self.gnss.tick(snap.telemetry, snap.now_mono_ms, in_flight=snap.in_flight)
        await self._send_sticks()

    def _landed(self) -> None:
        self._manual = None
        if self._pending is not None:  # a take-over still waiting: no LOITER on the ground
            self._on_event(
                "vehicle.takeover_dropped",
                "warning",
                {"takeover_id": self._pending.takeover_id, "reason": "landed"},
            )
            self._pending = None
        self._battery_low_since = None
        self.gnss.reset()
        latch = self.latches.landed(self.vehicle_id)
        self._on_event(
            "vehicle.landed", "info", {"latch": latch.action if latch is not None else None}
        )

    def _watchdog(self, snap: MavSnapshot) -> None:
        latch = self.latches.get(self.vehicle_id)
        was_alarm = self.watchdog.alarm_active
        fresh_mode = mode_shown(self.vehicle)
        action = self.watchdog.tick(
            WatchInput(
                now_ms=snap.now_mono_ms,
                onboard_seen_ms=snap.telemetry.onboard_seen_mono_ms,
                fc_mode=fresh_mode if snap.connected else None,
                in_flight=snap.in_flight,
                operator_override=latch is not None and latch.action in ("stop", "return_home"),
                manual=snap.manual,
                battery_latch=latch is not None
                and (latch.action == "battery_rtl" or latch.underlying == "battery_rtl"),
            )
        )
        if self.watchdog.alarm_active and not was_alarm:
            self._on_event("alarm.ONBOARD_SILENT", "critical", {"phase": self.watchdog.phase})
        if action == "watchdog_stop":
            self._on_event("vehicle.onboard_silent", "warning", {})
            self._system_action(self.vehicle_id, action)
        elif action == "watchdog_rtl":
            self._on_event("vehicle.onboard_silent_rtl", "critical", {})
            self._system_action(self.vehicle_id, action)

    def energy_short(self, short: bool) -> None:
        """The wind check's ``ENERGY_HOME`` (``sd_ground.services.wind``): the battery
        will not last the way home against the wind. Called in this loop through
        ``ThreadSafeMav``; it only adds a reason to fly home, never removes one."""
        self._energy_short_at = self.clock.monotonic() if short else None

    def _battery(self, snap: MavSnapshot) -> None:
        t = snap.telemetry
        latch = self.latches.get(self.vehicle_id)
        mode = snap.fc_mode
        at_level = t.battery_pct is not None and t.battery_pct <= self._rtl_pct()
        # Never while the pilot steers by hand: only the percentage level overrides that.
        energy = (
            self._energy_short_at is not None
            and self.clock.monotonic() - self._energy_short_at <= ENERGY_FRESH_S
            and not snap.by_hand
        )
        # A battery latch under a STOP blocks its own trigger again until Resume. One set by
        # ENERGY_HOME never blocks the percentage level: STOP and a take-over after an
        # energy flight home must not cost the flight home at the level.
        under = latch is not None and latch.underlying == "battery_rtl"
        if under and self.latches.battery_reason(self.vehicle_id) == "energy_home":
            energy = False
        elif under:
            at_level = energy = False
        low = at_level or energy
        blocked = (
            not snap.connected
            or not snap.in_flight
            or mode in _FLYING_HOME
            or (latch is not None and latch.action in ("battery_rtl", *HOME_LATCHES))
        )
        if not low or blocked:
            self._battery_low_since = None if not low else self._battery_low_since
            return
        now = self.clock.monotonic()
        if self._battery_low_since is None:
            self._battery_low_since = now
        if now - self._battery_low_since < BATTERY_CONFIRM_S:
            return
        self._battery_low_since = None
        reason: BatteryReason = "battery_level" if at_level else "energy_home"
        self._on_event(
            "vehicle.battery_rtl",
            "critical",
            {"battery_pct": t.battery_pct, "alarm": "BATTERY_RTL", "reason": reason},
        )
        self._on_event(
            *alarm(
                "BATTERY_RTL",
                "critical",
                ENERGY_HOME_SV if reason == "energy_home" else BATTERY_LOW_SV,
                battery_pct=t.battery_pct,
                reason=reason,
            )
        )
        # The reason belongs to the latch: only once the flight home was started (None:
        # no path to the drone, no latch).
        if self._system_action(self.vehicle_id, "battery_rtl") is not None:
            self.latches.set_battery_reason(self.vehicle_id, reason)


async def _run[T](work: Callable[[Vehicle], Awaitable[T]], vehicle: Vehicle) -> T:
    return await work(vehicle)


class ThreadSafeMav:
    """Calls into a ``GroundMav`` from any thread; the work runs in its own loop."""

    def __init__(
        self, core: GroundMav, loop: asyncio.AbstractEventLoop, wind: WindWatch | None = None
    ) -> None:
        self.core = core
        self.loop = loop
        # The wind warning is computed by the callers of ``telemetry`` (never in the
        # MAVLink thread, so it can never slow the emergency path).
        self.wind = WindWatch() if wind is None else wind
        self._wind_lock = threading.Lock()
        # The flight controller's own lean limit, RTL speed and battery numbers for the
        # wind check, read once per connection with ``mav.fence.read_params`` (missing ones
        # again until answered). The read runs in the MAVLink loop; its result is taken
        # here, by the callers.
        self._params_read: concurrent.futures.Future[dict[str, float | None]] | None = None
        self._params_done = False
        self._params_retry_at: float = -math.inf
        self._wind_check: WindCheck | None = None  # the newest from ``update_wind``
        # The flight controller connection the wind check's numbers belong to:
        # (connected, ``fc_link_epoch``). A new connection or a reboot starts over.
        self._wind_link: tuple[bool, int] = (False, 0)
        self._params_values: dict[str, float | None] = {}

    def _call(self, fn: Callable[..., None], *args: Any) -> bool:
        try:
            self.loop.call_soon_threadsafe(fn, *args)
        except RuntimeError:  # the loop is closed
            return False
        return True

    def submit(self, action: ModeAction) -> bool:
        return self._call(self.core.submit, action)

    def takeover(self, takeover_id: str, action_seq: int, wait_s: float = 0.0) -> bool:
        return self._call(self.core.takeover, takeover_id, action_seq, wait_s)

    def takeover_acknowledged(self, takeover_id: str) -> bool:
        return self._call(self.core.takeover_acknowledged, takeover_id)

    def handback(self) -> bool:
        return self._call(self.core.handback)

    def sticks(self, axes: StickAxes) -> bool:
        return self._call(self.core.sticks, axes)

    def snapshot(self) -> MavSnapshot | None:
        return self.core.snapshot()

    def start_tlog(self, path: Path) -> bool:
        """``flight.tlog``: the connection records under its own lock, from any thread."""
        return self.core.vehicle.start_tlog(path)

    def stop_tlog(self) -> None:
        self.core.vehicle.stop_tlog()

    async def call[T](self, work: Callable[[Vehicle], Awaitable[T]], timeout_s: float = 30.0) -> T:
        """Run ``work(vehicle)`` in the MAVLink thread's loop and wait for it from another
        loop (the fence upload and parameter reads; never the emergency path)."""
        future = asyncio.run_coroutine_threadsafe(_run(work, self.core.vehicle), self.loop)
        return await asyncio.wait_for(asyncio.wrap_future(future), timeout_s)

    # --- sd_ground.sources.VehicleLink ------------------------------------------------

    def state(self) -> PartState:
        snap = self.snapshot()
        return "ok" if snap is not None and snap.connected else "down"

    def connected(self, vehicle_id: str) -> bool:
        snap = self.snapshot()
        return snap is not None and snap.vehicle_id == vehicle_id and snap.connected

    def in_flight(self, vehicle_id: str) -> bool:
        snap = self.snapshot()
        return snap is not None and snap.vehicle_id == vehicle_id and snap.in_flight

    def airborne_since(self) -> dt.datetime | None:
        snap = self.snapshot()
        return None if snap is None else snap.airborne_since

    def onboard_state(self) -> PartState:
        snap = self.snapshot()
        return "ok" if snap is not None and snap.onboard_alive else "down"

    def telemetry(self, vehicle_id: str) -> Telemetry | None:
        """The newest telemetry with the newest wind check (``update_wind``). Reads only."""
        snap = self.snapshot()
        if snap is None or snap.vehicle_id != vehicle_id or not snap.connected:
            return None
        with self._wind_lock:
            wind = self._wind_check
        return to_telemetry(snap, self.core.clock.utc_now(), wind)

    def update_wind(self, vehicle_id: str) -> None:
        """The wind check's step, from the telemetry loop (never the MAVLink thread): take
        the newest samples, read the flight controller's numbers once per connection, and
        hand the energy check's FLY HOME reason to the battery guard."""
        snap = self.snapshot()
        if snap is None or snap.vehicle_id != vehicle_id:
            return
        link = (snap.connected, snap.telemetry.fc_link_epoch)
        with self._wind_lock:
            if link != self._wind_link:
                # A new connection (possibly another flight controller), a reboot and a
                # lost connection all start from nothing: the numbers are read again, and
                # the drag is measured again.
                self._forget_wind()
                self._wind_link = link
            if not snap.connected:
                return
            self._wind_params()
            self.wind.observe(snap.telemetry, snap.now_mono_ms)
            wind = self.wind.check(snap.telemetry, snap.now_mono_ms)
            self._wind_check = wind
            rtl = self.wind.energy_rtl
        self._call(self.core.energy_short, rtl)

    def _forget_wind(self) -> None:
        """Called under ``_wind_lock``."""
        if self._params_read is not None:
            self._params_read.cancel()
        self._params_read = None
        self._params_done = False
        self._params_retry_at = -math.inf
        self._params_values = {}
        self._wind_check = None
        self.wind.reset(FcParams())

    def _wind_params(self) -> None:
        """Take a finished read of ``WIND_PARAMS``, or start one for those not yet
        answered on this connection (all of them at first; after a partial answer only the
        missing ones, ``WIND_PARAMS_RETRY_S`` later). Until a number is read, the wind
        check uses its constant. Called under ``_wind_lock``."""
        future = self._params_read
        if future is not None:
            if not future.done():
                return
            self._params_read = None
            values: dict[str, float | None] = {}
            if not future.cancelled() and future.exception() is None:
                values = future.result()
            else:
                error = "cancelled" if future.cancelled() else type(future.exception()).__name__
                log.warning("wind.params_failed", error=error)
            answered = {k: v for k, v in values.items() if v is not None}
            self._params_values |= answered
            if self._missing_params():
                self._params_retry_at = self.core.clock.monotonic() + WIND_PARAMS_RETRY_S
            else:
                self._params_done = True
            if answered or not self._params_values:
                # logs every constant it falls back to
                self.wind.set_params(fc_params(self._params_values))
            return
        if self._params_done or self.core.clock.monotonic() < self._params_retry_at:
            return
        names = self._missing_params()

        async def read(vehicle: Vehicle) -> dict[str, float | None]:
            return await mav_fence.read_params(vehicle, names)

        try:
            self._params_read = asyncio.run_coroutine_threadsafe(
                _run(read, self.core.vehicle), self.loop
            )
        except RuntimeError:  # the loop is closed
            return

    def _missing_params(self) -> tuple[str, ...]:
        """``WIND_PARAMS`` the flight controller has not answered on this connection;
        ``WP_SPD`` only matters when ``RTL_SPEED_MS`` is 0."""
        got = self._params_values
        return tuple(
            name
            for name in WIND_PARAMS
            if name not in got and (name != "WP_SPD" or got.get("RTL_SPEED_MS", 0.0) == 0.0)
        )


def to_telemetry(snap: MavSnapshot, now: dt.datetime, wind: WindCheck | None = None) -> Telemetry:
    t = snap.telemetry
    fresh = (
        t.fc_heartbeat_mono_ms is not None
        and snap.now_mono_ms - t.fc_heartbeat_mono_ms <= MODE_FRESH_S * 1000
    )
    return Telemetry(
        t=now,
        time_boot_ms=t.position_mono_ms or 0,
        lat=t.lat,
        lon=t.lon,
        alt_amsl_m=t.alt_amsl_m,
        alt_rel_home_m=t.alt_rel_home_m,
        heading_deg=t.heading_deg,
        groundspeed_ms=t.groundspeed_ms,
        climb_ms=t.climb_ms,
        battery_pct=t.battery_pct,
        battery_v=t.battery_voltage_v,
        flight_mode=t.fc_mode if fresh else "UNKNOWN",
        armed=t.armed,
        landed_state=t.landed_state,
        gps_fix=t.gps_fix_type,
        gps_sats=t.gps_satellites,
        home=None if t.home is None else LatLon(lat=t.home.lat, lon=t.home.lon),
        dist_to_pilot_m=None,
        attitude=_attitude(t, snap.now_mono_ms),
        wind=wind,
    )


def _attitude(t: FcTelemetry, now_mono_ms: int) -> Attitude | None:
    """The latest ATTITUDE, or None when it is older than ``ATTITUDE_FRESH_S`` or not a
    possible attitude. Only rounding at the range edges is clamped; anything further out is
    dropped rather than shown as a horizon the drone never had."""
    if t.attitude_mono_ms is None or now_mono_ms - t.attitude_mono_ms > ATTITUDE_FRESH_S * 1000:
        return None
    # sd_mav already drops non-finite samples; the schema rejects them, so check here too.
    if not all(map(math.isfinite, (t.roll_deg, t.pitch_deg, t.yaw_deg))):
        return None
    if abs(t.roll_deg) > 180.0 + ATTITUDE_EDGE_DEG or abs(t.pitch_deg) > 90.0 + ATTITUDE_EDGE_DEG:
        return None
    yaw = t.yaw_deg % 360.0
    return Attitude(
        roll_deg=max(-180.0, min(180.0, t.roll_deg)),
        pitch_deg=max(-90.0, min(90.0, t.pitch_deg)),
        yaw_deg=0.0 if yaw >= 360.0 else yaw,
    )


class MavThread(threading.Thread):
    """Runs a ``GroundMav`` in its own thread and event loop."""

    def __init__(
        self, make_core: Callable[[], GroundMav], wind_stage: WindStage = "first_flights"
    ) -> None:
        super().__init__(name="sd-ground-mavlink", daemon=True)
        self._make_core = make_core
        self._wind_stage: WindStage = wind_stage
        self.ready = threading.Event()
        self.mav: ThreadSafeMav | None = None
        self._stop_event: asyncio.Event | None = None
        self.error: BaseException | None = None

    def run(self) -> None:
        try:
            asyncio.run(self._main())
        except BaseException as exc:  # reported to the starter; never silent
            self.error = exc
            self.ready.set()

    async def _main(self) -> None:
        loop = asyncio.get_running_loop()
        self._stop_event = asyncio.Event()
        core = self._make_core()
        await core.start()
        self.mav = ThreadSafeMav(core, loop, WindWatch(self._wind_stage))
        self.ready.set()
        try:
            await self._stop_event.wait()
        finally:
            await core.close()

    def stop(self, timeout: float = 5.0) -> None:
        mav, ev = self.mav, self._stop_event
        if mav is not None and ev is not None:
            with contextlib.suppress(RuntimeError):
                mav.loop.call_soon_threadsafe(ev.set)
        self.join(timeout)
