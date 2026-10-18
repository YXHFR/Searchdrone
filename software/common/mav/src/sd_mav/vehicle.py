"""``Vehicle``: the flight controller as seen through one MAVLink channel.

It keeps the telemetry (``FcTelemetry``), sends our 1 Hz HEARTBEAT, asks for the
telemetry streams after the first HEARTBEAT from the flight controller and after every
reconnection, notices a lost connection after ``FC_HEARTBEAT_TIMEOUT_S`` and waits for
command acknowledgements.

``Vehicle.send`` is the only way out. Every message is checked against the deny list
and the allowlist first, under every helper method, so no code path can go around them.
A refused message raises :class:`ForbiddenMavlinkError` and is reported as an event.
"""

from __future__ import annotations

import asyncio
import contextlib
import math
from collections.abc import Callable
from dataclasses import dataclass, replace
from enum import IntEnum, StrEnum
from pathlib import Path
from typing import Any, Final

from pymavlink.dialects.v20 import ardupilotmega as mavlink

from sd_mav.allowlist import (
    FC_COMPID,
    POSITION_TYPE_MASK,
    Profile,
    SendContext,
    SendEnvelope,
    check,
)
from sd_mav.connection import (
    GCS_COMPID,
    GCS_SYSID,
    ONBOARD_COMPID,
    MavTransport,
    Role,
    check_identity,
)
from sd_mav.denylist import denied_reason
from sd_mav.failsafe_signals import FailsafeSignal, signal_from
from sd_mav.modes import GROUND_DISARM_MAX_ALT_M, CopterMode, mode_name
from sd_obs.clock import Clock
from sd_safety.constants import (
    FC_HEARTBEAT_TIMEOUT_S,
    HEARTBEAT_INTERVAL_US,
    MAV_ACK_TIMEOUT_S,
    OVERRIDE_NOTICE_WINDOW_S,
)
from sd_schemas.base import LatLon, LatLonAlt
from sd_schemas.link import StickAxes
from sd_schemas.vehicle import LandedState, VehicleState

__all__ = [
    "ACK_ATTEMPTS",
    "MODE_FRESH_S",
    "RC_LINK_FRESH_S",
    "RC_MODE_CHANNEL",
    "ControlNotAllowedError",
    "EventSink",
    "FcTelemetry",
    "ForbiddenMavlinkError",
    "MavResult",
    "SendOrigin",
    "Vehicle",
    "rc_link_ok",
]

ACK_ATTEMPTS: Final = 3
RC_MODE_CHANNEL: Final = 5  # ArduCopter's FLTMODE_CH default: the RC flight-mode switch
# RC_CHANNELS comes at 4 Hz; older than this, the RC transmitter's link counts as down.
RC_LINK_FRESH_S: Final = 1.5
_RC_MIN_CHANNELS: Final = 4  # roll, pitch, throttle, yaw
_RC_THROTTLE_CHANNEL: Final = 3
MODE_FRESH_S: Final = 1.5  # a HEARTBEAT older than this does not tell the current mode
_OUR_HEARTBEAT_S: Final = 1.0
_WATCH_S: Final = 0.05
_ARMED_FLAG: Final = mavlink.MAV_MODE_FLAG_SAFETY_ARMED
# A failsafe signal not seen for this long counts as new when it comes back. Same
# window as the executor's classification of a mode change (a failsafe signal up to
# this long around the change counts), so both agree on what "new" means.
_SIGNAL_QUIET_MS: Final = int(OVERRIDE_NOTICE_WINDOW_S * 1000)
# Landed states that count as "in the air" (the flight controller's own words).
_AIRBORNE_STATES: Final = frozenset({"in_air", "takeoff", "landing"})
# Commands that move the vehicle; their retries stop when the flight controller decided.
_MOVING_COMMANDS: Final = frozenset(
    {
        mavlink.MAV_CMD_DO_SET_MODE,
        mavlink.MAV_CMD_COMPONENT_ARM_DISARM,
        mavlink.MAV_CMD_NAV_TAKEOFF,
        mavlink.MAV_CMD_DO_CHANGE_SPEED,
        mavlink.MAV_CMD_DO_SET_MISSION_CURRENT,
    }
)
_EKF_NEEDED: Final = (
    mavlink.EKF_ATTITUDE
    | mavlink.EKF_VELOCITY_HORIZ
    | mavlink.EKF_POS_HORIZ_ABS
    | mavlink.EKF_POS_VERT_ABS
)
_LANDED: Final[dict[int, LandedState]] = {
    mavlink.MAV_LANDED_STATE_ON_GROUND: "on_ground",
    mavlink.MAV_LANDED_STATE_IN_AIR: "in_air",
    mavlink.MAV_LANDED_STATE_TAKEOFF: "takeoff",
    mavlink.MAV_LANDED_STATE_LANDING: "landing",
}

# Telemetry streams and their intervals in microseconds.
ONBOARD_STREAM_US: Final = {
    mavlink.MAVLINK_MSG_ID_HEARTBEAT: HEARTBEAT_INTERVAL_US,
    mavlink.MAVLINK_MSG_ID_GLOBAL_POSITION_INT: 100_000,
    mavlink.MAVLINK_MSG_ID_ATTITUDE: 250_000,
    mavlink.MAVLINK_MSG_ID_VFR_HUD: 250_000,
    mavlink.MAVLINK_MSG_ID_SYS_STATUS: 500_000,
    mavlink.MAVLINK_MSG_ID_BATTERY_STATUS: 500_000,
    mavlink.MAVLINK_MSG_ID_GPS_RAW_INT: 500_000,
    mavlink.MAVLINK_MSG_ID_EKF_STATUS_REPORT: 500_000,
    mavlink.MAVLINK_MSG_ID_EXTENDED_SYS_STATE: 500_000,
    mavlink.MAVLINK_MSG_ID_MISSION_CURRENT: 500_000,
    mavlink.MAVLINK_MSG_ID_HOME_POSITION: 1_000_000,  # 1 Hz is the slowest allowed
    mavlink.MAVLINK_MSG_ID_RC_CHANNELS: 250_000,
}
GROUND_STREAM_US: Final = {
    mavlink.MAVLINK_MSG_ID_HEARTBEAT: HEARTBEAT_INTERVAL_US,
    mavlink.MAVLINK_MSG_ID_GLOBAL_POSITION_INT: 200_000,
    mavlink.MAVLINK_MSG_ID_ATTITUDE: 250_000,
    mavlink.MAVLINK_MSG_ID_VFR_HUD: 250_000,
    mavlink.MAVLINK_MSG_ID_SYS_STATUS: 500_000,
    mavlink.MAVLINK_MSG_ID_BATTERY_STATUS: 500_000,
    mavlink.MAVLINK_MSG_ID_GPS_RAW_INT: 500_000,
    mavlink.MAVLINK_MSG_ID_EXTENDED_SYS_STATE: 500_000,
    mavlink.MAVLINK_MSG_ID_HOME_POSITION: 1_000_000,
    mavlink.MAVLINK_MSG_ID_RC_CHANNELS: 250_000,
}


class SendOrigin(StrEnum):
    """Who inside a process sends. Required on every ``send``; there is no default."""

    SKILL = "skill"  # a skill handler via ExecutorIO
    LIMIT = "limit"  # a limit monitor (issuer = rule:onboard)
    STARTUP = "startup"  # stream requests, REQUEST_MESSAGE, PARAM_REQUEST_READ, HEARTBEAT
    OVERRIDE_FALLBACK = "override_fallback"  # only executor/override.py
    MANUAL = "manual"  # the pilot's sticks (Ta över): MANUAL_CONTROL from the ground, LOITER only
    GROUND = "ground"  # everything the ground station sends


class MavResult(IntEnum):
    ACCEPTED = mavlink.MAV_RESULT_ACCEPTED
    TEMPORARILY_REJECTED = mavlink.MAV_RESULT_TEMPORARILY_REJECTED
    DENIED = mavlink.MAV_RESULT_DENIED
    UNSUPPORTED = mavlink.MAV_RESULT_UNSUPPORTED
    FAILED = mavlink.MAV_RESULT_FAILED
    IN_PROGRESS = mavlink.MAV_RESULT_IN_PROGRESS
    CANCELLED = 6  # MAV_RESULT_CANCELLED
    TIMEOUT = -1  # no COMMAND_ACK after every attempt (ours, not MAVLink's)

    @classmethod
    def of(cls, value: int) -> MavResult:
        try:
            return cls(value)
        except ValueError:
            return cls.FAILED


class ForbiddenMavlinkError(Exception):
    """A message that the allowlist or the deny list refuses. Nothing was sent."""

    def __init__(self, msg_type: str, reason: str) -> None:
        super().__init__(f"{msg_type}: {reason}")
        self.msg_type = msg_type
        self.reason = reason


class ControlNotAllowedError(Exception):
    """The owner of the link (the onboard executor) does not steer right now. Nothing
    was sent; this is not a forbidden message, so no event is reported."""

    def __init__(self, reason: str) -> None:
        super().__init__(reason)
        self.reason = reason


EventSink = Callable[[str, dict[str, Any]], None]
# Asked before every single write, retries included: a reason string closes the gate.
SendGate = Callable[[SendOrigin], str | None]


@dataclass(frozen=True, slots=True)
class FcTelemetry:
    """The latest values from the flight controller. Times are monotonic milliseconds."""

    fc_connected: bool = False
    fc_heartbeat_mono_ms: int | None = None
    gcs_mavlink_seen_mono_ms: int | None = None  # HEARTBEAT from the ground station
    onboard_seen_mono_ms: int | None = None  # HEARTBEAT from the onboard computer
    armed: bool = False
    fc_custom_mode: int | None = None
    fc_system_status: int = mavlink.MAV_STATE_UNINIT
    landed_state: LandedState = "unknown"
    landed_mono_ms: int | None = None
    position_mono_ms: int | None = None
    lat: float = 0.0
    lon: float = 0.0
    alt_rel_home_m: float = 0.0
    alt_amsl_m: float = 0.0
    groundspeed_ms: float = 0.0
    climb_ms: float = 0.0
    heading_deg: float = 0.0
    battery_pct: float | None = None
    battery_charge_state: int | None = None
    battery_voltage_v: float | None = None
    gps_fix_type: int = 0
    gps_satellites: int = 0
    ekf_ok: bool = False
    home: LatLonAlt | None = None
    mission_seq_current: int | None = None
    fc_firmware: str | None = None
    last_failsafe_signal: FailsafeSignal | None = None
    last_failsafe_mono_ms: int | None = None  # any failsafe signal, repeated ones too
    failsafe_onset_mono_ms: int | None = None  # a signal that was not active just before
    rc_mode_pwm: int | None = None  # the RC mode-switch channel; None without an RC receiver
    rc_mono_ms: int | None = None  # last RC_CHANNELS with live sticks from an RC receiver
    rc_receiver_healthy: bool | None = None  # SYS_STATUS: RC receiver present and healthy

    @property
    def fc_mode(self) -> str:
        return "UNKNOWN" if self.fc_custom_mode is None else mode_name(self.fc_custom_mode)

    def to_vehicle_state(
        self,
        vehicle_id: str,
        now_mono_ms: int,
        pilot_position: LatLon,
        pilot_position_source: str = "charter",
    ) -> VehicleState:
        """The shared ``VehicleState`` model for these values."""
        return VehicleState.model_validate(
            {
                "vehicle_id": vehicle_id,
                "mono_ms": now_mono_ms,
                "fc_connected": self.fc_connected,
                "gcs_mavlink_seen_mono_ms": self.gcs_mavlink_seen_mono_ms,
                "armed": self.armed,
                "fc_mode": self.fc_mode,
                "fc_system_status": self.fc_system_status,
                "landed_state": self.landed_state,
                "lat": self.lat,
                "lon": self.lon,
                "alt_rel_home_m": self.alt_rel_home_m,
                "alt_amsl_m": self.alt_amsl_m,
                "groundspeed_ms": self.groundspeed_ms,
                "climb_ms": self.climb_ms,
                "heading_deg": self.heading_deg,
                "battery_pct": self.battery_pct,
                "battery_charge_state": self.battery_charge_state,
                "battery_voltage_v": self.battery_voltage_v,
                "gps_fix_type": self.gps_fix_type,
                "gps_satellites": self.gps_satellites,
                "ekf_ok": self.ekf_ok,
                "home": self.home,
                "mission_seq_current": self.mission_seq_current,
                "pilot_position": pilot_position,
                "pilot_position_source": pilot_position_source,
            }
        )


def rc_link_ok(t: FcTelemetry, now_mono_ms: int) -> bool:
    """A physical RC transmitter is on and the flight controller hears it: fresh
    ``RC_CHANNELS`` with live sticks and a healthy RC receiver in ``SYS_STATUS``."""
    return (
        t.rc_receiver_healthy is True
        and t.rc_mono_ms is not None
        and now_mono_ms - t.rc_mono_ms <= RC_LINK_FRESH_S * 1000
    )


def _text(value: Any) -> str:
    if isinstance(value, bytes):
        value = value.decode("utf-8", errors="replace")
    return str(value).rstrip("\x00")


class _Waiter:
    __slots__ = ("future", "predicate")

    def __init__(self, predicate: Callable[[Any], bool], future: asyncio.Future[Any]) -> None:
        self.predicate = predicate
        self.future = future


class Vehicle:
    """One MAVLink channel to the flight controller ``fc_sysid`` (compid 1)."""

    def __init__(
        self,
        transport: MavTransport,
        *,
        role: Role,
        clock: Clock,
        fc_sysid: int = 1,
        envelope: SendEnvelope | None = None,
        on_event: EventSink | None = None,
        heartbeat_allowed: Callable[[], bool] | None = None,
        send_gate: SendGate | None = None,
    ) -> None:
        check_identity(role, transport.source_system, transport.source_component)
        if role is Role.ONBOARD and transport.source_system != fc_sysid:
            raise ValueError("the onboard computer sends as the flight controller's sysid")
        self.role = role
        self._event_prefix = "executor" if role is Role.ONBOARD else "mav"
        self.profile = Profile.ONBOARD if role is Role.ONBOARD else Profile.GROUND
        self.fc_sysid = fc_sysid
        self._transport = transport
        self._clock = clock
        self._envelope = envelope or SendEnvelope()
        self._on_event = on_event
        self._heartbeat_allowed = heartbeat_allowed
        self._send_gate = send_gate
        self._t = FcTelemetry()
        self._subscribers: list[Callable[[FcTelemetry], None]] = []
        self._waiters: list[_Waiter] = []
        self._command_locks: dict[int, asyncio.Lock] = {}
        self._tasks: list[asyncio.Task[None]] = []
        self._streams_task: asyncio.Task[None] | None = None
        self._signal_seen_ms: dict[str, int] = {}
        self._connected_event = asyncio.Event()
        self._closed = False
        self.blocked_count = 0
        self.event_errors = 0
        self.sent_count = 0

    # --- the flight's telemetry log ------------------------------------------------------

    def start_tlog(self, path: Path) -> bool:
        """Record every message this channel receives into ``path`` (``flight.tlog``).
        False when the transport cannot record (a test double)."""
        start = getattr(self._transport, "start_tlog", None)
        if start is None:
            return False
        path.parent.mkdir(parents=True, exist_ok=True)
        start(path, self._clock)
        return True

    def stop_tlog(self) -> None:
        stop = getattr(self._transport, "stop_tlog", None)
        if stop is not None:
            stop()

    # --- lifecycle ---------------------------------------------------------------------

    async def start(self) -> None:
        self._transport.attach(self._on_message)
        self._tasks.append(asyncio.create_task(self._heartbeat_loop(), name="mav-heartbeat"))
        self._tasks.append(asyncio.create_task(self._watch_loop(), name="mav-watch"))

    async def close(self) -> None:
        self._closed = True
        tasks = [*self._tasks, *([self._streams_task] if self._streams_task else [])]
        try:
            for task in tasks:
                task.cancel()
            await asyncio.gather(*tasks, return_exceptions=True)
        finally:
            for waiter in self._waiters:
                if not waiter.future.done():
                    waiter.future.cancel()
            self._waiters.clear()
            self._transport.close()

    async def __aenter__(self) -> Vehicle:
        await self.start()
        return self

    async def __aexit__(self, *_exc: object) -> None:
        await self.close()

    # --- state -------------------------------------------------------------------------

    @property
    def telemetry(self) -> FcTelemetry:
        return self._t

    @property
    def clock(self) -> Clock:
        return self._clock

    @property
    def envelope(self) -> SendEnvelope:
        return self._envelope

    def set_envelope(self, envelope: SendEnvelope) -> None:
        """Use the number limits of the current mission (``SendEnvelope.from_limits``)."""
        self._envelope = envelope

    def subscribe(self, callback: Callable[[FcTelemetry], None]) -> Callable[[], None]:
        """Call ``callback`` on every position update and HEARTBEAT; returns an unsubscribe."""
        self._subscribers.append(callback)
        return lambda: self._subscribers.remove(callback)

    async def wait_connected(self, within_s: float) -> bool:
        """True once a HEARTBEAT from the flight controller has arrived."""
        if self._t.fc_connected:
            return True
        return await self._wait_event(self._connected_event, within_s)

    async def _wait_event(self, event: asyncio.Event, within_s: float) -> bool:
        waiter = asyncio.ensure_future(event.wait())
        sleeper = asyncio.ensure_future(self._clock.sleep(within_s))
        try:
            await asyncio.wait({waiter, sleeper}, return_when=asyncio.FIRST_COMPLETED)
        finally:
            for task in (waiter, sleeper):
                task.cancel()
        return event.is_set()

    async def wait_message(self, predicate: Callable[[Any], bool], within_s: float) -> Any | None:
        """The next message from the flight controller matching ``predicate``, or ``None``."""
        future: asyncio.Future[Any] = asyncio.get_running_loop().create_future()
        waiter = _Waiter(predicate, future)
        self._waiters.append(waiter)
        sleeper = asyncio.ensure_future(self._clock.sleep(within_s))
        try:
            await asyncio.wait({future, sleeper}, return_when=asyncio.FIRST_COMPLETED)
        finally:
            sleeper.cancel()
            if waiter in self._waiters:
                self._waiters.remove(waiter)
        if future.done() and not future.cancelled():
            return future.result()
        future.cancel()
        return None

    def _context(self, origin: SendOrigin) -> SendContext:
        t = self._t
        now = self._clock.monotonic_ms()
        fresh_ms = MODE_FRESH_S * 1000
        mode_fresh = t.fc_heartbeat_mono_ms is not None and now - t.fc_heartbeat_mono_ms <= fresh_ms
        on_ground = (
            t.landed_state == "on_ground"
            and t.alt_rel_home_m <= GROUND_DISARM_MAX_ALT_M
            and t.landed_mono_ms is not None
            and now - t.landed_mono_ms <= fresh_ms
            and t.position_mono_ms is not None
            and now - t.position_mono_ms <= fresh_ms
        )
        landed_fresh = t.landed_mono_ms is not None and now - t.landed_mono_ms <= fresh_ms
        armed = t.armed if mode_fresh else None
        return SendContext(
            profile=self.profile,
            origin=str(origin),
            target_system=self.fc_sysid,
            envelope=self._envelope,
            home=(t.home.lat, t.home.lon) if t.home else None,
            fc_mode=t.fc_custom_mode if mode_fresh else None,
            on_ground=on_ground,
            armed=armed,
            airborne=bool(armed) and landed_fresh and t.landed_state in _AIRBORNE_STATES,
        )

    def would_refuse(self, msg: Any, *, origin: SendOrigin) -> str | None:
        """Why ``send`` would refuse ``msg`` right now, or ``None``; nothing is sent."""
        return denied_reason(msg) or check(msg, self._context(origin))

    def _event(self, kind: str, details: dict[str, Any]) -> None:
        if self._on_event is None:
            return
        try:
            self._on_event(kind, details)
        except Exception:  # the event sink must never stop the link's own work
            self.event_errors += 1

    # --- sending -----------------------------------------------------------------------

    def _check_and_write(self, msg: Any, origin: SendOrigin) -> None:
        if not isinstance(origin, SendOrigin):
            raise TypeError("origin must be a SendOrigin")
        reason = denied_reason(msg) or check(msg, self._context(origin))
        if reason is not None:
            self.blocked_count += 1
            self._event(
                f"{self._event_prefix}.forbidden_mavlink_blocked",
                {"msg_type": msg.get_type(), "reason": reason, "origin": str(origin)},
            )
            raise ForbiddenMavlinkError(msg.get_type(), reason)
        if self._send_gate is not None:
            closed = self._send_gate(origin)
            if closed is not None:
                raise ControlNotAllowedError(closed)
        self._transport.write_message(msg)
        self.sent_count += 1

    async def send(
        self,
        msg: Any,
        *,
        origin: SendOrigin,
        attempts: int = ACK_ATTEMPTS,
        ack_timeout_s: float = MAV_ACK_TIMEOUT_S,
    ) -> MavResult | None:
        """Send ``msg`` if the allowlist allows it.

        A ``COMMAND_LONG`` waits for its ``COMMAND_ACK`` (``ack_timeout_s`` per attempt,
        ``attempts`` attempts with a rising ``confirmation``) and returns the result.
        Every attempt is checked again. A command that moves the vehicle is not sent
        again (``CANCELLED``) once a new failsafe began (a signal that was quiet before;
        a failsafe that was already active does not count) or the flight controller
        changed to a mode that is neither the starting one nor the one asked for, so a
        retry never overrides the flight controller's own decision. A mode change whose
        answer was lost but whose mode the HEARTBEAT already shows returns ``ACCEPTED``.
        Callers confirm a mode through the HEARTBEAT; an acknowledgement only says the
        command arrived.
        Other messages return ``None``.
        """
        if msg.get_type() != "COMMAND_LONG":
            self._check_and_write(msg, origin)
            return None
        if not 1 <= attempts <= ACK_ATTEMPTS:
            raise ValueError(f"attempts must be 1..{ACK_ATTEMPTS}")
        command = int(msg.command)
        moves = command in _MOVING_COMMANDS
        wanted_mode = int(msg.param2) if command == mavlink.MAV_CMD_DO_SET_MODE else None
        lock = self._command_locks.setdefault(command, asyncio.Lock())
        async with lock:
            start_mode = self._t.fc_custom_mode
            start_onset = self._t.failsafe_onset_mono_ms
            for attempt in range(attempts):
                if attempt and moves:
                    t = self._t
                    fresh = (
                        t.fc_heartbeat_mono_ms is not None
                        and self._clock.monotonic_ms() - t.fc_heartbeat_mono_ms
                        <= MODE_FRESH_S * 1000
                    )
                    if fresh and wanted_mode is not None and t.fc_custom_mode == wanted_mode:
                        # The answer was lost, but the HEARTBEAT already shows the mode.
                        self._event("mav.ack_lost_mode_reached", {"mode": t.fc_mode})
                        return MavResult.ACCEPTED
                    changed = t.fc_custom_mode not in (start_mode, wanted_mode)
                    new_failsafe = t.failsafe_onset_mono_ms != start_onset
                    if new_failsafe or changed:
                        self._event(
                            "mav.retry_cancelled",
                            {"command": command, "mode": t.fc_mode, "failsafe": new_failsafe},
                        )
                        return MavResult.CANCELLED
                msg.confirmation = attempt
                self._check_and_write(msg, origin)
                ack = await self.wait_message(
                    lambda m: (
                        m.get_type() == "COMMAND_ACK"
                        and int(m.command) == command
                        and self.addressed_to_us(m)
                        and int(m.result) != mavlink.MAV_RESULT_IN_PROGRESS
                    ),
                    ack_timeout_s,
                )
                if ack is not None:
                    return MavResult.of(int(ack.result))
        return MavResult.TIMEOUT

    def _command(self, command: int, *params: float) -> Any:
        p = [*params, *([0.0] * (7 - len(params)))]
        return mavlink.MAVLink_command_long_message(self.fc_sysid, FC_COMPID, command, 0, *p[:7])

    async def set_mode(self, mode: CopterMode, *, origin: SendOrigin) -> MavResult:
        msg = self._command(
            mavlink.MAV_CMD_DO_SET_MODE, mavlink.MAV_MODE_FLAG_CUSTOM_MODE_ENABLED, int(mode)
        )
        return await self._send_command(msg, origin)

    async def arm(self, *, origin: SendOrigin) -> MavResult:
        return await self._send_command(
            self._command(mavlink.MAV_CMD_COMPONENT_ARM_DISARM, 1, 0), origin
        )

    async def disarm(self, *, origin: SendOrigin) -> MavResult:
        """Ground station only, and only while the vehicle stands on the ground."""
        return await self._send_command(
            self._command(mavlink.MAV_CMD_COMPONENT_ARM_DISARM, 0, 0), origin
        )

    async def takeoff(self, alt_m: float, *, origin: SendOrigin) -> MavResult:
        msg = self._command(mavlink.MAV_CMD_NAV_TAKEOFF, 0, 0, 0, 0, 0, 0, alt_m)
        return await self._send_command(msg, origin)

    async def change_speed(self, speed_ms: float, *, origin: SendOrigin) -> MavResult:
        msg = self._command(mavlink.MAV_CMD_DO_CHANGE_SPEED, 1, speed_ms, -1, 0)
        return await self._send_command(msg, origin)

    async def set_mission_current(self, seq: int, *, origin: SendOrigin) -> MavResult:
        msg = self._command(mavlink.MAV_CMD_DO_SET_MISSION_CURRENT, seq)
        return await self._send_command(msg, origin)

    async def set_message_interval(
        self, msg_id: int, interval_us: int, *, origin: SendOrigin
    ) -> MavResult:
        msg = self._command(mavlink.MAV_CMD_SET_MESSAGE_INTERVAL, msg_id, interval_us)
        return await self._send_command(msg, origin)

    async def request_message(self, msg_id: int, *, origin: SendOrigin) -> MavResult:
        return await self._send_command(
            self._command(mavlink.MAV_CMD_REQUEST_MESSAGE, msg_id), origin
        )

    async def _send_command(self, msg: Any, origin: SendOrigin) -> MavResult:
        result = await self.send(msg, origin=origin)
        return MavResult.TIMEOUT if result is None else result

    async def goto(self, lat: float, lon: float, alt_m: float, *, origin: SendOrigin) -> None:
        """A GUIDED position target, height above home."""
        msg = mavlink.MAVLink_set_position_target_global_int_message(
            0,
            self.fc_sysid,
            FC_COMPID,
            mavlink.MAV_FRAME_GLOBAL_RELATIVE_ALT_INT,
            POSITION_TYPE_MASK,
            round(lat * 1e7),
            round(lon * 1e7),
            alt_m,
            0,
            0,
            0,
            0,
            0,
            0,
            0,
            0,
        )
        await self.send(msg, origin=origin)

    async def manual_control(self, axes: StickAxes, *, origin: SendOrigin) -> None:
        """The pilot's stick values as ``MANUAL_CONTROL`` (only in LOITER).

        pitch/roll/yaw -1..1 become -1000..1000; throttle 0..1 becomes 0..1000, where
        500 holds the height in LOITER.
        """

        def scale(v: float) -> int:
            return max(-1000, min(1000, round(v * 1000)))

        msg = mavlink.MAVLink_manual_control_message(
            self.fc_sysid,
            scale(axes.pitch),
            scale(axes.roll),
            max(0, min(1000, round(axes.throttle * 1000))),
            scale(axes.yaw),
            0,
        )
        await self.send(msg, origin=origin)

    async def read_param(
        self, name: str, *, origin: SendOrigin, within_s: float = 2.0
    ) -> float | None:
        """One parameter's value by name, or ``None`` if the flight controller did not answer."""
        msg = mavlink.MAVLink_param_request_read_message(
            self.fc_sysid, FC_COMPID, name.encode("ascii"), -1
        )
        self._check_and_write(msg, origin)
        reply = await self.wait_message(
            lambda m: m.get_type() == "PARAM_VALUE" and _text(m.param_id) == name, within_s
        )
        return None if reply is None else float(reply.param_value)

    async def set_fence_limit(
        self, name: str, value: float, *, origin: SendOrigin, within_s: float = 2.0
    ) -> float | None:
        """Move one of the fence's limits (height, radius, LOITER speed); returns the
        value the flight controller reports back, ``None`` without an answer.

        Only the ground station may, only on the ground and disarmed, and never below
        the standard fence: the allowlist checks every one of those.
        """
        msg = mavlink.MAVLink_param_set_message(
            self.fc_sysid,
            FC_COMPID,
            name.encode("ascii"),
            float(value),
            mavlink.MAV_PARAM_TYPE_REAL32,
        )
        self._check_and_write(msg, origin)
        reply = await self.wait_message(
            lambda m: m.get_type() == "PARAM_VALUE" and _text(m.param_id) == name, within_s
        )
        return None if reply is None else float(reply.param_value)

    # --- background tasks --------------------------------------------------------------

    def _our_heartbeat(self) -> Any:
        mav_type = (
            mavlink.MAV_TYPE_GCS
            if self.role is Role.GROUND
            else mavlink.MAV_TYPE_ONBOARD_CONTROLLER
        )
        return mavlink.MAVLink_heartbeat_message(
            mav_type, mavlink.MAV_AUTOPILOT_INVALID, 0, 0, mavlink.MAV_STATE_ACTIVE, 3
        )

    @property
    def _service_origin(self) -> SendOrigin:
        return SendOrigin.GROUND if self.role is Role.GROUND else SendOrigin.STARTUP

    async def _heartbeat_loop(self) -> None:
        while not self._closed:
            try:
                if self._heartbeat_allowed is None or self._heartbeat_allowed():
                    self._check_and_write(self._our_heartbeat(), self._service_origin)
            except Exception as exc:  # keep beating; the next second tries again
                self._event("mav.heartbeat_error", {"error": type(exc).__name__})
            await self._clock.sleep(_OUR_HEARTBEAT_S)

    async def _watch_loop(self) -> None:
        # The link is gone from the moment the next 5 Hz HEARTBEAT fails to arrive, so it
        # is declared lost one heartbeat interval and one check before the timeout: a
        # lost connection is noticed within FC_HEARTBEAT_TIMEOUT_S of the cut, even when
        # the last heartbeat was still on its way.
        timeout_ms = (FC_HEARTBEAT_TIMEOUT_S - HEARTBEAT_INTERVAL_US / 1e6 - _WATCH_S) * 1000
        while not self._closed:
            await self._clock.sleep(_WATCH_S)
            t = self._t
            if t.fc_connected and t.fc_heartbeat_mono_ms is not None:
                age = self._clock.monotonic_ms() - t.fc_heartbeat_mono_ms
                if age >= timeout_ms:
                    self._t = replace(t, fc_connected=False)
                    self._connected_event.clear()
                    if self._streams_task is not None and not self._streams_task.done():
                        # Ask again from the start once the flight controller is back.
                        self._streams_task.cancel()
                    try:
                        self._event("mav.fc_lost", {"heartbeat_age_ms": age})
                    finally:
                        self._notify()

    async def _request_streams(self) -> None:
        streams = GROUND_STREAM_US if self.role is Role.GROUND else ONBOARD_STREAM_US
        origin = self._service_origin
        failed: list[int] = []
        for msg_id, interval in streams.items():
            result = await self.set_message_interval(msg_id, interval, origin=origin)
            if result is not MavResult.ACCEPTED:
                failed.append(msg_id)
        if self.role is Role.ONBOARD:
            for msg_id in (
                mavlink.MAVLINK_MSG_ID_AUTOPILOT_VERSION,
                mavlink.MAVLINK_MSG_ID_HOME_POSITION,
            ):
                await self.request_message(msg_id, origin=origin)
        self._event("mav.streams_requested", {"failed": failed})

    def _streams_done(self, task: asyncio.Task[None]) -> None:
        if not task.cancelled() and task.exception() is not None:
            self._event("mav.streams_failed", {"error": type(task.exception()).__name__})

    # --- receiving ---------------------------------------------------------------------

    def _notify(self) -> None:
        for callback in list(self._subscribers):
            try:
                callback(self._t)
            except Exception as exc:  # a subscriber must never stop the link's own work
                self._event("mav.subscriber_error", {"error": type(exc).__name__})

    def addressed_to_us(self, msg: Any) -> bool:
        """True when a reply names this channel's sender (or names nobody, as older
        firmware does), so a reply meant for the other channel is never taken as ours."""
        target = (int(getattr(msg, "target_system", 0)), int(getattr(msg, "target_component", 0)))
        ours = self._transport.source_system
        return target in ((ours, self._transport.source_component), (ours, 0), (0, 0))

    def _on_message(self, msg: Any) -> None:
        if self._closed:
            return
        src = (msg.get_srcSystem(), msg.get_srcComponent())
        kind = msg.get_type()
        if src == (self.fc_sysid, FC_COMPID):
            self._from_fc(msg, kind)
            for waiter in list(self._waiters):
                if not waiter.future.done() and waiter.predicate(msg):
                    waiter.future.set_result(msg)
                    self._waiters.remove(waiter)
            return
        now = self._clock.monotonic_ms()
        if kind == "HEARTBEAT":
            if src == (GCS_SYSID, GCS_COMPID) and int(msg.type) == mavlink.MAV_TYPE_GCS:
                self._t = replace(self._t, gcs_mavlink_seen_mono_ms=now)
            elif src == (self.fc_sysid, ONBOARD_COMPID):
                self._t = replace(self._t, onboard_seen_mono_ms=now)
            return
        if kind in ("COMMAND_LONG", "COMMAND_INT") and int(msg.target_component) == (
            self._transport.source_component
        ):
            # Nothing is accepted as a command over MAVLink.
            self._event(
                f"{self._event_prefix}.inbound_command_ignored",
                {"sysid": src[0], "compid": src[1], "command": int(msg.command)},
            )

    def _from_fc(self, msg: Any, kind: str) -> None:
        now = self._clock.monotonic_ms()
        t = self._t
        signal = signal_from(msg)
        if signal is not None:
            seen = self._signal_seen_ms.get(signal.name)
            self._signal_seen_ms[signal.name] = now
            # ArduPilot repeats an active failsafe state (CRITICAL heartbeat, LOW battery)
            # several times a second; only a signal that was quiet before is a new one.
            onset = seen is None or now - seen > _SIGNAL_QUIET_MS
            t = replace(
                t,
                last_failsafe_signal=signal,
                last_failsafe_mono_ms=now,
                failsafe_onset_mono_ms=now if onset else t.failsafe_onset_mono_ms,
            )
            self._event("mav.failsafe_signal", {"source": signal.source, "name": signal.name})
        notify = False
        match kind:
            case "HEARTBEAT":
                if int(msg.type) in (mavlink.MAV_TYPE_GCS, mavlink.MAV_TYPE_ONBOARD_CONTROLLER):
                    self._t = t
                    return
                was_connected = t.fc_connected
                t = replace(
                    t,
                    fc_connected=True,
                    fc_heartbeat_mono_ms=now,
                    armed=bool(int(msg.base_mode) & _ARMED_FLAG),
                    fc_custom_mode=int(msg.custom_mode),
                    fc_system_status=int(msg.system_status),
                )
                notify = True
                if not was_connected:
                    self._t = t
                    self._connected_event.set()
                    self._event("mav.fc_connected", {"mode": t.fc_mode})
                    if self._streams_task is None or self._streams_task.done():
                        self._streams_task = asyncio.ensure_future(self._request_streams())
                        self._streams_task.add_done_callback(self._streams_done)
            case "GLOBAL_POSITION_INT":
                vx, vy = int(msg.vx) / 100.0, int(msg.vy) / 100.0
                hdg = int(msg.hdg)
                t = replace(
                    t,
                    position_mono_ms=now,
                    lat=int(msg.lat) / 1e7,
                    lon=int(msg.lon) / 1e7,
                    alt_rel_home_m=int(msg.relative_alt) / 1000.0,
                    alt_amsl_m=int(msg.alt) / 1000.0,
                    groundspeed_ms=math.hypot(vx, vy),
                    climb_ms=-int(msg.vz) / 100.0,
                    heading_deg=t.heading_deg if hdg == 65535 else hdg / 100.0,
                )
                notify = True
            case "VFR_HUD":
                t = replace(t, groundspeed_ms=float(msg.groundspeed), climb_ms=float(msg.climb))
            case "BATTERY_STATUS":
                remaining = int(msg.battery_remaining)
                volts = int(msg.voltages[0]) if msg.voltages else 65535
                t = replace(
                    t,
                    battery_pct=None if remaining < 0 else float(remaining),
                    battery_charge_state=int(msg.charge_state),
                    battery_voltage_v=None if volts == 65535 else volts / 1000.0,
                )
            case "SYS_STATUS":
                rc_bit = mavlink.MAV_SYS_STATUS_SENSOR_RC_RECEIVER
                present = bool(int(msg.onboard_control_sensors_present) & rc_bit)
                healthy = bool(int(msg.onboard_control_sensors_health) & rc_bit)
                t = replace(t, rc_receiver_healthy=present and healthy)
                if t.battery_voltage_v is None and int(msg.voltage_battery) != 65535:
                    t = replace(t, battery_voltage_v=int(msg.voltage_battery) / 1000.0)
            case "GPS_RAW_INT":
                sats = int(msg.satellites_visible)
                t = replace(
                    t, gps_fix_type=int(msg.fix_type), gps_satellites=0 if sats == 255 else sats
                )
            case "EKF_STATUS_REPORT":
                flags = int(msg.flags)
                ok = (flags & _EKF_NEEDED) == _EKF_NEEDED and not (
                    flags & mavlink.EKF_CONST_POS_MODE
                )
                t = replace(t, ekf_ok=ok)
            case "EXTENDED_SYS_STATE":
                t = replace(
                    t,
                    landed_state=_LANDED.get(int(msg.landed_state), "unknown"),
                    landed_mono_ms=now,
                )
            case "RC_CHANNELS":
                count = int(msg.chancount)
                raw = int(getattr(msg, f"chan{RC_MODE_CHANNEL}_raw", 0))
                usable = count >= RC_MODE_CHANNEL and 800 <= raw <= 2200
                t = replace(t, rc_mode_pwm=raw if usable else None)
                throttle = int(getattr(msg, f"chan{_RC_THROTTLE_CHANNEL}_raw", 0))
                if count >= _RC_MIN_CHANNELS and 800 <= throttle <= 2200:
                    t = replace(t, rc_mono_ms=now)
            case "MISSION_CURRENT":
                t = replace(t, mission_seq_current=int(msg.seq))
            case "HOME_POSITION":
                lat, lon = int(msg.latitude) / 1e7, int(msg.longitude) / 1e7
                with contextlib.suppress(ValueError):
                    t = replace(
                        t, home=LatLonAlt(lat=lat, lon=lon, alt_amsl_m=int(msg.altitude) / 1000.0)
                    )
            case "AUTOPILOT_VERSION":
                v = int(msg.flight_sw_version)
                t = replace(t, fc_firmware=f"{v >> 24 & 0xFF}.{v >> 16 & 0xFF}.{v >> 8 & 0xFF}")
            case "STATUSTEXT":
                self._event(
                    "mav.statustext", {"severity": int(msg.severity), "text": _text(msg.text)}
                )
            case _:
                pass
        self._t = t
        if notify:
            self._notify()
