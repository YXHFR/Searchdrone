"""A fake flight controller in memory, for tests and ``just dev`` without a simulator.

``FakeFlightController`` speaks real MAVLink 2 bytes with every link attached to it,
answers commands, the mission and fence protocol and parameter reads, flies a very
simple point-mass model and sends the telemetry streams. Tests can make it refuse
modes, go silent, change mode on its own or report a failsafe.

``FakeVehicle`` is a :class:`~sd_mav.vehicle.Vehicle` on such a link; ``sent`` lists
every message that reached the fake flight controller from it, so tests can prove that
nothing forbidden ever left.
"""

from __future__ import annotations

import asyncio
import contextlib
import math
import struct
from collections.abc import Callable
from dataclasses import dataclass, field
from typing import Any, Final

from pymavlink.dialects.v20 import ardupilotmega as mavlink

from sd_mav.allowlist import FC_COMPID
from sd_mav.connection import GCS_COMPID, GCS_SYSID, ONBOARD_COMPID, Role
from sd_mav.mission import FENCE, MISSION, MissionItem
from sd_mav.modes import CopterMode
from sd_mav.vehicle import Vehicle
from sd_obs.clock import Clock
from sd_safety.constants import FC_SAFETY_EXPECTED
from sd_safety.targets import distance_m

__all__ = ["CMAC_HOME", "FakeFlightController", "FakeLink", "FakeVehicle", "Received"]

CMAC_HOME: Final = (-35.363261, 149.165230, 584.0)  # the public ArduPilot SITL site
_TICK_S: Final = 0.1
_CLIMB_MS: Final = 2.5
_DEFAULT_SPEED_MS: Final = 5.0
_M_PER_DEG: Final = 111_320.0


def _f32(value: float) -> float:
    return struct.unpack("<f", struct.pack("<f", value))[0]


@dataclass(frozen=True, slots=True)
class Received:
    """One message that reached the fake flight controller."""

    sysid: int
    compid: int
    msg: Any


class FakeLink:
    """One MAVLink channel to a :class:`FakeFlightController` (a ``MavTransport``)."""

    def __init__(self, fc: FakeFlightController, source_system: int, source_component: int):
        self._fc = fc
        self._source_system = source_system
        self._source_component = source_component
        self._encoder = mavlink.MAVLink(
            None, srcSystem=source_system, srcComponent=source_component
        )
        self._parser = mavlink.MAVLink(None)
        self._parser.robust_parsing = True
        self._handler: Callable[[Any], None] | None = None
        self._loop: asyncio.AbstractEventLoop | None = None
        self.up = True  # False: nothing passes in either direction
        self.closed = False

    @property
    def source_system(self) -> int:
        return self._source_system

    @property
    def source_component(self) -> int:
        return self._source_component

    def write_message(self, msg: Any) -> None:
        if self.closed:
            raise ConnectionError("link closed")
        data = msg.pack(self._encoder)
        if self.up:
            self._fc.deliver(self, bytes(data))

    def attach(self, on_message: Callable[[Any], None]) -> None:
        self._handler = on_message
        self._loop = asyncio.get_running_loop()

    def receive(self, data: bytes) -> None:
        """Bytes from the fake flight controller; delivered on the next loop turn."""
        if not self.up or self.closed or self._handler is None or self._loop is None:
            return
        for msg in self._parser.parse_buffer(data) or []:
            if msg.get_type() != "BAD_DATA":
                self._loop.call_soon(self._handler, msg)

    def close(self) -> None:
        self.closed = True


@dataclass
class _Upload:
    mission_type: int
    count: int
    items: dict[int, MissionItem] = field(default_factory=dict)


class FakeFlightController:
    """ArduCopter in a few hundred lines. Not physics: just enough for protocol tests."""

    def __init__(
        self, clock: Clock, *, sysid: int = 1, home: tuple[float, float, float] = CMAC_HOME
    ):
        self.clock = clock
        self.sysid = sysid
        self.home = home
        self._encoder = mavlink.MAVLink(None, srcSystem=sysid, srcComponent=FC_COMPID)
        self._parser = mavlink.MAVLink(None)
        self._parser.robust_parsing = True
        self.links: list[FakeLink] = []
        self.received: list[Received] = []
        self.mode = CopterMode.STABILIZE
        self.armed = False
        self.lat, self.lon, self.alt_rel = home[0], home[1], 0.0
        self.target: tuple[float, float, float] | None = None
        self.speed_ms = _DEFAULT_SPEED_MS
        self.battery_pct = 100.0
        self.charge_state = mavlink.MAV_BATTERY_CHARGE_STATE_OK
        self.system_status = mavlink.MAV_STATE_STANDBY
        self.missions: dict[int, list[MissionItem]] = {MISSION: [], FENCE: []}
        self.mission_current = 0
        self.params: dict[str, float] = {k: float(v) for k, v in FC_SAFETY_EXPECTED.items()}
        self.intervals_us: dict[int, int] = {}
        self.refuse_modes: set[int] = set()
        self.drop_acks = 0  # the next n COMMAND_ACKs are lost
        self.silent = False  # no heartbeat or telemetry, no answers
        self.sticks: Any | None = None  # last MANUAL_CONTROL
        self._upload: _Upload | None = None
        self._task: asyncio.Task[None] | None = None

    # --- links -----------------------------------------------------------------------

    def link(self, source_system: int, source_component: int) -> FakeLink:
        link = FakeLink(self, source_system, source_component)
        self.links.append(link)
        return link

    def deliver(self, link: FakeLink, data: bytes) -> None:
        for msg in self._parser.parse_buffer(data) or []:
            if msg.get_type() == "BAD_DATA":
                continue
            self.received.append(Received(msg.get_srcSystem(), msg.get_srcComponent(), msg))
            # ArduPilot forwards untargeted messages (like HEARTBEAT) to its other links.
            if msg.get_type() == "HEARTBEAT":
                for other in self.links:
                    if other is not link:
                        other.receive(data)
            if not self.silent:
                self._handle(link, msg)

    def _send(self, msg: Any, link: FakeLink | None = None) -> None:
        if self.silent:
            return
        data = bytes(msg.pack(self._encoder))
        for target in [link] if link is not None else self.links:
            target.receive(data)

    # --- time ------------------------------------------------------------------------

    def start(self) -> None:
        if self._task is None or self._task.done():
            self._task = asyncio.create_task(self._run(), name="fake-fc")

    async def stop(self) -> None:
        if self._task is not None:
            self._task.cancel()
            with contextlib.suppress(asyncio.CancelledError):
                await self._task
            self._task = None

    async def _run(self) -> None:
        tick = 0
        while True:
            self.step(_TICK_S)
            self._send(self._position())
            if tick % 2 == 0:
                self._send(self.heartbeat())
            if tick % 5 == 0:
                for msg in (
                    self._extended_state(),
                    self._battery(),
                    self._gps(),
                    self._ekf(),
                    self._sys_status(),
                ):
                    self._send(msg)
            if tick % 10 == 0:
                self._send(self._home_position())
            tick += 1
            await self.clock.sleep(_TICK_S)

    def step(self, dt: float) -> None:
        """Move the point mass for ``dt`` seconds."""
        if not self.armed:
            return
        if self.mode in (CopterMode.RTL, CopterMode.SMART_RTL):
            self.target = (self.home[0], self.home[1], 0.0)
        if self.mode == CopterMode.LAND:
            self.target = (self.lat, self.lon, 0.0)
        if self.target is None:
            return
        lat, lon, alt = self.target
        north = (lat - self.lat) * _M_PER_DEG
        east = (lon - self.lon) * _M_PER_DEG * math.cos(math.radians(self.lat))
        dist = math.hypot(north, east)
        move = min(dist, self.speed_ms * dt)
        if dist > 0:
            self.lat += north / dist * move / _M_PER_DEG
            self.lon += east / dist * move / (_M_PER_DEG * math.cos(math.radians(self.lat)))
        climb = max(-_CLIMB_MS * dt, min(_CLIMB_MS * dt, alt - self.alt_rel))
        self.alt_rel = max(0.0, self.alt_rel + climb)
        landing = self.mode in (CopterMode.LAND, CopterMode.RTL, CopterMode.SMART_RTL)
        if landing and self.alt_rel <= 0.0 and dist < 1.0:
            self.armed = False
            self.target = None

    # --- what the fake sends ---------------------------------------------------------

    def heartbeat(self) -> Any:
        base = mavlink.MAV_MODE_FLAG_CUSTOM_MODE_ENABLED
        if self.armed:
            base |= mavlink.MAV_MODE_FLAG_SAFETY_ARMED
        status = self.system_status
        if status in (mavlink.MAV_STATE_STANDBY, mavlink.MAV_STATE_ACTIVE):
            status = mavlink.MAV_STATE_ACTIVE if self.armed else mavlink.MAV_STATE_STANDBY
        return mavlink.MAVLink_heartbeat_message(
            mavlink.MAV_TYPE_QUADROTOR,
            mavlink.MAV_AUTOPILOT_ARDUPILOTMEGA,
            base,
            int(self.mode),
            status,
            3,
        )

    def _position(self) -> Any:
        return mavlink.MAVLink_global_position_int_message(
            int(self.clock.monotonic_ms()) & 0xFFFFFFFF,
            round(self.lat * 1e7),
            round(self.lon * 1e7),
            round((self.home[2] + self.alt_rel) * 1000),
            round(self.alt_rel * 1000),
            0,
            0,
            0,
            0,
        )

    def landed_state(self) -> int:
        if not self.armed:
            return mavlink.MAV_LANDED_STATE_ON_GROUND
        if self.alt_rel < 0.3 and self.target is None:
            return mavlink.MAV_LANDED_STATE_ON_GROUND
        return mavlink.MAV_LANDED_STATE_IN_AIR

    def _extended_state(self) -> Any:
        return mavlink.MAVLink_extended_sys_state_message(0, self.landed_state())

    def _battery(self) -> Any:
        return mavlink.MAVLink_battery_status_message(
            0,
            0,
            0,
            3000,
            [16000] + [65535] * 9,
            -1,
            -1,
            -1,
            round(self.battery_pct),
            0,
            self.charge_state,
        )

    def _gps(self) -> Any:
        return mavlink.MAVLink_gps_raw_int_message(
            0, 3, round(self.lat * 1e7), round(self.lon * 1e7), 0, 100, 100, 0, 0, 12
        )

    def _ekf(self) -> Any:
        flags = (
            mavlink.EKF_ATTITUDE
            | mavlink.EKF_VELOCITY_HORIZ
            | mavlink.EKF_VELOCITY_VERT
            | mavlink.EKF_POS_HORIZ_ABS
            | mavlink.EKF_POS_VERT_ABS
        )
        return mavlink.MAVLink_ekf_status_report_message(flags, 0, 0, 0, 0, 0)

    def _sys_status(self) -> Any:
        return mavlink.MAVLink_sys_status_message(
            0, 0, 0, 0, 16000, -1, round(self.battery_pct), 0, 0, 0, 0, 0, 0
        )

    def _home_position(self) -> Any:
        return mavlink.MAVLink_home_position_message(
            round(self.home[0] * 1e7),
            round(self.home[1] * 1e7),
            round(self.home[2] * 1000),
            0,
            0,
            0,
            [1, 0, 0, 0],
            0,
            0,
            0,
        )

    def statustext(self, text: str, severity: int = mavlink.MAV_SEVERITY_WARNING) -> None:
        self._send(mavlink.MAVLink_statustext_message(severity, text.encode()))

    def set_mode_external(self, mode: CopterMode) -> None:
        """A mode change that Searchdrone did not ask for (a person or a failsafe)."""
        self.mode = mode
        self._send(self.heartbeat())

    # --- what the fake answers -------------------------------------------------------

    def _ack(self, link: FakeLink, command: int, result: int) -> None:
        if self.drop_acks > 0:
            self.drop_acks -= 1
            return
        self._send(mavlink.MAVLink_command_ack_message(command, result), link)

    def _handle(self, link: FakeLink, msg: Any) -> None:
        kind = msg.get_type()
        if kind == "COMMAND_LONG" and int(msg.target_system) == self.sysid:
            self._ack(link, int(msg.command), self._command(link, msg))
        elif kind == "SET_POSITION_TARGET_GLOBAL_INT" and self.mode == CopterMode.GUIDED:
            self.target = (msg.lat_int / 1e7, msg.lon_int / 1e7, float(msg.alt))
        elif kind == "PARAM_REQUEST_READ":
            name = msg.param_id if isinstance(msg.param_id, str) else msg.param_id.decode()
            name = name.rstrip("\x00")
            if name in self.params:
                self._send(
                    mavlink.MAVLink_param_value_message(
                        name.encode(),
                        self.params[name],
                        mavlink.MAV_PARAM_TYPE_REAL32,
                        len(self.params),
                        0,
                    ),
                    link,
                )
        elif kind == "MANUAL_CONTROL":
            self.sticks = msg
        elif kind == "MISSION_SET_CURRENT":
            self.mission_current = int(msg.seq)
        elif kind.startswith("MISSION_"):
            self._mission(link, msg, kind)

    def _command(self, link: FakeLink, msg: Any) -> int:
        cmd = int(msg.command)
        accepted, denied = mavlink.MAV_RESULT_ACCEPTED, mavlink.MAV_RESULT_DENIED
        if cmd == mavlink.MAV_CMD_DO_SET_MODE:
            mode = int(msg.param2)
            if mode in self.refuse_modes:
                return denied
            try:
                self.mode = CopterMode(mode)
            except ValueError:
                return denied
            if self.mode in (CopterMode.LOITER, CopterMode.BRAKE):
                self.target = None
            return accepted
        if cmd == mavlink.MAV_CMD_COMPONENT_ARM_DISARM:
            if int(msg.param1) == 1:
                self.armed = True
                return accepted
            if self.landed_state() == mavlink.MAV_LANDED_STATE_ON_GROUND:
                self.armed = False
                return accepted
            return denied
        if cmd == mavlink.MAV_CMD_NAV_TAKEOFF:
            if not self.armed or self.mode != CopterMode.GUIDED:
                return mavlink.MAV_RESULT_FAILED
            self.target = (self.lat, self.lon, float(msg.param7))
            return accepted
        if cmd == mavlink.MAV_CMD_DO_CHANGE_SPEED:
            self.speed_ms = float(msg.param2)
            return accepted
        if cmd == mavlink.MAV_CMD_SET_MESSAGE_INTERVAL:
            self.intervals_us[int(msg.param1)] = int(msg.param2)
            return accepted
        if cmd == mavlink.MAV_CMD_REQUEST_MESSAGE:
            if int(msg.param1) == mavlink.MAVLINK_MSG_ID_HOME_POSITION:
                self._send(self._home_position(), link)
            elif int(msg.param1) == mavlink.MAVLINK_MSG_ID_AUTOPILOT_VERSION:
                self._send(
                    mavlink.MAVLink_autopilot_version_message(
                        0,
                        (4 << 24) | (7 << 16) | (1 << 8),
                        0,
                        0,
                        0,
                        [0] * 8,
                        [0] * 8,
                        [0] * 8,
                        0,
                        0,
                        0,
                    ),
                    link,
                )
            return accepted
        if cmd == mavlink.MAV_CMD_DO_SET_MISSION_CURRENT:
            self.mission_current = int(msg.param1)
            return accepted
        return mavlink.MAV_RESULT_UNSUPPORTED

    def _mission(self, link: FakeLink, msg: Any, kind: str) -> None:
        mtype = int(msg.mission_type)
        if kind == "MISSION_COUNT":
            self._upload = _Upload(mtype, int(msg.count))
            self._request_next(link)
        elif kind == "MISSION_ITEM_INT" and self._upload is not None:
            item = MissionItem.from_message(msg)
            self._upload.items[item.seq] = item
            if len(self._upload.items) == self._upload.count:
                stored = [self._store(self._upload.items[i]) for i in range(self._upload.count)]
                self.missions[mtype] = stored
                self._upload = None
                self._send(
                    mavlink.MAVLink_mission_ack_message(
                        link.source_system,
                        link.source_component,
                        mavlink.MAV_MISSION_ACCEPTED,
                        mtype,
                    ),
                    link,
                )
            else:
                self._request_next(link)
        elif kind == "MISSION_REQUEST_LIST":
            self._send(
                mavlink.MAVLink_mission_count_message(
                    link.source_system,
                    link.source_component,
                    len(self.missions.get(mtype, [])),
                    mtype,
                ),
                link,
            )
        elif kind == "MISSION_REQUEST_INT":
            items = self.missions.get(mtype, [])
            seq = int(msg.seq)
            if 0 <= seq < len(items):
                self._send(items[seq].to_message(link.source_system), link)

    def _request_next(self, link: FakeLink) -> None:
        if self._upload is None:
            return
        seq = next(i for i in range(self._upload.count) if i not in self._upload.items)
        self._send(
            mavlink.MAVLink_mission_request_int_message(
                link.source_system, link.source_component, seq, self._upload.mission_type
            ),
            link,
        )

    def _store(self, item: MissionItem) -> MissionItem:
        """What ArduPilot gives back: item 0 is its home, values pass through float32."""
        if item.mission_type == MISSION and item.seq == 0:
            return MissionItem(
                0,
                item.command,
                item.frame,
                x=round(self.home[0] * 1e7),
                y=round(self.home[1] * 1e7),
                z=self.home[2],
            )
        return MissionItem(
            item.seq,
            item.command,
            item.frame,
            param1=_f32(item.param1),
            param2=_f32(item.param2),
            param3=_f32(item.param3),
            param4=_f32(item.param4),
            x=item.x,
            y=item.y,
            z=_f32(item.z),
            mission_type=item.mission_type,
        )

    def distance_from_home_m(self) -> float:
        return distance_m(self.home[0], self.home[1], self.lat, self.lon)


class FakeVehicle(Vehicle):
    """A ``Vehicle`` connected to a :class:`FakeFlightController`.

    ``start()`` also starts the fake flight controller's telemetry. ``sent`` is every
    message that reached the flight controller from this vehicle.
    """

    def __init__(
        self,
        clock: Clock,
        *,
        role: Role = Role.ONBOARD,
        fc: FakeFlightController | None = None,
        fc_sysid: int = 1,
        **kwargs: Any,
    ) -> None:
        self.fc = fc or FakeFlightController(clock, sysid=fc_sysid)
        ids = (self.fc.sysid, ONBOARD_COMPID) if role is Role.ONBOARD else (GCS_SYSID, GCS_COMPID)
        self.link = self.fc.link(*ids)
        super().__init__(self.link, role=role, clock=clock, fc_sysid=self.fc.sysid, **kwargs)

    @property
    def sent(self) -> list[Any]:
        ids = (self.link.source_system, self.link.source_component)
        return [r.msg for r in self.fc.received if (r.sysid, r.compid) == ids]

    async def start(self) -> None:
        await super().start()
        self.fc.start()

    async def close(self) -> None:
        await self.fc.stop()
        await super().close()
