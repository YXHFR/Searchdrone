"""The MAVLink connection and who may use which identity on it.

``sd_mav`` is the only production package that imports pymavlink; import it before
anything else imports ``pymavlink.mavutil``, which reads ``MAVLINK20`` once (a check
below refuses to run on MAVLink 1). A connection has a role: the onboard computer
(sysid = the flight controller's, compid 191) or the ground station (250/190). The
onboard computer can never pretend to be the ground station: only the ground station's
HEARTBEAT keeps the flight controller's GCS failsafe away.

Nothing outside :class:`sd_mav.vehicle.Vehicle` writes to a connection; ``Vehicle.send``
runs the allowlist before every message.
"""

from __future__ import annotations

import asyncio
import contextlib
import os
import struct
import threading
from collections.abc import Callable
from enum import StrEnum
from pathlib import Path
from typing import Any, BinaryIO, Final, Protocol

from sd_obs.clock import Clock

# MAVLink 2 everywhere. Set before pymavlink.mavutil is imported, which reads it.
os.environ.setdefault("MAVLINK20", "1")

from pymavlink import mavutil

__all__ = [
    "BROADCAST_SYSID",
    "GCS_COMPID",
    "GCS_SYSID",
    "ONBOARD_COMPID",
    "MavConnection",
    "MavTransport",
    "Role",
    "check_identity",
    "connect",
]

GCS_SYSID: Final = 250  # MAV_GCS_SYSID in the flight controller
GCS_COMPID: Final = 190  # MAV_COMP_ID_MISSIONPLANNER
ONBOARD_COMPID: Final = 191  # MAV_COMP_ID_ONBOARD_COMPUTER
BROADCAST_SYSID: Final = 255  # the default GCS id of most ground control programs

MessageHandler = Callable[[Any], None]


class Role(StrEnum):
    ONBOARD = "onboard"
    GROUND = "ground"


def check_identity(role: Role, source_system: int, source_component: int) -> None:
    """``ValueError`` unless ``role`` may send as ``source_system``/``source_component``."""
    if role is Role.GROUND:
        if (source_system, source_component) != (GCS_SYSID, GCS_COMPID):
            raise ValueError(f"the ground station sends as {GCS_SYSID}/{GCS_COMPID}")
        return
    if source_system in (GCS_SYSID, BROADCAST_SYSID) or not 1 <= source_system <= 249:
        raise ValueError(f"the onboard computer may not send as system {source_system}")
    if source_component != ONBOARD_COMPID:
        raise ValueError(f"the onboard computer sends as component {ONBOARD_COMPID}")


class MavTransport(Protocol):
    """Where a :class:`~sd_mav.vehicle.Vehicle` reads and writes MAVLink messages."""

    @property
    def source_system(self) -> int: ...

    @property
    def source_component(self) -> int: ...

    def write_message(self, msg: Any) -> None:
        """Pack and send one message. Only ``Vehicle.send`` calls this."""
        ...

    def attach(self, on_message: MessageHandler) -> None:
        """Start delivering every received message to ``on_message`` (event loop thread)."""
        ...

    def close(self) -> None: ...


class MavConnection:
    """A pymavlink connection (UDP, TCP or serial) with a checked identity.

    ``url`` uses pymavlink's syntax, e.g. ``udpin:127.0.0.1:14563``. A reader thread
    owns every receive call, so reconnects triggered by a read (a new socket or serial
    port, blocking retries) never run on the event loop; each message is handed to the
    loop with ``call_soon_threadsafe``. A read error waits ``READ_RETRY_S`` and, for a
    serial port, reopens it. Writes run on the caller's thread; for TCP and serial a
    failed write can make pymavlink reconnect there, so v0.1.0 uses UDP only.
    """

    READ_RETRY_S: Final = 0.5
    _POLL_S: Final = 0.2

    def __init__(self, url: str, *, role: Role, source_system: int, source_component: int):
        check_identity(role, source_system, source_component)
        self.role = role
        self.url = url
        self._source_system = source_system
        self._source_component = source_component
        self._conn: Any = mavutil.mavlink_connection(
            url,
            source_system=source_system,
            source_component=source_component,
            dialect="ardupilotmega",
            autoreconnect=True,
        )
        if not self._conn.mavlink20():
            self._conn.close()
            raise RuntimeError("pymavlink did not select MAVLink 2")
        self._loop: asyncio.AbstractEventLoop | None = None
        self._handler: MessageHandler | None = None
        self._reader: threading.Thread | None = None
        self._stop = threading.Event()
        self._tlog: BinaryIO | None = None
        self._tlog_clock: Clock | None = None
        self._tlog_lock = threading.Lock()
        self.read_errors = 0
        self.dispatch_errors = 0

    @property
    def source_system(self) -> int:
        return self._source_system

    @property
    def source_component(self) -> int:
        return self._source_component

    def write_message(self, msg: Any) -> None:
        self._conn.mav.send(msg)

    def attach(self, on_message: MessageHandler) -> None:
        self._handler = on_message
        self._loop = asyncio.get_running_loop()
        self._reader = threading.Thread(target=self._read_loop, name="mav-reader", daemon=True)
        self._reader.start()

    def _read_loop(self) -> None:
        while not self._stop.is_set():
            try:
                msg = self._conn.recv_match(blocking=True, timeout=self._POLL_S)
            except Exception:
                self.read_errors += 1
                self._stop.wait(self.READ_RETRY_S)
                reset = getattr(self._conn, "reset", None)
                if callable(reset) and not self._stop.is_set():
                    with contextlib.suppress(Exception):
                        reset()
                continue
            if msg is None or msg.get_type() == "BAD_DATA":
                continue
            self._log(msg)
            loop = self._loop
            if loop is None or loop.is_closed():
                return
            try:
                loop.call_soon_threadsafe(self._dispatch, msg)
            except RuntimeError:  # the loop closed meanwhile
                return

    def _dispatch(self, msg: Any) -> None:
        handler = self._handler
        if handler is None:
            return
        try:
            handler(msg)
        except Exception:  # one bad message must not stop the rest; counted
            self.dispatch_errors += 1

    def _log(self, msg: Any) -> None:
        with self._tlog_lock:
            if self._tlog is not None and self._tlog_clock is not None:
                usec = int(self._tlog_clock.utc_now().timestamp() * 1e6) & ~3
                self._tlog.write(struct.pack(">Q", usec) + msg.get_msgbuf())

    def start_tlog(self, path: Path, clock: Clock) -> None:
        """Record every received message in a telemetry log (``flight.tlog``).

        Same format as pymavlink's logs: an 8-byte big-endian UTC time in microseconds
        before each message, so ``mavutil.mavlink_connection(path)`` reads it back.
        """
        self.stop_tlog()
        with self._tlog_lock:
            self._tlog = path.open("wb")
            self._tlog_clock = clock

    def stop_tlog(self) -> None:
        with self._tlog_lock:
            if self._tlog is not None:
                self._tlog.close()
                self._tlog = None

    def close(self) -> None:
        self._stop.set()
        self._handler = None
        if self._reader is not None:
            self._reader.join(timeout=2 * self._POLL_S + self.READ_RETRY_S)
            self._reader = None
        self.stop_tlog()
        self._conn.close()


def connect(url: str, *, role: Role, source_system: int, source_component: int) -> MavConnection:
    """Open a MAVLink connection for ``role``; ``ValueError`` for a wrong identity."""
    return MavConnection(
        url, role=role, source_system=source_system, source_component=source_component
    )
