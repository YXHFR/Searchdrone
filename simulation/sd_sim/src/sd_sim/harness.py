"""The test rig's own MAVLink channel to the flight controller (SITL serial0).

The harness is MAVLink system 253, component 190. Its HEARTBEAT is a GCS heartbeat but
does not keep the GCS failsafe alive, because the flight controller only counts
system ``MAV_GCS_SYSID`` (250, the ground station). Everything that writes to the
flight controller from test code lives here.

``python -m sd_sim.harness observe --instance N --state-file F`` runs it as an
observer for the launcher (it requests telemetry streams and sends its heartbeat, but
no flight commands): it writes the vehicle state as JSON twice a second so the
launcher can wait for a healthy vehicle.
"""

from __future__ import annotations

import argparse
import json
import os
import signal
import sys
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

from pymavlink import mavutil

from sd_sim import ports
from sd_sim.errors import InfraError

HARNESS_SYSID = 253
HARNESS_COMPID = 190
FC_SYSID = 1
FC_COMPID = 1
EKF_PRED_POS_HORIZ_ABS = 512
GPS_FIX_3D = 3
STREAM_HZ = 5
_STREAMS = ("GLOBAL_POSITION_INT", "EKF_STATUS_REPORT", "GPS_RAW_INT", "SYS_STATUS")

mavlink: Any = mavutil.mavlink


@dataclass
class VehicleState:
    heartbeat: bool = False
    mode: str = ""
    armed: bool = False
    ekf_pos_ok: bool = False
    gps_fix: int = 0
    rel_alt_m: float = 0.0
    max_rel_alt_m: float = 0.0

    @property
    def ready(self) -> bool:
        return self.heartbeat and self.ekf_pos_ok and self.gps_fix >= GPS_FIX_3D


class Harness:
    def __init__(self, port: int) -> None:
        self.port = port
        self.state = VehicleState()
        self._conn: Any = mavutil.mavlink_connection(
            f"udpin:127.0.0.1:{port}",
            source_system=HARNESS_SYSID,
            source_component=HARNESS_COMPID,
            dialect="ardupilotmega",
        )
        self._next_heartbeat = 0.0
        self._streams_requested = False

    @classmethod
    def for_instance(cls, instance: int) -> Harness:
        return cls(ports.for_instance(instance)["harness_mavlink"])

    def close(self) -> None:
        self._conn.close()

    # --- receiving -----------------------------------------------------------------

    def pump(self, seconds: float) -> None:
        """Read and apply messages for ``seconds``, sending our heartbeat once a second."""
        end = time.monotonic() + seconds
        while True:
            now = time.monotonic()
            if now >= self._next_heartbeat:
                self._send_heartbeat()
                self._next_heartbeat = now + 1.0
            remaining = end - now
            if remaining <= 0:
                return
            msg = self._conn.recv_match(blocking=True, timeout=min(remaining, 0.2))
            if msg is not None:
                self._apply(msg)

    def _apply(self, msg: Any) -> None:
        if msg.get_srcSystem() != FC_SYSID or msg.get_srcComponent() != FC_COMPID:
            return
        kind = msg.get_type()
        s = self.state
        if kind == "HEARTBEAT" and msg.type != mavlink.MAV_TYPE_GCS:
            s.heartbeat = True
            s.armed = bool(msg.base_mode & mavlink.MAV_MODE_FLAG_SAFETY_ARMED)
            s.mode = mavutil.mode_string_v10(msg)
            if not self._streams_requested:
                self._request_streams()
        elif kind == "EKF_STATUS_REPORT":
            s.ekf_pos_ok = bool(msg.flags & EKF_PRED_POS_HORIZ_ABS)
        elif kind == "GPS_RAW_INT":
            s.gps_fix = int(msg.fix_type)
        elif kind == "GLOBAL_POSITION_INT":
            s.rel_alt_m = msg.relative_alt / 1000.0
            s.max_rel_alt_m = max(s.max_rel_alt_m, s.rel_alt_m)

    def wait(self, predicate: Any, timeout: float, what: str) -> None:
        end = time.monotonic() + timeout
        while not predicate(self.state):
            if time.monotonic() > end:
                raise InfraError(f"tidsgränsen {timeout:.0f} s gick ut: {what}")
            self.pump(0.2)

    # --- sending -------------------------------------------------------------------

    def _send_heartbeat(self) -> None:
        self._conn.mav.heartbeat_send(
            mavlink.MAV_TYPE_GCS, mavlink.MAV_AUTOPILOT_INVALID, 0, 0, mavlink.MAV_STATE_ACTIVE
        )

    def _request_streams(self) -> None:
        for name in _STREAMS:
            msg_id = getattr(mavlink, f"MAVLINK_MSG_ID_{name}")
            self._command(
                mavlink.MAV_CMD_SET_MESSAGE_INTERVAL, msg_id, 1_000_000 / STREAM_HZ, wait_ack=False
            )
        self._streams_requested = True

    def _command(self, command: int, *params: float, wait_ack: bool = True) -> None:
        values = [*params, 0, 0, 0, 0, 0, 0, 0][:7]
        self._conn.mav.command_long_send(FC_SYSID, FC_COMPID, command, 0, *values)
        if not wait_ack:
            return
        end = time.monotonic() + 5.0
        while time.monotonic() < end:
            msg = self._conn.recv_match(blocking=True, timeout=0.2)
            if msg is None:
                continue
            self._apply(msg)
            if msg.get_type() == "COMMAND_ACK" and msg.command == command:
                if msg.result != mavlink.MAV_RESULT_ACCEPTED:
                    raise InfraError(f"kommandot {command} avvisades (resultat {msg.result})")
                return
        raise InfraError(f"inget svar på kommandot {command}")

    def set_mode(self, name: str, timeout: float = 10.0) -> None:
        mapping = self._conn.mode_mapping() or {}
        if name not in mapping:
            raise InfraError(f"okänt läge {name}")
        self._command(
            mavlink.MAV_CMD_DO_SET_MODE, mavlink.MAV_MODE_FLAG_CUSTOM_MODE_ENABLED, mapping[name]
        )
        self.wait(lambda s: s.mode == name, timeout, f"läget {name}")

    def arm(self, timeout: float = 30.0) -> None:
        end = time.monotonic() + timeout
        while True:
            try:
                self._command(mavlink.MAV_CMD_COMPONENT_ARM_DISARM, 1)
                break
            except InfraError:
                # Pre-arm checks can still be settling right after the EKF is ready.
                if time.monotonic() > end:
                    raise
                self.pump(1.0)
        self.wait(lambda s: s.armed, 10.0, "armering")

    def takeoff(self, alt_m: float) -> None:
        self._command(mavlink.MAV_CMD_NAV_TAKEOFF, 0, 0, 0, 0, 0, 0, alt_m)


def observe(instance: int, state_file: Path) -> int:
    """Observer for the launcher (no flight commands); writes the state file until SIGTERM."""
    harness = Harness.for_instance(instance)
    stopping = False

    def stop(_sig: int, _frame: object) -> None:
        nonlocal stopping
        stopping = True

    signal.signal(signal.SIGTERM, stop)
    signal.signal(signal.SIGINT, stop)
    tmp = state_file.with_suffix(".tmp")
    while not stopping:
        harness.pump(0.5)
        payload = {**asdict(harness.state), "ready": harness.state.ready, "pid": os.getpid()}
        tmp.write_text(json.dumps(payload), encoding="utf-8")
        tmp.replace(state_file)
    harness.close()
    return 0


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="sd_sim.harness")
    sub = parser.add_subparsers(dest="command", required=True)
    obs = sub.add_parser("observe")
    obs.add_argument("--instance", type=int, default=0)
    obs.add_argument("--state-file", type=Path, required=True)
    args = parser.parse_args(argv)
    return observe(args.instance, args.state_file)


if __name__ == "__main__":
    sys.exit(main())
