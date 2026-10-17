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
import math
import os
import signal
import sys
import time
from collections.abc import Callable
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

from pymavlink import mavutil

from sd_sim import ports
from sd_sim.errors import InfraError, VehicleError

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
    lat: float = 0.0
    lon: float = 0.0
    ground_speed_ms: float = 0.0
    battery_current_a: float = -1.0  # -1: not reported yet
    battery_pct: int = -1  # SYS_STATUS battery_remaining; -1: not reported yet

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
        # Called with (message type, state) after every flight-controller message.
        self.listeners: list[Callable[[str, VehicleState], None]] = []

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
            self._heartbeat_tick()
            remaining = end - time.monotonic()
            if remaining <= 0:
                return
            msg = self._conn.recv_match(blocking=True, timeout=min(remaining, 0.2))
            if msg is not None:
                self._apply(msg)

    def _heartbeat_tick(self) -> None:
        now = time.monotonic()
        if now >= self._next_heartbeat:
            self._send_heartbeat()
            self._next_heartbeat = now + 1.0

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
            s.lat, s.lon = msg.lat / 1e7, msg.lon / 1e7
            s.ground_speed_ms = math.hypot(msg.vx, msg.vy) / 100.0
        elif kind == "SYS_STATUS":
            if msg.current_battery >= 0:
                s.battery_current_a = msg.current_battery / 100.0
            s.battery_pct = int(msg.battery_remaining)
        for listener in self.listeners:
            listener(kind, s)

    def wait(self, predicate: Any, timeout: float, what: str) -> None:
        end = time.monotonic() + timeout
        while not predicate(self.state):
            if time.monotonic() > end:
                raise VehicleError(f"tidsgränsen {timeout:.0f} s gick ut: {what}")
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
                    raise VehicleError(f"kommandot {command} avvisades (resultat {msg.result})")
                return
        raise VehicleError(f"inget svar på kommandot {command}")

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
            except VehicleError:
                # Pre-arm checks can still be settling right after the EKF is ready.
                if time.monotonic() > end:
                    raise
                self.pump(1.0)
        self.wait(lambda s: s.armed, 10.0, "armering")

    def takeoff(self, alt_m: float) -> None:
        self._command(mavlink.MAV_CMD_NAV_TAKEOFF, 0, 0, 0, 0, 0, 0, alt_m)

    def set_speed(self, speed_ms: float) -> None:
        """Horizontal speed for GUIDED and AUTO legs (MAV_CMD_DO_CHANGE_SPEED, ground speed)."""
        self._command(mavlink.MAV_CMD_DO_CHANGE_SPEED, 1, speed_ms, -1)

    def goto(self, lat: float, lon: float, rel_alt_m: float) -> None:
        """Fly to a point in GUIDED (position target, height above home)."""
        mask = (
            mavlink.POSITION_TARGET_TYPEMASK_VX_IGNORE
            | mavlink.POSITION_TARGET_TYPEMASK_VY_IGNORE
            | mavlink.POSITION_TARGET_TYPEMASK_VZ_IGNORE
            | mavlink.POSITION_TARGET_TYPEMASK_AX_IGNORE
            | mavlink.POSITION_TARGET_TYPEMASK_AY_IGNORE
            | mavlink.POSITION_TARGET_TYPEMASK_AZ_IGNORE
            | mavlink.POSITION_TARGET_TYPEMASK_YAW_IGNORE
            | mavlink.POSITION_TARGET_TYPEMASK_YAW_RATE_IGNORE
        )
        self._conn.mav.set_position_target_global_int_send(
            0,
            FC_SYSID,
            FC_COMPID,
            mavlink.MAV_FRAME_GLOBAL_RELATIVE_ALT_INT,
            mask,
            round(lat * 1e7),
            round(lon * 1e7),
            rel_alt_m,
            0,
            0,
            0,
            0,
            0,
            0,
            0,
            0,
        )

    # --- fault injection (the only writes to the flight controller from test code) ---

    def set_sim_param(self, name: str, value: float, timeout: float = 5.0) -> None:
        """``PARAM_SET`` for a simulator parameter (``SIM_*`` only), e.g. ``SIM_RC_FAIL``
        or the wind. Waits until the flight controller reports the new value."""
        if not name.startswith("SIM_"):
            raise InfraError(f"harnessen får bara ändra SIM_*-parametrar, inte {name}")
        self._conn.mav.param_set_send(
            FC_SYSID, FC_COMPID, name.encode(), float(value), mavlink.MAV_PARAM_TYPE_REAL32
        )
        end = time.monotonic() + timeout
        while time.monotonic() < end:
            if abs(self.read_param(name, timeout=max(0.5, end - time.monotonic())) - value) < 1e-3:
                return
        raise InfraError(f"{name} blev inte {value:g}")

    def battery_set_remaining(self, pct: float) -> None:
        """``MAV_CMD_BATTERY_RESET`` (battery mask 1): the pack shows ``pct`` % left, the
        capacity stays, so the flight controller's mAh thresholds work as with a real pack.

        ArduCopter 4.7.1 takes the new failsafe level at the reset without acting on it
        (``reset_remaining`` sets the state directly), so a test sets a level just above a
        threshold and lets the pack cross it, as a real one does."""
        try:
            self._command(mavlink.MAV_CMD_BATTERY_RESET, 1, pct)
        except VehicleError as exc:
            raise InfraError(f"batteriet gick inte att ställa till {pct:g} %: {exc}") from exc

    def guided_velocity(
        self, vx: float, vy: float, vz: float, duration_s: float, stop: Callable[[], bool]
    ) -> None:
        """GUIDED and a velocity target (north, east, down m/s) twice a second for
        ``duration_s``. Stops at once when the flight controller leaves GUIDED (a failsafe
        took over) or ``stop()`` says so. Only for the fence test: it flies past the gate."""
        self.set_mode("GUIDED")
        mask = 0x0DC7  # velocity only
        end = time.monotonic() + duration_s
        while time.monotonic() < end and self.state.mode == "GUIDED" and not stop():
            self._conn.mav.set_position_target_local_ned_send(
                0, FC_SYSID, FC_COMPID, mavlink.MAV_FRAME_LOCAL_NED, mask,
                0, 0, 0, vx, vy, vz, 0, 0, 0, 0, 0,
            )  # fmt: skip
            self.pump(0.5)

    def read_param(self, name: str, timeout: float = 5.0) -> float:
        """Read one parameter (read only, nothing is set)."""
        end = time.monotonic() + timeout
        while time.monotonic() < end:
            self._conn.mav.param_request_read_send(FC_SYSID, FC_COMPID, name.encode(), -1)
            stop = min(end, time.monotonic() + 1.0)
            while time.monotonic() < stop:
                self._heartbeat_tick()
                msg = self._conn.recv_match(blocking=True, timeout=0.2)
                if msg is None:
                    continue
                self._apply(msg)
                if msg.get_type() == "PARAM_VALUE" and msg.get_srcSystem() == FC_SYSID:
                    got = msg.param_id
                    if isinstance(got, bytes):
                        got = got.decode("ascii", errors="replace")
                    if str(got).rstrip("\x00") == name:
                        return float(msg.param_value)
        raise InfraError(f"parametern {name} lästes inte inom {timeout:.0f} s")

    def safety_violations(self, layer: dict[str, float] | None = None) -> list[str]:
        """Read every parameter that ``sd_sim.params.SAFETY_RULES`` names and return the
        rules the flight controller breaks (read only). Every SITL flight test calls it
        before it flies: a weakened safety parameter fails the test, it is never retried.
        ``layer`` is a test's own parameter layer: its values are then the exact rule."""
        from sd_sim.params import SAFETY_RULES, rule_violations

        own = layer or {}
        rules = [(n, "==", own[n]) if n in own else (n, op, v) for n, op, v in SAFETY_RULES]
        actual = {name: self.read_param(name) for name, _, _ in rules}
        return rule_violations(actual, rules)

    def read_all_params(self, timeout: float = 60.0) -> dict[str, float]:
        """Read the flight controller's whole parameter list (read only, nothing is set).

        Sends PARAM_REQUEST_LIST, then asks again for any index that did not arrive.
        """
        values: dict[int, tuple[str, float]] = {}
        total: int | None = None
        end = time.monotonic() + timeout
        self._conn.mav.param_request_list_send(FC_SYSID, FC_COMPID)
        quiet_since = time.monotonic()
        while time.monotonic() < end:
            self._heartbeat_tick()
            msg = self._conn.recv_match(blocking=True, timeout=0.2)
            if msg is None:
                if total is not None and time.monotonic() - quiet_since > 1.0:
                    for index in [i for i in range(total) if i not in values][:20]:
                        self._conn.mav.param_request_read_send(FC_SYSID, FC_COMPID, b"", index)
                    quiet_since = time.monotonic()
                continue
            self._apply(msg)
            if msg.get_type() != "PARAM_VALUE" or msg.get_srcSystem() != FC_SYSID:
                continue
            quiet_since = time.monotonic()
            total = int(msg.param_count)
            if not 0 <= int(msg.param_index) < total:
                continue
            name = msg.param_id
            if isinstance(name, bytes):
                name = name.decode("ascii", errors="replace")
            values[int(msg.param_index)] = (str(name).rstrip("\x00"), float(msg.param_value))
            if len(values) >= total:
                return dict(values.values())
        got = f"{len(values)} av {total}" if total is not None else "inga"
        raise InfraError(f"parameterlistan blev inte komplett inom {timeout:.0f} s ({got})")


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
