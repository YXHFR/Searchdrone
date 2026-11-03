"""The simulated laser pointing down (``rangefinder=tf02|grf250``), for SITL.

The real laser hangs under the drone and is wired to the flight controller, which
reports it to the onboard computer as MAVLink ``DISTANCE_SENSOR``
(``sd_onboard.rangefinder``). In the simulator the flight controller is SITL with its
MAVLink rangefinder driver (``RNGFND1_TYPE = 10``): this process sits on SITL's serial6
(``rangefinder_mavlink`` in ``simulation/ports.yaml``), asks for the vehicle's true
position and attitude (``SIM_STATE``, at the laser's rate), works out the distance
along the drone's down axis to the first thing the beam meets in the world
(``sd_sim.obstacles``: crowns, power lines, the ground) and sends it back as
``DISTANCE_SENSOR``. SITL then reports it to sd-onboard exactly as it would a real laser,
so the onboard computer reads the laser the way it does in the air. It works with
builtin physics and with Gazebo alike: the scene, not the renderer, is the truth.

Profiles (spec "Titta närmare" §16.2):

* ``tf02`` (Benewake TF02-Pro): 0.1-13.5 m over forest and dark ground, up to 40 m over
  light ground; noise ±5 cm under 5 m, else ±1 %; 100 Hz.
* ``grf250`` (LightWare GRF-250): 10-250 m, noise ±0.5 m, first return; 50 Hz.

A laser that sees nothing reports past its maximum (``max + 1 m``), as ArduPilot's
serial drivers do; below its minimum it reports what it measures and SITL marks it out
of range. Faults (``Faults``; scenario keys ``rf_*``): it stops answering
(``dropout``), sticks on one value (``stuck``), sends nonsense (``garbage``: zero, the
largest value, random values, "invalid" signal quality), sees nothing
(``out_of_range``), now and then far too short (``spikes``), sees nothing over water
(``water``) or sees through a gap in the crowns to what lies below (``gap_rate``). A
test can also switch a fault on at any moment by writing ``{"fault": "<name>"}`` (or
``{"fault": null}``) to the control file (``Stack.set_rangefinder_fault``).
"""

from __future__ import annotations

import argparse
import contextlib
import json
import math
import random
import signal
import socket
import time
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from types import FrameType
from typing import Any, Final, Literal

from pymavlink.dialects.v20 import ardupilotmega as mavlink

from sd_safety.targets import Plane
from sd_sim import ports
from sd_sim.obstacles import Scene

__all__ = [
    "FAULTS",
    "MODELS",
    "PROFILES",
    "Faults",
    "Laser",
    "Pose",
    "Profile",
    "Reading",
    "beam",
    "distance_sensor",
    "mav_channel",
    "param_layer",
    "write_param_layer",
]

Model = Literal["none", "tf02", "grf250"]
MODELS: Final[tuple[Model, ...]] = ("none", "tf02", "grf250")
FaultName = Literal["dropout", "stuck", "garbage", "out_of_range"]
FAULTS: Final[tuple[FaultName, ...]] = ("dropout", "stuck", "garbage", "out_of_range")
ORIENT_DOWN: Final = mavlink.MAV_SENSOR_ROTATION_PITCH_270  # RNGFND1_ORIENT 25
# The laser's own component on the flight controller's system: ArduPilot takes MAVLink
# from its own system id whatever the GCS id settings say.
COMPID: Final = mavlink.MAV_COMP_ID_PERIPHERAL
SERIAL: Final = 6  # SITL's serial port for the laser (0-2 and 5 are taken, 3-4 are GPS)
QUALITY_GOOD: Final = 100
QUALITY_NONE: Final = 0  # "unknown": what a laser that sees nothing reports
MAX_CM: Final = 65535
RESEND_S: Final = 2.0  # ask for SIM_STATE again when none came for this long
LOG_PERIOD_S: Final = 0.1


@dataclass(frozen=True)
class Profile:
    """A laser's range and noise. ``max_dark_m`` over forest and dark ground,
    ``max_light_m`` over light ground. Noise is ±``noise_m`` below ``noise_rel_from_m``
    and ±``noise_rel`` of the distance from there."""

    name: str
    min_m: float
    max_dark_m: float
    max_light_m: float
    noise_m: float
    noise_rel: float
    noise_rel_from_m: float
    rate_hz: float

    @property
    def no_return_m(self) -> float:
        """What it reports when it sees nothing: past its maximum."""
        return self.max_light_m + 1.0

    @property
    def tested_m(self) -> tuple[float, float]:
        """``[rangefinder] min_tested_m`` / ``max_tested_m`` for sd-onboard."""
        return self.min_m, self.max_light_m


PROFILES: Final[dict[str, Profile]] = {
    "tf02": Profile("tf02", 0.1, 13.5, 40.0, 0.05, 0.01, 5.0, 100.0),
    "grf250": Profile("grf250", 10.0, 250.0, 250.0, 0.5, 0.0, math.inf, 50.0),
}


@dataclass(frozen=True)
class Faults:
    """The faults of a run. ``*_at_s`` start at that many seconds after the laser first
    saw the vehicle (``None``: never) and last to the end; ``spikes`` and ``gap_rate``
    are the share of readings affected."""

    dropout_at_s: float | None = None
    stuck_at_s: float | None = None
    garbage_at_s: float | None = None
    out_of_range_at_s: float | None = None
    spikes: float = 0.0
    water: bool = False
    gap_rate: float = 0.0

    def __post_init__(self) -> None:
        if not (0.0 <= self.spikes <= 1.0 and 0.0 <= self.gap_rate <= 1.0):
            raise ValueError("rf_spikes och rf_gap_rate ska ligga mellan 0 och 1")

    @classmethod
    def from_mapping(cls, data: Mapping[str, Any]) -> Faults:
        """From a scenario's keys: ``rf_dropout_at_s``, ``rf_stuck_at_s``,
        ``rf_garbage_at_s``, ``rf_out_of_range_at_s``, ``rf_spikes``, ``rf_water``,
        ``rf_gap_rate``."""
        known = {f"rf_{name}" for name in cls.__dataclass_fields__}
        unknown = sorted(set(data) - known)
        if unknown:
            raise ValueError(f"okända laserfel: {', '.join(unknown)}")

        def at(key: str) -> float | None:
            value = data.get(f"rf_{key}")
            return None if value is None else float(value)

        return cls(
            dropout_at_s=at("dropout_at_s"),
            stuck_at_s=at("stuck_at_s"),
            garbage_at_s=at("garbage_at_s"),
            out_of_range_at_s=at("out_of_range_at_s"),
            spikes=float(data.get("rf_spikes", 0.0)),
            water=bool(data.get("rf_water", False)),
            gap_rate=float(data.get("rf_gap_rate", 0.0)),
        )

    def started(self, name: FaultName, t: float) -> bool:
        at = getattr(self, f"{name}_at_s")
        return at is not None and t >= at


@dataclass(frozen=True)
class Pose:
    """The vehicle in the world: metres east, north and up from home on the ground, and
    its attitude (radians, ArduPilot's: roll right, pitch nose up, yaw from north)."""

    east: float
    north: float
    z: float
    roll: float = 0.0
    pitch: float = 0.0
    yaw: float = 0.0


@dataclass(frozen=True)
class Reading:
    """One measurement as the laser sends it. ``kind``: what it hit (``ground``,
    ``tree``, ``line``), ``none`` (nothing in range), or the fault that made it."""

    distance_m: float
    quality: int
    kind: str


def beam(pose: Pose) -> tuple[float, float, float]:
    """The drone's down axis in the world (east, north, up): where the laser points."""
    cr, sr = math.cos(pose.roll), math.sin(pose.roll)
    cp, sp = math.cos(pose.pitch), math.sin(pose.pitch)
    cy, sy = math.cos(pose.yaw), math.sin(pose.yaw)
    north = cy * sp * cr + sy * sr
    east = sy * sp * cr - cy * sr
    down = cp * cr
    return east, north, -down


class Laser:
    """The laser's readings in a scene, with its profile's range and noise and the
    run's faults. Deterministic for a seed."""

    def __init__(
        self, scene: Scene, profile: Profile, faults: Faults | None = None, seed: int = 0
    ) -> None:
        self.scene, self.profile = scene, profile
        self.faults = Faults() if faults is None else faults
        self._rng = random.Random(seed)  # noqa: S311 - simulated noise, not security
        self._now: dict[FaultName, float] = {}
        self._last: Reading | None = None
        self._stuck: Reading | None = None

    def set_fault(self, name: FaultName | None, t: float) -> None:
        """Switch a fault on from ``t`` on (``None``: the ones switched on here end)."""
        self._now.clear()
        self._stuck = None
        if name is not None:
            self._now[name] = t

    def active(self, name: FaultName, t: float) -> bool:
        return self.faults.started(name, t) or (name in self._now and t >= self._now[name])

    def read(self, pose: Pose, t: float) -> Reading | None:
        """The reading at ``t`` (seconds since the laser first saw the vehicle), or
        ``None`` when the laser says nothing."""
        if self.active("dropout", t):
            return None
        if self.active("garbage", t):
            return self._garbage()
        if self.active("stuck", t) and self._last is not None:
            if self._stuck is None:
                self._stuck = self._last
            return Reading(self._stuck.distance_m, self._stuck.quality, "stuck")
        reading = self._measure(pose, t)
        self._last = reading
        return reading

    def _garbage(self) -> Reading:
        rng = self._rng
        distance = rng.choice((0.0, MAX_CM / 100, rng.uniform(0.0, MAX_CM / 100)))
        return Reading(distance, rng.choice((0, 1, 1, 37, QUALITY_GOOD)), "garbage")

    def _measure(self, pose: Pose, t: float) -> Reading:
        p = self.profile
        nothing = Reading(p.no_return_m, QUALITY_NONE, "none")
        if self.active("out_of_range", t):
            return nothing
        gap = self._rng.random() < self.faults.gap_rate
        spike = self._rng.random() < self.faults.spikes
        noise = self._rng.uniform(-1.0, 1.0)
        direction = beam(pose)
        start = (pose.east, pose.north, pose.z)
        hit = self.scene.ray(start, direction, p.max_light_m + 1.0, crowns=not gap)
        if hit is None:
            return nothing
        x = pose.east + direction[0] * hit.distance
        y = pose.north + direction[1] * hit.distance
        light = hit.kind == "ground" and self.scene.is_light(x, y)
        if hit.distance > (p.max_light_m if light else p.max_dark_m):
            return nothing
        if self.faults.water and hit.kind == "ground" and self.scene.is_water(x, y):
            return nothing
        spread = p.noise_m if hit.distance < p.noise_rel_from_m else p.noise_rel * hit.distance
        distance = max(hit.distance + noise * spread, 0.0)
        if spike:
            return Reading(distance * self._rng.uniform(0.2, 0.7), QUALITY_GOOD, "spike")
        return Reading(distance, QUALITY_GOOD, hit.kind)


def distance_sensor(reading: Reading, profile: Profile, boot_ms: int) -> Any:
    """The reading as ArduPilot's MAVLink driver takes it (centimetres)."""
    return mavlink.MAVLink_distance_sensor_message(
        boot_ms & 0xFFFFFFFF,
        round(profile.min_m * 100),
        min(round(profile.max_light_m * 100), MAX_CM),
        min(max(round(reading.distance_m * 100), 0), MAX_CM),
        mavlink.MAV_DISTANCE_SENSOR_LASER,
        0,
        ORIENT_DOWN,
        0,
        signal_quality=reading.quality,
    )


# --- the flight controller's side ---------------------------------------------------------


def mav_channel(qgc: bool) -> int:
    """SITL's MAVLink channel (``MAVn_*``) for serial6: channels go to the MAVLink serial
    ports in order: serial0-2, then serial5 when QGroundControl is on."""
    return 5 if qgc else 4


def param_layer(model: str, qgc: bool) -> str:
    """The SITL parameter layer for a laser: the MAVLink driver on serial6, pointing
    down, never flown by (as ``searchdrone-rangefinder-*.parm`` on the real drone), and
    the laser's channel kept to itself (no forwarding to the other links)."""
    p = PROFILES[model]
    return (
        f"# Simulated laser {p.name} (sd_sim.rangefinder), written by the launcher.\n"
        f"SERIAL{SERIAL}_PROTOCOL 2\n"
        f"MAV{mav_channel(qgc)}_OPTIONS 2\n"
        "RNGFND1_TYPE 10\n"
        f"RNGFND1_ORIENT {ORIENT_DOWN}\n"
        f"RNGFND1_MIN {p.min_m:g}\n"
        f"RNGFND1_MAX {p.max_light_m:g}\n"
        "WP_RFND_USE 0\n"
        "SURFTRAK_MODE 0\n"
        "EK3_RNG_USE_HGT -1\n"
    )


def write_param_layer(directory: Path, model: str, qgc: bool) -> Path:
    directory.mkdir(parents=True, exist_ok=True)
    path = directory / "rangefinder.parm"
    path.write_text(param_layer(model, qgc), encoding="utf-8")
    return path


def _command(fc: tuple[int, int], msg_id: int, rate_hz: float) -> Any:
    return mavlink.MAVLink_command_long_message(
        fc[0],
        fc[1],
        mavlink.MAV_CMD_SET_MESSAGE_INTERVAL,
        0,
        float(msg_id),
        1e6 / rate_hz,
        0,
        0,
        0,
        0,
        0,
    )


class Service:
    """The laser on one UDP port: SITL's serial6 sends to it, it answers to SITL."""

    def __init__(
        self,
        laser: Laser,
        plane: Plane,
        home_alt: float,
        sock: socket.socket,
        *,
        control: Path | None = None,
        log: Path | None = None,
    ) -> None:
        self.laser, self.plane, self.home_alt, self.sock = laser, plane, home_alt, sock
        self.control, self.log = control, log
        self._parser = mavlink.MAVLink(None)
        self._mav: Any = None
        self._peer: tuple[str, int] | None = None
        self._fc: tuple[int, int] | None = None
        self._t0: float | None = None
        self._last_state = -math.inf
        self._last_ask = -math.inf
        self._control_mtime = 0.0
        self._last_log = -math.inf
        self.sent = 0
        self.stop = False

    def _now(self) -> float:
        return time.monotonic()

    def run(self) -> None:
        self.sock.settimeout(0.05)
        out = self.log.open("a", encoding="utf-8") if self.log is not None else None
        try:
            while not self.stop:
                try:
                    data, addr = self.sock.recvfrom(65535)
                except TimeoutError:
                    data, addr = b"", None
                except OSError:
                    data, addr = b"", None
                if addr is not None:
                    self._peer = addr
                for msg in self._parse(data):
                    self.handle(msg, out)
                self._ask()
                self._poll_control()
        finally:
            if out is not None:
                out.close()

    def _parse(self, data: bytes) -> list[Any]:
        if not data:
            return []
        try:
            return list(self._parser.parse_buffer(data) or [])
        except mavlink.MAVError:
            return []

    def handle(self, msg: Any, out: Any = None) -> None:
        kind = msg.get_type()
        src = (msg.get_srcSystem(), msg.get_srcComponent())
        if kind == "HEARTBEAT" and src[1] == mavlink.MAV_COMP_ID_AUTOPILOT1 and self._fc != src:
            self._fc = src
            self._mav = mavlink.MAVLink(None, srcSystem=src[0], srcComponent=COMPID)
            self._last_ask = -math.inf
        if kind != "SIM_STATE" or self._mav is None:
            return
        now = self._now()
        self._last_state = now
        if self._t0 is None:
            self._t0 = now
        t = now - self._t0
        east, north = self.plane.xy(msg.lat_int / 1e7, msg.lon_int / 1e7)
        pose = Pose(east, north, float(msg.alt) - self.home_alt, msg.roll, msg.pitch, msg.yaw)
        reading = self.laser.read(pose, t)
        if reading is not None:
            self._send(distance_sensor(reading, self.laser.profile, round(t * 1000)))
            self.sent += 1
        if out is not None and now - self._last_log >= LOG_PERIOD_S:
            self._last_log = now
            got = ",silent" if reading is None else f"{reading.distance_m:.3f},{reading.kind}"
            out.write(f"{t:.2f},{east:.2f},{north:.2f},{pose.z:.2f},{got}\n")

    def _send(self, msg: Any) -> None:
        if self._peer is None or self._mav is None:
            return
        with contextlib.suppress(OSError):
            self.sock.sendto(msg.pack(self._mav), self._peer)

    def _ask(self) -> None:
        """Ask SITL for SIM_STATE at the laser's rate while none is coming."""
        now = self._now()
        if self._fc is None or now - self._last_state < RESEND_S / 2:
            return
        if now - self._last_ask < RESEND_S:
            return
        self._last_ask = now
        self._send(_command(self._fc, mavlink.MAVLINK_MSG_ID_SIM_STATE, self.laser.profile.rate_hz))

    def _poll_control(self) -> None:
        if self.control is None:
            return
        try:
            mtime = self.control.stat().st_mtime
        except OSError:
            return
        if mtime == self._control_mtime:
            return
        self._control_mtime = mtime
        try:
            name = json.loads(self.control.read_text(encoding="utf-8")).get("fault")
        except (OSError, ValueError, AttributeError):
            return
        if name is None or name in FAULTS:
            t = 0.0 if self._t0 is None else self._now() - self._t0
            self.laser.set_fault(name, t)


def main(argv: Sequence[str] | None = None) -> int:
    ap = argparse.ArgumentParser(prog="python -m sd_sim.rangefinder", description=__doc__)
    ap.add_argument("--instance", type=int, default=0)
    ap.add_argument("--model", choices=sorted(PROFILES), required=True)
    ap.add_argument("--scene", type=Path, required=True, help="mappen med scene.json")
    ap.add_argument("--faults", default="{}", help="JSON med rf_*-nycklar")
    ap.add_argument("--control", type=Path, default=None)
    ap.add_argument("--log", type=Path, default=None)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args(argv)
    scene = Scene.load(args.scene)
    laser = Laser(
        scene, PROFILES[args.model], Faults.from_mapping(json.loads(args.faults)), args.seed
    )
    sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    sock.bind(("127.0.0.1", ports.for_instance(args.instance)["rangefinder_mavlink"]))
    service = Service(
        laser, Plane(*scene.origin), scene.home_alt, sock, control=args.control, log=args.log
    )

    def _stop(_sig: int, _frame: FrameType | None) -> None:
        service.stop = True

    signal.signal(signal.SIGTERM, _stop)
    signal.signal(signal.SIGINT, _stop)
    try:
        service.run()
    finally:
        sock.close()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
