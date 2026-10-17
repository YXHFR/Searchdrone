"""The sd_quad14 model in flight (ArduCopter SITL, no wind).

One flight covers the first two checks, so the CI suite stays short:

* hover: take off to 20 m and hold for 60 s; the height stays within 1 m of 20 m and the
  vehicle drifts less than 2 m sideways;
* parity: climb to 30 m, hover 60 s, fly 300 m straight ahead at 15 m/s and land. The
  learned hover throttle, the height band while hovering, the ground speed on the
  straight leg and (builtin physics only) the hover time the battery current gives must
  fall inside the bands below;
* camera (Gazebo only): the nadir camera's stream gives at least 135 frames in 10 s.

The builtin-physics tests run in CI. The Gazebo variants need Gazebo Harmonic with
rendering and run on a workstation: ``just test-sitl physics=gz sc=gz``.
"""

from __future__ import annotations

import json
import math
import statistics
import time
from collections.abc import Callable
from dataclasses import dataclass, field

import pytest

from sd_safety.targets import Plane
from sd_sim import config, fixture, paths
from sd_sim.errors import InfraError
from sd_sim.harness import Harness, VehicleState
from sd_sim.launch import Stack
from sd_sim.report import Report
from sd_sim.smoke import count_frames
from sd_sim.util import new_run_dir

SITL = [
    pytest.mark.sitl,
    pytest.mark.slow,
    pytest.mark.skipif(not paths.arducopter_bin().is_file(), reason="ArduCopter SITL is not built"),
]

HOVER_ALT_M = 20.0
HOVER_S = 60.0
HOVER_ALT_TOLERANCE_M = 1.0
HOVER_DRIFT_M = 2.0
SETTLE_S = 10.0

PARITY_ALT_M = 30.0
LEG_M = 300.0
LEG_SPEED_MS = 15.0
CRUISE_FROM_M, CRUISE_TO_M = 100.0, 250.0  # the part of the leg at full speed
HOVER_THROTTLE_BAND = (0.35, 0.55)
HOVER_BAND_M = 1.0
SPEED_BAND_MS = (14.0, 15.5)
HOVER_TIME_MIN = (25.0, 40.0)
USABLE_CAPACITY = 0.85

CAMERA_FRAMES_10S = 135


def battery_ah() -> float:
    """The pack capacity the builtin model simulates."""
    frame = paths.REPO / "simulation" / "sitl" / "frames" / "sd_quad14.json"
    return float(json.loads(frame.read_text(encoding="utf-8"))["battCapacityAh"])


@dataclass
class Track:
    """Samples recorded from the harness while a phase runs."""

    alt: list[float] = field(default_factory=list)
    pos: list[tuple[float, float]] = field(default_factory=list)
    speed: list[float] = field(default_factory=list)
    current: list[float] = field(default_factory=list)

    def listener(self, kind: str, s: VehicleState) -> None:
        if kind == "GLOBAL_POSITION_INT":
            self.alt.append(s.rel_alt_m)
            self.pos.append((s.lat, s.lon))
            self.speed.append(s.ground_speed_ms)
        elif kind == "SYS_STATUS" and s.battery_current_a >= 0:
            self.current.append(s.battery_current_a)


def record(harness: Harness, seconds: float) -> Track:
    track = Track()
    harness.listeners.append(track.listener)
    try:
        harness.pump(seconds)
    finally:
        harness.listeners.remove(track.listener)
    return track


def take_off(harness: Harness, alt_m: float) -> None:
    harness.wait(lambda s: s.heartbeat, 60, "HEARTBEAT från flygkontrollen")
    harness.wait(lambda s: s.ready, 90, "position från EKF och GPS")
    harness.set_mode("GUIDED")
    harness.arm()
    harness.takeoff(alt_m)
    harness.wait(lambda s: s.rel_alt_m >= alt_m - 0.5, 90, f"höjden {alt_m:.0f} m")
    harness.pump(SETTLE_S)


def land(harness: Harness) -> None:
    harness.set_mode("LAND")
    harness.wait(lambda s: not s.armed, 120, "landning och desarmering")


def drift_m(track: Track) -> float:
    plane = Plane(*track.pos[0])
    return max(math.hypot(*plane.xy(lat, lon)) for lat, lon in track.pos)


def fly_scenario(
    name: str, physics: str, instance: int, brain: str, flight: Callable[[Harness, Report], None]
) -> Report:
    cfg = config.load(
        {
            "physics": physics,
            "gui": "false",
            "video": "off",
            "qgc": "false",
            "instance": str(instance),
        }
    )
    cfg.home = fixture.load().home.sitl_arg()
    cfg.personal_home = False
    cfg.vehicle = "sd_quad14"
    run_dir = new_run_dir(instance, name)
    rep = Report(name, run_dir, instance, physics, brain)
    stack = Stack(cfg, run_dir, observer=False)
    harness: Harness | None = None
    try:
        harness = Harness(stack.ports["harness_mavlink"])
        stack.up()
        flight(harness, rep)
    except InfraError as exc:
        rep.infra_failure = stack.mask(str(exc))
        raise
    except BaseException as exc:
        rep.error = stack.mask(f"{type(exc).__name__}: {exc}")
        raise
    finally:
        if harness is not None:
            harness.close()
        stack.down()
        rep.write()
    return rep


# --- hover and parity ---------------------------------------------------------------------


def check_hover(harness: Harness, rep: Report) -> None:
    track = record(harness, HOVER_S)
    worst = max(abs(a - HOVER_ALT_M) for a in track.alt)
    rep.check(
        "Q14.1",
        f"Höjden inom ±{HOVER_ALT_TOLERANCE_M:g} m från {HOVER_ALT_M:g} m under {HOVER_S:g} s",
        worst <= HOVER_ALT_TOLERANCE_M,
        measured=round(worst, 2),
        limit=HOVER_ALT_TOLERANCE_M,
        unit="m",
    )
    drift = drift_m(track)
    rep.check(
        "Q14.2",
        f"Horisontell drift under {HOVER_DRIFT_M:g} m",
        drift < HOVER_DRIFT_M,
        measured=round(drift, 2),
        limit=HOVER_DRIFT_M,
        unit="m",
    )


def climb(harness: Harness, alt_m: float) -> None:
    s = harness.state
    harness.goto(s.lat, s.lon, alt_m)
    harness.wait(lambda st: st.rel_alt_m >= alt_m - 0.5, 60, f"höjden {alt_m:.0f} m")
    harness.pump(SETTLE_S)


def parity_flight(physics: str) -> Callable[[Harness, Report], None]:
    """One flight for both checks: the 20 m hover first, then the parity flight at 30 m."""

    def flight(harness: Harness, rep: Report) -> None:
        take_off(harness, HOVER_ALT_M)
        check_hover(harness, rep)
        climb(harness, PARITY_ALT_M)
        hover = record(harness, HOVER_S)
        hover_throttle = harness.read_param("MOT_THST_HOVER")
        rep.check(
            "PAR.1",
            "MOT_THST_HOVER efter 60 s sväv",
            HOVER_THROTTLE_BAND[0] <= hover_throttle <= HOVER_THROTTLE_BAND[1],
            measured=round(hover_throttle, 3),
            limit=list(HOVER_THROTTLE_BAND),
        )
        band = max(hover.alt) - min(hover.alt)
        rep.check(
            "PAR.2",
            "Höjdavvikelse under svävet (max - min)",
            band <= HOVER_BAND_M,
            measured=round(band, 2),
            limit=HOVER_BAND_M,
            unit="m",
        )
        if physics == "builtin":
            mean_a = statistics.fmean(hover.current) if hover.current else 0.0
            minutes = USABLE_CAPACITY * battery_ah() / mean_a * 60 if mean_a > 0 else 0.0
            rep.check(
                "PAR.4",
                "Uppskattad svävtid (0,85 x kapacitet / medelström)",
                HOVER_TIME_MIN[0] <= minutes <= HOVER_TIME_MIN[1],
                measured={"minutes": round(minutes, 1), "mean_current_a": round(mean_a, 2)},
                limit=list(HOVER_TIME_MIN),
                unit="min",
            )
        else:
            rep.not_built("PAR.4", "Uppskattad svävtid", "bara med inbyggd fysik")

        start = Plane(*hover.pos[-1])
        lat1, lon1 = start.latlon(0.0, LEG_M)
        target = Plane(lat1, lon1)
        cruise: list[float] = []

        def on_message(kind: str, s: VehicleState) -> None:
            if kind == "GLOBAL_POSITION_INT":
                _, north = start.xy(s.lat, s.lon)
                if CRUISE_FROM_M <= north <= CRUISE_TO_M:
                    cruise.append(s.ground_speed_ms)

        harness.listeners.append(on_message)
        try:
            harness.goto(lat1, lon1, PARITY_ALT_M)
            # A new GUIDED target resets the speed to WP_SPD, so the speed comes after it.
            harness.pump(0.5)
            harness.set_speed(LEG_SPEED_MS)
            harness.wait(
                lambda s: math.hypot(*target.xy(s.lat, s.lon)) < 3.0,
                90,
                f"punkten {LEG_M:.0f} m fram",
            )
        finally:
            harness.listeners.remove(on_message)
        speed = statistics.median(cruise) if cruise else 0.0
        rep.check(
            "PAR.3",
            f"Markfart på raka sträckan ({CRUISE_FROM_M:.0f}-{CRUISE_TO_M:.0f} m)",
            SPEED_BAND_MS[0] <= speed <= SPEED_BAND_MS[1],
            measured={"median_ms": round(speed, 2), "samples": len(cruise)},
            limit=list(SPEED_BAND_MS),
            unit="m/s",
        )
        land(harness)

    return flight


def run_parity(physics: str, instance: int, brain: str) -> None:
    rep = fly_scenario("model_parity", physics, instance, brain, parity_flight(physics))
    assert rep.passed, "\n".join(rep.failures())


@pytest.mark.budget_s(330)
def test_model_parity(sim_instance: int, sim_brain: str) -> None:
    run_parity("builtin", sim_instance, sim_brain)


@pytest.mark.gz
@pytest.mark.budget_s(420)
def test_model_parity_gz(sim_instance: int, sim_brain: str) -> None:
    run_parity("gz", sim_instance, sim_brain)


# --- camera ------------------------------------------------------------------------------


@pytest.mark.gz
@pytest.mark.budget_s(180)
def test_quad14_camera_stream_gz(sim_instance: int, sim_brain: str) -> None:
    cfg = config.load(
        {
            "physics": "gz",
            "gui": "false",
            "video": "off",
            "qgc": "false",
            "instance": str(sim_instance),
        }
    )
    cfg.home = fixture.load().home.sitl_arg()
    cfg.personal_home = False
    run_dir = new_run_dir(sim_instance, "quad14_camera")
    rep = Report("quad14_camera", run_dir, sim_instance, "gz", sim_brain)
    stack = Stack(cfg, run_dir, observer=True)
    try:
        stack.up()
        assert stack.world is not None and stack.world.vehicle == "sd_quad14"
        started = time.monotonic()
        frames = count_frames(stack)
        rep.check(
            "CAM.1",
            "Kamerans ström ger minst 135 bildrutor på 10 s",
            frames >= CAMERA_FRAMES_10S,
            measured={"frames": frames, "wall_s": round(time.monotonic() - started, 1)},
            limit=CAMERA_FRAMES_10S,
        )
    finally:
        stack.down()
        rep.write()
    assert rep.passed, "\n".join(rep.failures())


pytestmark = SITL
