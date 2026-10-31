"""The sd_quad14 model in wind (ArduCopter SITL's own wind, builtin physics).

Uncalibrated: the model's drag comes from the frame file (``refSpd``/``refAngle``), not
from a measured flight, so these tests show how the parameters and the reserve behave
in the simulator, not what the real drone can take. Every report carries
``variant="okalibrerad"`` and every criterion id starts with ``W``.

The wind is the same at every height (``SIM_WIND_T 1``). ``SIM_WIND_DIR`` is the
direction the wind comes from (SIM_Aircraft.cpp turns the vector round), so 180 blows
the drone north.

* W1 hold in 10 m/s: hold 60 s at 30 m, then 60 s more with turbulence (``SIM_WIND_TURB``
  2); 95 % of the time within 3 m of the hold point, the height within 1.5 m.
* W2 home against 12 m/s: fly 500 m north with the wind behind, then RTL against it;
  lands within 5 m of home, the way home (until 5 m from home, before the descent) takes
  at most 1.5 x the model's time and uses at most the model's mAh + 20 %, and at least
  ``BATT_LOW_MAH`` is left after the landing.
* W3 gust to 15 m/s: a straight survey leg across 8 m/s; in the middle 15 m/s for 10 s
  (``SIM_WIND_TC`` 1). Off the line by less than 5 m, the height within 3 m, the EKF's
  position estimate never lost and no EKF message.
* W4 battery margin: 500 m out, the pack just above the flight controller's 15 % and
  10 m/s against the way home; SmartRTL/RTL within 5 s of the low-battery message, lands
  within 5 m of home, the critical level (LAND) never triggers.
* W5 the flight controller's own lean limit: ``sd_fpv7`` (``ATC_ANGLE_MAX`` 35 in its
  sim layer, the constant is 30); the ground station's ``ThreadSafeMav`` reads it on
  connect and the tilt warning is at 35 - 10 degrees, with no parameter on the
  constants.
* W6 ENERGY_HOME flies home: 150 m out, the ground station's own ``GroundMav`` and
  ``VehicleActions`` on 250/190, the pack set to 14 % (under ``BATT_LOW_MAH``, which the
  flight controller takes without acting on it at the reset, over the critical 7 %). The
  estimate with the flight controller's own numbers is short, so the ground station flies
  home: SmartRTL/RTL, the event's reason is ``energy_home``, Resume is refused with
  ``BATTERY_RTL_ACTIVE``; after STOP and "Ta över" the drone stays in LOITER while the
  estimate is still short (the energy latch under the STOP blocks only its own trigger).
* Nightly sweep: W2 with 0-18 m/s in steps of 2; red only if the drone does not get home
  in 12 m/s or less. Each run's report gives the speed, so the summary shows the
  strongest wind it still got home in.

The model the limits come from: the frame leans ``refAngle`` at ``refSpd`` airspeed, and
the position controller leans at most ``ATC_ANGLE_MAX``; both are 30 degrees, so the
fastest the model flies through the air is ``refSpd``. RTL keeps ``RTL_SPEED_MS`` over
the ground when it can, so home takes ``distance / min(RTL_SPEED_MS, refSpd - wind)``.
The mAh model is that time at the current measured while the drone held position in the
same wind. (The ground station's ``ENERGY_HOME`` takes the measured WIND's headwind and
crosswind on the way home, the drag it measured while holding position, and the
flight controller's own ``ATC_ANGLE_MAX`` and ``RTL_SPEED_MS``; ``sd_ground.services.wind``.)

Landing in wind: SITL keeps pushing a drone on the ground, and ArduCopter's land
detector then never confirms the landing. A landing counts when the drone has been on
the ground for 3 s, and the wind is set to 0 before any disarm the test waits for.
"""

from __future__ import annotations

import asyncio
import json
import math
import re
import statistics
import sys
import time
from collections.abc import Callable
from dataclasses import dataclass, field
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent))

from test_sd_quad14_sitl import (  # pyright: ignore[reportMissingImports]
    SITL,
    Track,
    fly_scenario,
    land,
    record,
    take_off,
)

from sd_ground.auth import Principal
from sd_ground.mav.thread import GroundMav, ThreadSafeMav
from sd_ground.services import wind
from sd_ground.services.latches import Latches
from sd_ground.services.vehicle_actions import ActionRefusedError, VehicleActions
from sd_mav import Role, Vehicle, connect
from sd_obs.clock import MonotonicClock
from sd_safety.constants import FC_ANGLE_MAX_DEG, FC_SAFETY_EXPECTED
from sd_safety.targets import Plane
from sd_sim import paths, ports
from sd_sim.harness import Harness, VehicleState
from sd_sim.report import Report

VARIANT = "okalibrerad"
ALT_M = 30.0  # = RTL_ALT_M, so RTL flies home at the height the leg flew
SETTLE_S = 15.0
FROM_SOUTH = 180.0  # SIM_WIND_DIR: the wind blows north
FROM_WEST = 270.0

HOLD_WIND_MS = 10.0
HOLD_S = 60.0
HOLD_TURB_MS = 2.0
HOLD_DRIFT_M = 3.0
HOLD_DRIFT_SHARE = 0.95
HOLD_ALT_M = 1.5

HOME_WIND_MS = 12.0
OUT_M = 500.0
OUT_SPEED_MS = 10.0
HOME_RADIUS_M = 5.0
TIME_FACTOR = 1.5
MAH_FACTOR = 1.2
HOLD_CURRENT_S = 20.0
RTL_WAIT_S = 600.0  # the slowest leg the sweep waits for (about 14 m/s against the model)
ON_GROUND_M = 0.3
ON_GROUND_S = 3.0

GUST_BASE_MS = 8.0
GUST_MS = 15.0
GUST_S = 10.0
GUST_TC_S = 1.0
LEG_M = 300.0
LEG_SPEED_MS = 5.0
LEG_OFF_M = 5.0
LEG_ALT_M = 3.0
EKF_TEXT = re.compile(r"\bEKF", re.IGNORECASE)

MARGIN_WIND_MS = 10.0
MARGIN_PCT = 15.2  # just above BATT_LOW_MAH (15 %): the pack crosses it within seconds
FS_DELAY_S = 5.0
LOW_BATTERY = re.compile(r"battery.*\blow\b", re.IGNORECASE)
CRITICAL_BATTERY = re.compile(r"battery.*\bcritical\b", re.IGNORECASE)
HOME_MODES = ("SMART_RTL", "RTL")

SWEEP_MS = tuple(range(0, 20, 2))
SWEEP_MUST_MS = 12.0

RTL_SPEED_MS = float(FC_SAFETY_EXPECTED["RTL_SPEED_MS"])
CAPACITY_MAH = float(FC_SAFETY_EXPECTED["BATT_CAPACITY"])
LOW_MAH = float(FC_SAFETY_EXPECTED["BATT_LOW_MAH"])


def frame() -> dict[str, float]:
    path = paths.REPO / "simulation" / "sitl" / "frames" / "sd_quad14.json"
    return json.loads(path.read_text(encoding="utf-8"))


def max_airspeed_ms() -> float:
    """The airspeed at which the model leans ``ATC_ANGLE_MAX``; the drag grows with the
    square of the speed, so the lean's tangent scales with it squared."""
    f = frame()
    ratio = math.tan(math.radians(FC_ANGLE_MAX_DEG)) / math.tan(math.radians(f["refAngle"]))
    return float(f["refSpd"]) * math.sqrt(ratio)


def model_home_s(distance: float, headwind_ms: float) -> float | None:
    """Seconds home against ``headwind_ms``; None when the model cannot get there."""
    ground = min(RTL_SPEED_MS, max_airspeed_ms() - headwind_ms)
    return distance / ground if ground > 0 else None


def set_wind(harness: Harness, speed_ms: float, from_deg: float, turb_ms: float = 0.0) -> None:
    harness.set_sim_param("SIM_WIND_T", 1)
    harness.set_sim_param("SIM_WIND_DIR", from_deg)
    harness.set_sim_param("SIM_WIND_TURB", turb_ms)
    harness.set_sim_param("SIM_WIND_SPD", speed_ms)


@dataclass
class Energy:
    """mAh from SYS_STATUS current over wall time (SITL runs in real time)."""

    mah: float = 0.0
    _last: float | None = None
    samples: list[float] = field(default_factory=list)

    def listener(self, kind: str, s: VehicleState) -> None:
        if kind != "SYS_STATUS" or s.battery_current_a < 0:
            return
        now = time.monotonic()
        if self._last is not None:
            self.mah += s.battery_current_a * (now - self._last) / 3.6
        self._last = now
        self.samples.append(s.battery_current_a)


def remaining_mah(s: VehicleState) -> float:
    return s.battery_pct / 100.0 * CAPACITY_MAH


def wind_report(
    name: str, instance: int, brain: str, flight: Callable[[Harness, Report], None]
) -> Report:
    def tagged(harness: Harness, rep: Report) -> None:
        rep.variant = VARIANT
        flight(harness, rep)

    return fly_scenario(name, "builtin", instance, brain, tagged)


def hold_drift(track: Track, centre: Plane) -> float:
    """The distance from ``centre`` that ``HOLD_DRIFT_SHARE`` of the samples stay within."""
    dist = sorted(math.hypot(*centre.xy(lat, lon)) for lat, lon in track.pos)
    return dist[min(len(dist) - 1, int(HOLD_DRIFT_SHARE * len(dist)))]


# --- W1 hold in 10 m/s --------------------------------------------------------------------


def hold_flight(harness: Harness, rep: Report) -> None:
    take_off(harness, ALT_M)
    centre = Plane(harness.state.lat, harness.state.lon)
    for cid, turb in (("W1.a", 0.0), ("W1.b", HOLD_TURB_MS)):
        set_wind(harness, HOLD_WIND_MS, FROM_WEST, turb)
        harness.pump(SETTLE_S)
        track = record(harness, HOLD_S)
        drift = hold_drift(track, centre)
        what = f"{HOLD_WIND_MS:g} m/s" + (f", turbulens {turb:g} m/s" if turb else "")
        rep.check(
            f"{cid}1",
            f"Håller läget i {what} (okalibrerad): {HOLD_DRIFT_SHARE:.0%} av tiden inom "
            f"{HOLD_DRIFT_M:g} m",
            drift < HOLD_DRIFT_M,
            measured=round(drift, 2),
            limit=HOLD_DRIFT_M,
            unit="m",
        )
        worst = max(abs(a - ALT_M) for a in track.alt)
        rep.check(
            f"{cid}2",
            f"Höjden inom ±{HOLD_ALT_M:g} m i {what} (okalibrerad)",
            worst <= HOLD_ALT_M,
            measured=round(worst, 2),
            limit=HOLD_ALT_M,
            unit="m",
        )
    harness.set_sim_param("SIM_WIND_SPD", 0)  # see wait_on_ground
    land(harness)


@pytest.mark.budget_s(330)
def test_w1_hold_in_10_ms(sim_instance: int, sim_brain: str) -> None:
    rep = wind_report("wind_w1_hold", sim_instance, sim_brain, hold_flight)
    assert rep.passed, "\n".join(rep.failures())


# --- W2 home against the wind (and the nightly sweep) --------------------------------------


@dataclass
class HomeRun:
    home: bool
    seconds: float | None  # RTL until over home (horizontal leg only); None: never got there
    mah: float
    model_s: float | None
    model_mah: float | None
    hold_a: float
    left_mah: float
    off_home_m: float


def wait_on_ground(
    harness: Harness, timeout_s: float, on_tick: Callable[[VehicleState], None] | None = None
) -> bool:
    """True once the drone has been on the ground for ``ON_GROUND_S``, False after
    ``timeout_s``. SITL keeps pushing a landed drone with the wind, and ArduCopter's land
    detector then never confirms the landing, so the tests read the height instead of
    waiting for the disarm, and set the wind to 0 on the ground."""
    end = time.monotonic() + timeout_s
    since: float | None = None
    while time.monotonic() < end:
        harness.pump(0.5)
        s = harness.state
        if on_tick is not None:
            on_tick(s)
        if s.rel_alt_m > ON_GROUND_M:
            since = None
        elif since is None:
            since = time.monotonic()
        elif time.monotonic() - since >= ON_GROUND_S:
            harness.set_sim_param("SIM_WIND_SPD", 0)
            return True
    return False


def reach_out(harness: Harness, home: Plane) -> None:
    """GUIDED ``OUT_M`` north at ``OUT_SPEED_MS``. Waits until the drone has got that far,
    not until it holds there: in more wind than the model can hold it drifts on."""
    lat, lon = home.latlon(0.0, OUT_M)
    harness.goto(lat, lon, ALT_M)
    harness.pump(0.5)  # a new GUIDED target resets the speed to WP_SPD
    harness.set_speed(OUT_SPEED_MS)
    harness.wait(
        lambda s: home.xy(s.lat, s.lon)[1] >= OUT_M - 3.0, 180, f"punkten {OUT_M:.0f} m ut"
    )
    harness.pump(SETTLE_S)


def fly_home_against(harness: Harness, wind_ms: float) -> HomeRun:
    """Out ``OUT_M`` north with ``wind_ms`` behind, measure the current while holding in
    it, then RTL south into it. Times the leg until the drone is over home and waits for
    the landing, at most ``RTL_WAIT_S``."""
    take_off(harness, ALT_M)
    home = Plane(harness.state.lat, harness.state.lon)
    set_wind(harness, wind_ms, FROM_SOUTH)
    reach_out(harness, home)
    hold = record(harness, HOLD_CURRENT_S)
    hold_a = statistics.fmean(hold.current) if hold.current else 0.0
    s = harness.state
    model_s = model_home_s(math.hypot(*home.xy(s.lat, s.lon)), wind_ms)
    energy = Energy()
    harness.listeners.append(energy.listener)
    started = time.monotonic()
    seconds: float | None = None

    def arrived(st: VehicleState) -> None:
        nonlocal seconds
        if seconds is None and math.hypot(*home.xy(st.lat, st.lon)) <= HOME_RADIUS_M:
            seconds = time.monotonic() - started
            harness.listeners.remove(energy.listener)

    try:
        harness.set_mode("RTL")
        # Not getting home is a measured result: past the fence its failsafe lands.
        landed = wait_on_ground(harness, RTL_WAIT_S, arrived)
    finally:
        if energy.listener in harness.listeners:
            harness.listeners.remove(energy.listener)
    s = harness.state
    off = math.hypot(*home.xy(s.lat, s.lon))
    return HomeRun(
        home=landed and off <= HOME_RADIUS_M,
        seconds=seconds,
        mah=energy.mah,
        model_s=model_s,
        model_mah=None if model_s is None else model_s * hold_a / 3.6,
        hold_a=hold_a,
        left_mah=remaining_mah(harness.state),
        off_home_m=off,
    )


def home_flight(harness: Harness, rep: Report) -> None:
    run = fly_home_against(harness, HOME_WIND_MS)
    rep.check(
        "W2.1",
        f"Landar inom {HOME_RADIUS_M:g} m från hem efter RTL mot {HOME_WIND_MS:g} m/s "
        "(okalibrerad)",
        run.home,
        measured={"off_home_m": round(run.off_home_m, 1)},
        limit=HOME_RADIUS_M,
        unit="m",
    )
    limit_s = None if run.model_s is None else TIME_FACTOR * run.model_s
    rep.check(
        "W2.2",
        f"Vägen hem (till {HOME_RADIUS_M:g} m från hem) tar högst {TIME_FACTOR:g} x "
        "modellens tid (okalibrerad)",
        limit_s is not None and run.seconds is not None and run.seconds <= limit_s,
        measured={
            "seconds": run.seconds and round(run.seconds),
            "model_s": run.model_s and round(run.model_s),
        },
        limit=limit_s and round(limit_s),
        unit="s",
    )
    limit_mah = None if run.model_mah is None else MAH_FACTOR * run.model_mah
    rep.check(
        "W2.3",
        f"Vägen hem förbrukar högst modellens mAh + {MAH_FACTOR - 1:.0%} (okalibrerad)",
        limit_mah is not None and run.seconds is not None and run.mah <= limit_mah,
        measured={
            "mah": round(run.mah),
            "model_mah": run.model_mah and round(run.model_mah),
            "hold_current_a": round(run.hold_a, 1),
        },
        limit=limit_mah and round(limit_mah),
        unit="mAh",
    )
    rep.check(
        "W2.4",
        "Minst BATT_LOW_MAH kvar vid landning (okalibrerad)",
        run.left_mah >= LOW_MAH,
        measured=round(run.left_mah),
        limit=LOW_MAH,
        unit="mAh",
    )


@pytest.mark.budget_s(480)
def test_w2_home_against_12_ms(sim_instance: int, sim_brain: str) -> None:
    rep = wind_report("wind_w2_home", sim_instance, sim_brain, home_flight)
    assert rep.passed, "\n".join(rep.failures())


def sweep_flight(wind_ms: float) -> Callable[[Harness, Report], None]:
    def flight(harness: Harness, rep: Report) -> None:
        rep.variant = f"{VARIANT}-{wind_ms:g}ms"
        run = fly_home_against(harness, wind_ms)
        must = wind_ms <= SWEEP_MUST_MS
        rep.check(
            "WS.1",
            f"Kommer hem mot {wind_ms:g} m/s (krav upp till {SWEEP_MUST_MS:g} m/s, okalibrerad)",
            run.home or not must,
            measured={
                "wind_ms": wind_ms,
                "home": run.home,
                "seconds": run.seconds and round(run.seconds),
                "mah": round(run.mah),
                "left_mah": round(run.left_mah),
                "off_home_m": round(run.off_home_m, 1),
            },
            limit=HOME_RADIUS_M,
            unit="m",
        )

    return flight


@pytest.mark.nightly
@pytest.mark.budget_s(900)
@pytest.mark.parametrize("wind_ms", SWEEP_MS)
def test_wind_sweep_home(sim_instance: int, sim_brain: str, wind_ms: int) -> None:
    rep = fly_scenario(
        f"wind_sweep_{wind_ms:02d}", "builtin", sim_instance, sim_brain, sweep_flight(wind_ms)
    )
    assert rep.passed, "\n".join(rep.failures())


# --- W3 gust to 15 m/s ---------------------------------------------------------------------


def gust_flight(harness: Harness, rep: Report) -> None:
    take_off(harness, ALT_M)
    start = Plane(harness.state.lat, harness.state.lon)
    harness.set_sim_param("SIM_WIND_TC", GUST_TC_S)
    set_wind(harness, GUST_BASE_MS, FROM_SOUTH)
    harness.pump(SETTLE_S)
    lat, lon = start.latlon(LEG_M, 0.0)
    target = Plane(lat, lon)
    off: list[float] = []
    alt: list[float] = []
    ekf_lost = False

    def on_message(kind: str, s: VehicleState) -> None:
        nonlocal ekf_lost
        if kind == "GLOBAL_POSITION_INT":
            east, north = start.xy(s.lat, s.lon)
            if 0.0 <= east <= LEG_M:
                off.append(abs(north))
                alt.append(s.rel_alt_m)
        elif kind == "EKF_STATUS_REPORT" and not s.ekf_pos_ok:
            ekf_lost = True

    texts_from = time.monotonic()
    harness.listeners.append(on_message)
    try:
        harness.goto(lat, lon, ALT_M)
        harness.pump(0.5)
        harness.set_speed(LEG_SPEED_MS)
        harness.wait(lambda s: start.xy(s.lat, s.lon)[0] >= LEG_M / 2 - GUST_S * LEG_SPEED_MS / 2,
                     120, "mitten av sträckan")  # fmt: skip
        harness.set_sim_param("SIM_WIND_SPD", GUST_MS)
        harness.pump(GUST_S)
        harness.set_sim_param("SIM_WIND_SPD", GUST_BASE_MS)
        harness.wait(lambda s: math.hypot(*target.xy(s.lat, s.lon)) < 3.0, 120, "slutet")
    finally:
        harness.listeners.remove(on_message)
    worst = max(off, default=math.inf)
    rep.check(
        "W3.1",
        f"Av linjen mindre än {LEG_OFF_M:g} m med by till {GUST_MS:g} m/s (okalibrerad)",
        worst < LEG_OFF_M,
        measured={"max_m": round(worst, 2), "samples": len(off)},
        limit=LEG_OFF_M,
        unit="m",
    )
    band = max((abs(a - ALT_M) for a in alt), default=math.inf)
    rep.check(
        "W3.2",
        f"Höjden inom ±{LEG_ALT_M:g} m (okalibrerad)",
        band <= LEG_ALT_M,
        measured=round(band, 2),
        limit=LEG_ALT_M,
        unit="m",
    )
    ekf_texts = [t for at, t in harness.state.texts if at >= texts_from and EKF_TEXT.search(t)]
    rep.check(
        "W3.3",
        "EKF:ns position finns hela tiden och inget EKF-meddelande (okalibrerad)",
        not ekf_lost and not ekf_texts,
        measured={"position_lost": ekf_lost, "texts": ekf_texts[:5]},
    )
    harness.set_sim_param("SIM_WIND_SPD", 0)  # see wait_on_ground
    land(harness)


@pytest.mark.budget_s(300)
def test_w3_gust_to_15_ms(sim_instance: int, sim_brain: str) -> None:
    rep = wind_report("wind_w3_gust", sim_instance, sim_brain, gust_flight)
    assert rep.passed, "\n".join(rep.failures())


# --- W4 battery margin ---------------------------------------------------------------------


def margin_flight(harness: Harness, rep: Report) -> None:
    take_off(harness, ALT_M)
    home = Plane(harness.state.lat, harness.state.lon)
    set_wind(harness, MARGIN_WIND_MS, FROM_SOUTH)
    reach_out(harness, home)
    modes: set[str] = set()

    def on_message(kind: str, s: VehicleState) -> None:
        if kind == "HEARTBEAT":
            modes.add(s.mode)

    harness.listeners.append(on_message)
    set_at = time.monotonic()
    try:
        harness.battery_set_remaining(MARGIN_PCT)
        harness.wait(lambda s: s.mode in HOME_MODES, 60, "flygkontrollens batterifailsafe")
        home_at = time.monotonic()
        assert wait_on_ground(harness, RTL_WAIT_S), f"landade inte inom {RTL_WAIT_S:.0f} s"
    finally:
        harness.listeners.remove(on_message)
    low = [at for at, text in harness.state.texts if at >= set_at and LOW_BATTERY.search(text)]
    delay = home_at - low[0] if low else math.inf
    rep.check(
        "W4.1",
        f"SmartRTL/RTL inom {FS_DELAY_S:g} s efter batterivarningen (okalibrerad)",
        delay <= FS_DELAY_S,
        measured={"seconds": round(delay, 1), "mode": sorted(modes & set(HOME_MODES))},
        limit=FS_DELAY_S,
        unit="s",
    )
    s = harness.state
    off = math.hypot(*home.xy(s.lat, s.lon))
    rep.check(
        "W4.2",
        f"Landar inom {HOME_RADIUS_M:g} m från hem mot {MARGIN_WIND_MS:g} m/s (okalibrerad)",
        off <= HOME_RADIUS_M,
        measured=round(off, 1),
        limit=HOME_RADIUS_M,
        unit="m",
    )
    critical = [t for at, t in harness.state.texts if at >= set_at and CRITICAL_BATTERY.search(t)]
    rep.check(
        "W4.3",
        "Batteriets kritiska nivå (LAND) utlöses aldrig (okalibrerad)",
        "LAND" not in modes and not critical,
        measured={"modes": sorted(modes), "texts": critical[:3], "left_pct": s.battery_pct},
    )


@pytest.mark.budget_s(360)
def test_w4_battery_margin_against_10_ms(sim_instance: int, sim_brain: str) -> None:
    rep = wind_report("wind_w4_margin", sim_instance, sim_brain, margin_flight)
    assert rep.passed, "\n".join(rep.failures())


# --- W5 the flight controller's own lean limit ----------------------------------------------

FPV7_ANGLE_MAX_DEG = 35.0  # sd_fpv7-sim.parm
PARAMS_WAIT_S = 30.0
W5_VID = "sim-01"


async def ground_wind_check(instance: int) -> tuple[wind.FcParams, float | None]:
    """The ground station's own path: ``GroundMav`` on 250/190 and ``ThreadSafeMav``,
    whose ``update_wind`` reads ``WIND_PARAMS`` on connect. Returns what it read and the
    tilt warning it then shows."""
    clock = MonotonicClock()
    url = f"udpin:127.0.0.1:{ports.for_instance(instance)['ground_mavlink']}"
    conn = connect(url, role=Role.GROUND, source_system=250, source_component=190)
    vehicle = Vehicle(conn, role=Role.GROUND, clock=clock)
    core = GroundMav(
        vehicle,
        clock,
        vehicle_id=W5_VID,
        latches=Latches(clock),
        on_report=lambda _: None,
        on_event=lambda *_: None,
        system_action=lambda *_: None,
    )
    mav = ThreadSafeMav(core, asyncio.get_running_loop())
    await core.start()
    try:
        assert await vehicle.wait_connected(20), "ingen HEARTBEAT på markstationens kanal"
        warn: float | None = None
        end = time.monotonic() + PARAMS_WAIT_S
        while time.monotonic() < end:
            mav.update_wind(W5_VID)
            out = mav.telemetry(W5_VID)
            if out is not None and out.wind is not None:
                warn = out.wind.tilt_warn_deg
                if mav.wind.params.angle_max_deg != FC_ANGLE_MAX_DEG:
                    break
            await asyncio.sleep(0.2)
        return mav.wind.params, warn
    finally:
        await core.close()


def fc_params_flight(instance: int) -> Callable[[Harness, Report], None]:
    def flight(harness: Harness, rep: Report) -> None:
        rep.variant = VARIANT
        params, warn = asyncio.run(ground_wind_check(instance))
        rep.check(
            "W5.1",
            "Markstationen läser ATC_ANGLE_MAX från flygkontrollen (okalibrerad)",
            params.angle_max_deg == FPV7_ANGLE_MAX_DEG and "ATC_ANGLE_MAX" not in params.fallback,
            measured={"angle_max_deg": params.angle_max_deg, "fallback": list(params.fallback)},
            limit=FPV7_ANGLE_MAX_DEG,
            unit="deg",
        )
        rep.check(
            "W5.2",
            "Lutningsvarningen ligger 10 grader under flygkontrollens egen gräns (okalibrerad)",
            warn == FPV7_ANGLE_MAX_DEG - wind.TILT_MARGIN_DEG,
            measured=warn,
            limit=FPV7_ANGLE_MAX_DEG - wind.TILT_MARGIN_DEG,
            unit="deg",
        )
        rep.check(
            "W5.3",
            "RTL-fart och batteriets mAh kommer från flygkontrollen, inte konstanterna",
            params.fallback == ()
            and params.rtl_speed_ms == RTL_SPEED_MS
            and params.reserve_mah == LOW_MAH,
            measured={
                "rtl_speed_ms": params.rtl_speed_ms,
                "capacity_mah": params.capacity_mah,
                "reserve_mah": params.reserve_mah,
                "fallback": list(params.fallback),
            },
        )

    return flight


@pytest.mark.budget_s(120)
def test_w5_ground_reads_the_fc_lean_limit(sim_instance: int, sim_brain: str) -> None:
    rep = fly_scenario(
        "wind_w5_fc_params",
        "builtin",
        sim_instance,
        sim_brain,
        fc_params_flight(sim_instance),
        vehicle="sd_fpv7",
    )
    assert rep.passed, "\n".join(rep.failures())


# --- W6 ENERGY_HOME flies home ----------------------------------------------------------

W6_OUT_M = 150.0
W6_PCT = 14.0  # under BATT_LOW_MAH (15 %), taken silently at the reset; over BATT_CRT (7 %)
W6_RTL_PCT = 5.0  # the percentage level stays out of the way: only the energy can fly home
W6_FIRE_S = 20.0
W6_TAKEOVER_S = 10.0
W6_HOLD_S = 10.0
W6_PILOT = Principal(role="pilot", issuer_ref="operator:sitl", auth_mode="local")


@dataclass
class EnergyRun:
    fired_mode: str | None = None
    reasons: list[str] = field(default_factory=list[str])
    resume_code: str | None = None
    resume_text: str = ""
    takeover_mode: str | None = None
    hold_modes: set[str] = field(default_factory=set[str])
    reasons_in_hold: list[str] = field(default_factory=list[str])
    landed: bool = False


async def ground_energy_home(harness: Harness, instance: int) -> EnergyRun:
    """The ground station's battery guard against SITL: ``GroundMav`` with ``Latches`` and
    ``VehicleActions`` as in ``sd_ground``, ``update_wind`` five times a second as the
    telemetry publisher calls it. The harness runs in a worker thread while the ground
    station's loop keeps its heartbeat (``MAV_GCS_SYSID`` 250) going."""
    clock = MonotonicClock()
    url = f"udpin:127.0.0.1:{ports.for_instance(instance)['ground_mavlink']}"
    conn = connect(url, role=Role.GROUND, source_system=250, source_component=190)
    vehicle = Vehicle(conn, role=Role.GROUND, clock=clock)
    latches = Latches(clock)
    actions = VehicleActions(clock, latches, vehicle_ids=[W5_VID])
    events: list[tuple[str, dict[str, object]]] = []

    def on_event(type_: str, severity: str, payload: dict[str, object]) -> None:
        events.append((type_, payload))
        actions.on_event(type_, severity, payload)

    core = GroundMav(
        vehicle,
        clock,
        vehicle_id=W5_VID,
        latches=latches,
        on_report=actions.on_report,
        on_event=on_event,
        system_action=actions.system,
        rtl_pct=lambda: W6_RTL_PCT,
    )
    mav = ThreadSafeMav(core, asyncio.get_running_loop())
    actions.mav = mav
    run = EnergyRun()

    def mode() -> str | None:
        snap = mav.snapshot()
        return None if snap is None else snap.telemetry.fc_mode

    def reasons() -> list[str]:
        return [str(p["reason"]) for t, p in events if t == "vehicle.battery_rtl" and "alarm" in p]

    async def until(ok: Callable[[], bool], timeout_s: float) -> bool:
        end = time.monotonic() + timeout_s
        while time.monotonic() < end:
            mav.update_wind(W5_VID)
            if ok():
                return True
            await asyncio.sleep(0.2)
        return ok()

    await core.start()
    try:
        assert await vehicle.wait_connected(20), "ingen HEARTBEAT på markstationens kanal"
        snap_ok = await until(
            lambda: (snap := mav.snapshot()) is not None and snap.connected and snap.in_flight,
            PARAMS_WAIT_S,
        )
        assert snap_ok, "markstationen ser inte drönaren i luften"
        await until(lambda: mav.wind.params.fallback == (), PARAMS_WAIT_S)
        await asyncio.to_thread(harness.battery_set_remaining, W6_PCT)
        await until(lambda: mode() in HOME_MODES and bool(reasons()), W6_FIRE_S)
        run.fired_mode, run.reasons = mode(), reasons()

        try:
            actions.resume(W5_VID, client_action_id="w6-resume", by=W6_PILOT)
        except ActionRefusedError as exc:
            run.resume_code, run.resume_text = exc.code, str(exc.detail_sv)

        actions.dispatch("stop", W5_VID, client_action_id="w6-stop", via="button", by=W6_PILOT)
        await until(lambda: False, 2.0)
        actions.takeover(W5_VID, client_action_id="w6-takeover", by=W6_PILOT)
        await until(lambda: mode() == "LOITER", W6_TAKEOVER_S)
        run.takeover_mode = mode()
        before = len(reasons())
        end = time.monotonic() + W6_HOLD_S
        while time.monotonic() < end:
            await until(lambda: False, 0.5)
            if (m := mode()) is not None:
                run.hold_modes.add(m)
        run.reasons_in_hold = reasons()[before:]

        # Home and down with the ground station still connected (its heartbeat keeps the
        # flight controller's GCS failsafe away).
        actions.dispatch(
            "return_home", W5_VID, client_action_id="w6-home", via="button", by=W6_PILOT
        )
        run.landed = await asyncio.to_thread(wait_on_ground, harness, RTL_WAIT_S)
        return run
    finally:
        await core.close()


def energy_home_flight(instance: int) -> Callable[[Harness, Report], None]:
    def flight(harness: Harness, rep: Report) -> None:
        rep.variant = VARIANT
        take_off(harness, ALT_M)
        home = Plane(harness.state.lat, harness.state.lon)
        lat, lon = home.latlon(0.0, W6_OUT_M)
        harness.goto(lat, lon, ALT_M)
        harness.wait(
            lambda s: home.xy(s.lat, s.lon)[1] >= W6_OUT_M - 3.0, 120, f"{W6_OUT_M:.0f} m ut"
        )
        run = asyncio.run(ground_energy_home(harness, instance))
        rep.check(
            "W6.1",
            "ENERGY_HOME flyger hem: SmartRTL/RTL från markstationen (okalibrerad)",
            run.fired_mode in HOME_MODES,
            measured=run.fired_mode,
        )
        rep.check(
            "W6.2",
            "Händelsens orsak är energy_home",
            run.reasons[:1] == ["energy_home"],
            measured=run.reasons,
        )
        rep.check(
            "W6.3",
            "Resume nekas med BATTERY_RTL_ACTIVE och texten nämner vinden",
            run.resume_code == "BATTERY_RTL_ACTIVE" and "vinden" in run.resume_text,
            measured={"code": run.resume_code, "text": run.resume_text},
        )
        rep.check(
            "W6.4",
            f"Efter STOPP och Ta över ligger drönaren i LOITER i {W6_HOLD_S:g} s, "
            "ingen ny energiutlösning",
            run.takeover_mode == "LOITER"
            and run.hold_modes == {"LOITER"}
            and not run.reasons_in_hold,
            measured={
                "takeover_mode": run.takeover_mode,
                "modes": sorted(run.hold_modes),
                "reasons": run.reasons_in_hold,
            },
        )
        assert run.landed, f"landade inte inom {RTL_WAIT_S:.0f} s"

    return flight


@pytest.mark.budget_s(300)
def test_w6_energy_home_flies_home(sim_instance: int, sim_brain: str) -> None:
    rep = fly_scenario(
        "wind_w6_energy_home", "builtin", sim_instance, sim_brain, energy_home_flight(sim_instance)
    )
    assert rep.passed, "\n".join(rep.failures())


pytestmark = SITL
