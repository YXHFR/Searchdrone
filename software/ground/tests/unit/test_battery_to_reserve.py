"""The AI's "till hemgräns" minutes: seconds to the home limit from the ground's own
telemetry (level, the wind check's mean current, the pack's capacity), None without them."""

from __future__ import annotations

import dataclasses
import math

import pytest

from sd_ground.mav.thread import MavSnapshot
from sd_ground.services.situation import (
    _vehicle,  # pyright: ignore[reportPrivateUsage]
    battery_to_reserve_s,
    home_limit_pct,
)
from sd_ground.services.wind import CAPACITY_MAH, TILT_WINDOW_S, BatteryDrain, FcParams, WindWatch
from sd_mav.vehicle import FcTelemetry

FLYING = FcTelemetry(
    armed=True,
    landed_state="in_air",
    alt_rel_home_m=40.0,
    battery_pct=60.0,
    battery_current_a=30.0,
)
DRAIN = BatteryDrain(current_a=30.0, capacity_mah=10_000.0)


def _snap(t: FcTelemetry, *, connected: bool = True) -> MavSnapshot:
    return MavSnapshot(
        vehicle_id="sim-01",
        connected=connected,
        telemetry=t,
        now_mono_ms=1_000,
        in_flight=True,
        onboard_alive=True,
        airborne_since=None,
        manual=False,
        watchdog_phase="ok",
    )


def test_the_seconds_are_the_charge_above_the_limit_over_the_current() -> None:
    # 30 % of 10 Ah at 30 A: 3000 mAh * 3.6 s / 30 A = 360 s.
    assert battery_to_reserve_s(60.0, rtl_pct=30.0, drain=DRAIN) == 360
    # The capacity the connection read counts, not the constant.
    small = dataclasses.replace(DRAIN, capacity_mah=5_000.0)
    assert battery_to_reserve_s(60.0, rtl_pct=30.0, drain=small) == 180


def test_at_or_below_the_limit_it_is_zero() -> None:
    assert battery_to_reserve_s(30.0, rtl_pct=30.0, drain=DRAIN) == 0
    assert battery_to_reserve_s(25.0, rtl_pct=30.0, drain=DRAIN) == 0


@pytest.mark.parametrize(
    ("pct", "rtl", "drain"),
    [
        (None, 30.0, DRAIN),
        (60.0, 30.0, None),
        (60.0, 30.0, BatteryDrain(0.0, 10_000.0)),  # no current measured
        (60.0, 30.0, BatteryDrain(0.4, 10_000.0)),  # not a flight's current
        (60.0, 30.0, BatteryDrain(-5.0, 10_000.0)),
        (60.0, 30.0, BatteryDrain(30.0, 0.0)),
        (math.nan, 30.0, DRAIN),
        (60.0, math.nan, DRAIN),
        (60.0, 30.0, BatteryDrain(math.inf, 10_000.0)),
    ],
)
def test_without_usable_data_it_is_none(
    pct: float | None, rtl: float, drain: BatteryDrain | None
) -> None:
    assert battery_to_reserve_s(pct, rtl_pct=rtl, drain=drain) is None


def test_the_home_limit_is_the_charters_never_below_the_mission_limit() -> None:
    assert home_limit_pct(25.0, None) == 25.0
    assert home_limit_pct(25.0, 35.0) == 35.0
    assert home_limit_pct(25.0, 20.0) == 25.0
    # A charter at 35 % leaves 25 % of the pack, not the 35 % above the mission's 25 %.
    rtl = home_limit_pct(25.0, 35.0)
    assert battery_to_reserve_s(60.0, rtl_pct=rtl, drain=DRAIN) == 300


def test_the_ai_sees_the_seconds_in_flight() -> None:
    v = _vehicle(_snap(FLYING), None, True, None, rtl_pct=30.0, drain=DRAIN)
    assert v.battery_to_reserve_s == 360


@pytest.mark.parametrize(
    "t",
    [
        dataclasses.replace(FLYING, battery_pct=None),
        dataclasses.replace(FLYING, landed_state="on_ground", alt_rel_home_m=0.0),
        dataclasses.replace(FLYING, armed=False),
    ],
)
def test_without_a_level_or_a_flight_the_ai_sees_none(t: FcTelemetry) -> None:
    v = _vehicle(_snap(t), None, True, None, rtl_pct=30.0, drain=DRAIN)
    assert v.battery_to_reserve_s is None


def test_without_a_link_or_a_current_the_ai_sees_none() -> None:
    off = _vehicle(_snap(FLYING, connected=False), None, True, None, rtl_pct=30.0, drain=DRAIN)
    assert off.battery_to_reserve_s is None
    assert _vehicle(_snap(FLYING), None, True, None, rtl_pct=30.0).battery_to_reserve_s is None
    assert _vehicle(None, None, True, None, rtl_pct=30.0, drain=DRAIN).battery_to_reserve_s is None


def _feed(watch: WindWatch, t: FcTelemetry, start_ms: int, seconds: float) -> int:
    now = start_ms
    for i in range(round(seconds * 4) + 1):  # ATTITUDE at 4 Hz
        now = start_ms + i * 250
        watch.observe(dataclasses.replace(t, attitude_mono_ms=now), now)
    return now


def test_the_current_is_the_mean_of_the_last_window() -> None:
    watch = WindWatch()
    now = _feed(watch, dataclasses.replace(FLYING, battery_current_a=40.0), 0, 5.0)
    now = _feed(watch, dataclasses.replace(FLYING, battery_current_a=20.0), now + 250, 5.0)
    drain = watch.drain(now)
    assert drain is not None
    assert drain.current_a == pytest.approx(30.0, abs=1.0)  # one moment does not stand alone
    assert drain.capacity_mah == CAPACITY_MAH


def test_the_drain_uses_the_capacity_the_connection_read() -> None:
    watch = WindWatch(params=FcParams(capacity_mah=5_000.0, fallback=()))
    now = _feed(watch, FLYING, 0, 2.0)
    drain = watch.drain(now)
    assert drain is not None and drain.capacity_mah == 5_000.0


def test_without_fresh_current_samples_there_is_no_drain() -> None:
    watch = WindWatch()
    assert watch.drain(0) is None
    now = _feed(watch, dataclasses.replace(FLYING, battery_current_a=None), 0, 2.0)
    assert watch.drain(now) is None
    now = _feed(watch, FLYING, now + 250, 2.0)
    assert watch.drain(now) is not None
    # Samples older than the window are not a current the drone draws now.
    assert watch.drain(now + int(TILT_WINDOW_S * 1000) + 1) is None
    # On the ground the window starts over.
    watch.observe(dataclasses.replace(FLYING, armed=False, landed_state="on_ground"), now + 250)
    assert watch.drain(now + 250) is None
