"""The downward laser's reader: which readings count, when the laser is confirmed, and
which surface it reports (spec "Titta närmare" §6.4)."""

from __future__ import annotations

import asyncio
import math
from dataclasses import replace
from types import SimpleNamespace
from typing import Any

import pytest

from sd_mav.fake import CMAC_HOME, FakeVehicle
from sd_mav.vehicle import FcTelemetry
from sd_onboard import app
from sd_onboard.config import Loaded, OnboardConfig, RangefinderConfig
from sd_onboard.rangefinder import LaserReader
from sd_schemas.rangefinder import LaserState
from sd_testing import FakeClock

TF02 = RangefinderConfig(model="tf02_pro", min_tested_m=0.1, max_tested_m=13.0)
GRF = RangefinderConfig(model="grf250", min_tested_m=10.0, max_tested_m=120.0)
ALT = 600.0  # the drone's height above sea level


def reading(ms: int, distance_m: float, **changes: Any) -> FcTelemetry:
    """Telemetry right after one DISTANCE_SENSOR, with fresh height and attitude."""
    base = FcTelemetry(
        fc_link_epoch=1,
        position_mono_ms=ms,
        alt_amsl_m=ALT,
        attitude_mono_ms=ms,
        rangefinder_mono_ms=ms,
        rangefinder_m=distance_m,
        rangefinder_min_m=0.1,
        rangefinder_max_m=40.0,
    )
    return replace(base, **changes)


def feed(laser: LaserReader, start_ms: int, end_ms: int, distance_m: float, **changes: Any) -> int:
    """Readings at 10 Hz from ``start_ms`` up to and including ``end_ms``; the last time."""
    ms = start_ms
    while ms <= end_ms:
        laser.observe(reading(ms, distance_m, **changes))
        ms += 100
    return ms - 100


def confirmed_at(laser: LaserReader, ms: int) -> LaserState:
    state = laser.state(ms)
    assert state.confirmed, state
    return state


def test_two_seconds_of_valid_readings_confirm_the_laser() -> None:
    laser = LaserReader(TF02)
    assert not laser.state(0).confirmed  # nothing read yet
    feed(laser, 0, 1900, 12.0)
    early = laser.state(1900)
    assert not early.confirmed and early.surface_amsl_m is None
    assert early.distance_m == 12.0 and early.min_range_m == 0.1
    feed(laser, 2000, 2000, 12.0)
    state = confirmed_at(laser, 2000)
    assert state.surface_amsl_m == pytest.approx(ALT - 12.0)


def test_the_surface_is_the_highest_of_the_last_two_seconds() -> None:
    """The nearest treetop counts, never an average; an older one drops out."""
    laser = LaserReader(TF02)
    feed(laser, 0, 1000, 12.0)
    laser.observe(reading(1100, 6.0))  # a treetop 6 m higher, once
    end = feed(laser, 1200, 3000, 12.0)
    assert confirmed_at(laser, end).surface_amsl_m == pytest.approx(ALT - 6.0)
    end = feed(laser, 3100, 3200, 12.0)  # 2.1 s after the treetop
    assert confirmed_at(laser, end).surface_amsl_m == pytest.approx(ALT - 12.0)


def test_the_distance_is_corrected_for_the_tilt() -> None:
    laser = LaserReader(TF02)
    end = feed(laser, 0, 2000, 12.0, roll_deg=15.0)
    state = confirmed_at(laser, end)
    assert state.distance_m == pytest.approx(12.0 * math.cos(math.radians(15.0)))
    assert state.surface_amsl_m == pytest.approx(ALT - 12.0 * math.cos(math.radians(15.0)))
    end = feed(laser, end + 100, end + 2100, 12.0, roll_deg=-10.0, pitch_deg=10.0)
    lean = math.cos(math.radians(10.0)) ** 2  # rolled and pitched: it leans both ways
    assert confirmed_at(laser, end).distance_m == pytest.approx(12.0 * lean)


def _bad(case: str) -> dict[str, Any]:
    return {
        "tilt": {"roll_deg": 20.0},  # at the limit
        "tilt_combined": {"roll_deg": 15.0, "pitch_deg": -14.0},  # each alone is fine
        "invalid_signal": {"rangefinder_quality": 1},
        "beyond_tested": {"rangefinder_m": 13.5},  # the laser reaches 40 m, tested 13
        "beyond_fc_max": {"rangefinder_m": 9.0, "rangefinder_max_m": 8.0},
        "below_fc_min": {"rangefinder_m": 0.05, "rangefinder_min_m": 0.1},
        "no_signal": {"rangefinder_m": 41.0},  # Benewake reports past its maximum
        "old_height": {"position_mono_ms": -1},
        "no_height": {"position_mono_ms": None},
        "old_attitude": {"attitude_mono_ms": -1},
    }[case]


@pytest.mark.parametrize(
    "case",
    [
        "tilt",
        "tilt_combined",
        "invalid_signal",
        "beyond_tested",
        "beyond_fc_max",
        "below_fc_min",
        "no_signal",
        "old_height",
        "no_height",
        "old_attitude",
    ],
)
def test_one_unusable_reading_breaks_the_run(case: str) -> None:
    laser = LaserReader(TF02)
    end = feed(laser, 0, 3000, 12.0)
    confirmed_at(laser, end)
    bad = _bad(case)
    ms = end + 100
    if bad.get("position_mono_ms") == -1:
        bad["position_mono_ms"] = ms - 600
    if bad.get("attitude_mono_ms") == -1:
        bad["attitude_mono_ms"] = ms - 600
    laser.observe(reading(ms, bad.pop("rangefinder_m", 12.0), **bad))
    state = laser.state(ms)
    assert not state.confirmed and state.distance_m is None
    end = feed(laser, ms + 100, ms + 2000, 12.0)  # two more seconds from the next good one
    assert not laser.state(end).confirmed
    feed(laser, end + 100, end + 100, 12.0)
    confirmed_at(laser, end + 100)


def test_the_tested_range_is_used_even_when_the_laser_claims_more() -> None:
    """GRF-250: under 10 m it is not trusted, whatever ArduPilot's range says."""
    laser = LaserReader(GRF)
    end = feed(laser, 0, 2500, 9.5, rangefinder_min_m=0.2, rangefinder_max_m=250.0)
    assert not laser.state(end).confirmed
    end = feed(laser, end + 100, end + 2100, 45.0, rangefinder_min_m=0.2, rangefinder_max_m=250)
    state = confirmed_at(laser, end)
    assert state.min_range_m == 10.0 and state.surface_amsl_m == pytest.approx(ALT - 45.0)


def test_a_silent_laser_is_no_longer_confirmed() -> None:
    laser = LaserReader(TF02)
    end = feed(laser, 0, 2500, 12.0)
    assert confirmed_at(laser, end + 500).distance_m == 12.0  # 0.5 s without a reading
    late = laser.state(end + 501)
    assert not late.confirmed and late.distance_m is None and late.surface_amsl_m is None


def test_a_gap_between_readings_starts_the_run_again() -> None:
    laser = LaserReader(TF02)
    end = feed(laser, 0, 2500, 12.0)
    end = feed(laser, end + 600, end + 1500, 12.0)  # 0.6 s gap, then 0.9 s of readings
    assert not laser.state(end).confirmed
    end = feed(laser, end + 100, end + 1100, 12.0)
    confirmed_at(laser, end)


def test_a_stuck_laser_is_not_trusted() -> None:
    """The same distance to the centimetre while the drone came 1 m down: stuck."""
    laser = LaserReader(TF02)
    ms = 0
    for step in range(25):  # coming down 0.1 m per reading, the laser keeps saying 12.00
        laser.observe(reading(ms, 12.0, alt_amsl_m=ALT - 0.1 * step))
        ms += 100
    assert not laser.state(ms - 100).confirmed
    # Hovering, the same value is no sign of a fault.
    end = feed(laser, ms, ms + 2500, 11.0, alt_amsl_m=ALT - 2.4)
    confirmed_at(laser, end)


def test_each_reading_counts_once_and_telemetry_without_one_is_skipped() -> None:
    laser = LaserReader(TF02)
    laser.observe(FcTelemetry())  # no DISTANCE_SENSOR yet
    assert laser.state(0) == LaserState(confirmed=False, surface_amsl_m=None, min_range_m=0.1)
    end = feed(laser, 0, 2000, 12.0)
    first = laser.state(end)
    for _ in range(3):  # a position update notifies again with the same reading
        laser.observe(reading(end, 12.0))
    assert laser.state(end) == first


def test_a_new_flight_controller_connection_starts_over() -> None:
    laser = LaserReader(TF02)
    end = feed(laser, 0, 2500, 12.0)
    confirmed_at(laser, end)
    laser.observe(reading(end + 100, 12.0, fc_link_epoch=2))
    assert not laser.state(end + 100).confirmed


def test_no_laser_means_no_reader(tmp_path: Any) -> None:
    with pytest.raises(ValueError):
        LaserReader(RangefinderConfig())
    vehicle = SimpleNamespace(subscribed=[])
    vehicle.subscribe = vehicle.subscribed.append
    none = Loaded(OnboardConfig(), tmp_path, {})
    assert app._laser(none, vehicle) is None  # type: ignore[arg-type]  # pyright: ignore[reportPrivateUsage, reportArgumentType]
    assert vehicle.subscribed == []
    fitted = Loaded(OnboardConfig(rangefinder=TF02), tmp_path, {})
    laser = app._laser(fitted, vehicle)  # type: ignore[arg-type]  # pyright: ignore[reportPrivateUsage, reportArgumentType]
    assert laser is not None and vehicle.subscribed == [laser.observe]


def test_the_reader_follows_the_flight_controller() -> None:
    """End to end: DISTANCE_SENSOR at 10 Hz from the fake flight controller, the reader on
    the vehicle's telemetry. A laser that stops sending is dropped within a second."""

    async def run() -> None:
        clock = FakeClock()
        vehicle = FakeVehicle(clock)
        laser = LaserReader(TF02)
        vehicle.subscribe(laser.observe)
        vehicle.fc.laser_m = 12.0
        vehicle.fc.laser_range_m = (0.1, 40.0)
        await vehicle.start()
        await clock.run_for(3.0)
        state = laser.state(clock.monotonic_ms())
        assert state.confirmed
        assert state.surface_amsl_m == pytest.approx(CMAC_HOME[2] - 12.0)
        vehicle.fc.laser_m = None
        await clock.run_for(1.0)
        assert not laser.state(clock.monotonic_ms()).confirmed
        await vehicle.close()

    asyncio.run(run())
