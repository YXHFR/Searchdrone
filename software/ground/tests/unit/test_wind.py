"""The ground station's wind check (``sd_ground.services.wind``): the warnings, the
flight controller's parameters it uses, the energy home against the wind and the
battery guard's FLY HOME on ``ENERGY_HOME``."""

from __future__ import annotations

import asyncio
import datetime as dt
import math
import random
from dataclasses import replace
from types import SimpleNamespace
from typing import cast

import pytest
import structlog
from conftest import FakeVehicleLink, telemetry  # pyright: ignore[reportMissingImports]
from stop_helpers import PILOT, VID, stack  # pyright: ignore[reportMissingImports]

from sd_ground.context import Ground
from sd_ground.mav.thread import GroundMav, MavSnapshot, ThreadSafeMav
from sd_ground.services import wind
from sd_ground.services.telemetry import TelemetryService
from sd_ground.services.vehicle_actions import ActionRefusedError
from sd_ground.services.wind import (
    FcParams,
    WindWatch,
    fc_params,
    home_ground_speed,
    home_mah,
    tilt_deg,
)
from sd_mav.modes import HOME_MODES, CopterMode
from sd_mav.vehicle import FcTelemetry
from sd_safety.constants import FC_ANGLE_MAX_DEG
from sd_safety.targets import distance_m
from sd_schemas.api import WindCheck
from sd_schemas.base import LatLonAlt
from sd_testing import FakeClock

HOME = LatLonAlt(lat=-35.363262, lon=149.165237, alt_amsl_m=584.0)
# About 500 m north of home.
FAR = (-35.358758, 149.165237)


def flying(**kw: object) -> FcTelemetry:
    base = FcTelemetry(
        armed=True,
        landed_state="in_air",
        home=HOME,
        lat=HOME.lat,
        lon=HOME.lon,
        battery_pct=80.0,
        battery_current_a=20.0,
    )
    return replace(base, **kw)  # type: ignore[arg-type]


def run(
    watch: WindWatch, t: FcTelemetry, seconds: float, *, start_ms: int = 0, hz: float = 4.0
) -> int:
    """Feed ATTITUDE samples at ``hz`` for ``seconds``; returns the last time."""
    now = start_ms
    for i in range(round(seconds * hz) + 1):
        now = start_ms + round(i * 1000 / hz)
        watch.observe(replace(t, attitude_mono_ms=now), now)
    return now


def test_tilt_is_the_angle_from_vertical() -> None:
    assert tilt_deg(0.0, 0.0) == 0.0
    assert tilt_deg(20.0, 0.0) == pytest.approx(20.0)
    assert tilt_deg(0.0, -25.0) == pytest.approx(25.0)
    assert tilt_deg(15.0, 15.0) == pytest.approx(21.1, abs=0.05)


def test_home_cost_uses_rtl_speed_and_margin() -> None:
    # 500 m at 8 m/s = 62.5 s; 20 A for 62.5 s = 347.2 mAh; x 1.3.
    assert home_mah(500.0, 20.0) == pytest.approx(451.4, abs=0.1)
    assert wind.RTL_SPEED_MS == 8.0
    assert (wind.CAPACITY_MAH, wind.RESERVE_MAH) == (10000.0, 1500.0)


def test_calm_hover_gives_no_warning() -> None:
    watch = WindWatch()
    t = flying(roll_deg=2.0, pitch_deg=-3.0)
    now = run(watch, t, 12.0)
    check = watch.check(t, now)
    assert not check.warning and check.reasons == []
    assert check.tilt_avg_deg == pytest.approx(tilt_deg(2.0, -3.0))
    assert check.tilt_warn_deg == FC_ANGLE_MAX_DEG - 10.0
    assert check.calibrated is False
    assert check.home_mah == 0.0 and check.left_mah == 8000.0


def test_tilt_near_the_limit_for_10_s_warns() -> None:
    watch = WindWatch()
    t = flying(pitch_deg=-(FC_ANGLE_MAX_DEG - 8.0))
    early = run(watch, t, 5.0)
    assert watch.check(t, early).tilt_avg_deg is None  # not 10 s yet
    now = run(watch, t, 7.0, start_ms=early + 250)
    check = watch.check(t, now)
    assert check.warning and check.reasons == ["TILT_MARGIN"]


def test_a_short_gust_does_not_warn() -> None:
    watch = WindWatch()
    calm, gust = flying(pitch_deg=-5.0), flying(pitch_deg=-28.0)
    now = run(watch, calm, 8.0)
    now = run(watch, gust, 2.0, start_ms=now + 250)
    assert "TILT_MARGIN" not in watch.check(gust, now).reasons


def test_landing_restarts_the_tilt_window() -> None:
    watch = WindWatch()
    t = flying(pitch_deg=-25.0)
    now = run(watch, t, 12.0)
    assert watch.check(t, now).warning
    on_ground = replace(t, armed=False, landed_state="on_ground", battery_current_a=0.0)
    watch.observe(replace(on_ground, attitude_mono_ms=now + 250), now + 250)
    check = watch.check(on_ground, now + 250)
    assert check.reasons == [] and check.tilt_avg_deg is None and check.home_mah is None


def test_not_enough_energy_home_warns() -> None:
    watch = WindWatch()
    # 500 m out at 20 A: about 451 mAh home + 1500 reserve > 19 % of 10000 mAh.
    t = flying(lat=FAR[0], lon=FAR[1], battery_pct=20.0)
    now = run(watch, t, 12.0)
    assert not watch.check(t, now).warning
    low = replace(t, battery_pct=19.0)
    check = watch.check(low, now)
    assert check.reasons == ["ENERGY_HOME"]
    assert check.home_mah == pytest.approx(451.4, abs=1.0)
    assert check.left_mah == 1900.0 and check.reserve_mah == 1500.0


def test_no_current_means_no_energy_estimate() -> None:
    watch = WindWatch()
    t = flying(lat=FAR[0], lon=FAR[1], battery_pct=10.0, battery_current_a=None)
    now = run(watch, t, 12.0)
    check = watch.check(t, now)
    assert check.home_mah is None and "ENERGY_HOME" not in check.reasons


def feed_wind(watch: WindWatch, t: FcTelemetry, speeds: list[float], from_deg: float) -> int:
    now = 0
    for i, speed in enumerate(speeds):
        now = i * 1000
        watch.observe(
            replace(t, wind_mono_ms=now, wind_speed_ms=speed, wind_from_deg=from_deg), now
        )
    return now


@pytest.mark.parametrize(
    ("stage", "speeds", "warns"),
    [
        ("first_flights", [3.5] * 30, False),
        ("first_flights", [4.5] * 30, True),
        ("first_flights", [3.0] * 29 + [9.0], False),  # no gust limit at the first stage
        ("until_calibrated", [7.5] * 30, False),
        ("until_calibrated", [8.5] * 30, True),
        ("until_calibrated", [7.0] * 29 + [12.5], True),  # one gust over 12 m/s
    ],
)
def test_wind_message_against_the_stage_limit(
    stage: wind.WindStage, speeds: list[float], warns: bool
) -> None:
    watch = WindWatch(stage)
    t = flying(armed=False, landed_state="on_ground")  # before takeoff too
    now = feed_wind(watch, t, speeds, 225.0)
    check = watch.check(t, now)
    assert ("WIND_LIMIT" in check.reasons) is warns
    assert check.wind_from_deg == 225.0
    assert check.wind_gust_ms == max(speeds)
    assert (check.wind_limit_ms, check.wind_gust_limit_ms) == wind.STAGES[stage]


def test_stale_or_missing_wind_is_not_shown() -> None:
    watch = WindWatch()
    t = flying()
    assert watch.check(t, 0).wind_ms is None  # EK3_DRAG_* not set: no WIND at all
    now = feed_wind(watch, t, [6.0] * 5, 0.0)
    late = now + round(wind.WIND_FRESH_S * 1000) + 1
    check = watch.check(t, late)
    assert check.wind_ms is None and "WIND_LIMIT" not in check.reasons


def test_wind_mean_covers_the_last_60_s() -> None:
    watch = WindWatch()
    t = flying()
    now = feed_wind(watch, t, [20.0] * 10 + [2.0] * 61, 90.0)
    check = watch.check(t, now)
    assert check.wind_ms == pytest.approx(2.0) and check.wind_gust_ms == 2.0


def test_each_sample_counts_once() -> None:
    """``observe`` runs on every telemetry read; a repeated ATTITUDE is not a new sample."""
    watch = WindWatch()
    t = flying(pitch_deg=-25.0)
    for now in range(0, 12_000, 50):  # 20 reads a second, ATTITUDE only at 4 Hz
        watch.observe(replace(t, attitude_mono_ms=now - now % 250), now)
    calm = flying(pitch_deg=0.0)
    for now in range(12_000, 14_000, 50):
        watch.observe(replace(calm, attitude_mono_ms=now - now % 250), now)
    avg = watch.check(calm, 14_000).tilt_avg_deg
    assert avg is not None
    assert avg == pytest.approx(25.0 * 32 / 41, abs=0.5)  # 8 s at 25 deg, 2 s level


def test_unreachable_home_is_no_estimate() -> None:
    watch = WindWatch()
    t = flying(lat=math.nan)
    now = run(watch, t, 12.0)
    assert watch.check(t, now).home_mah is None


# --- wiring: the panel's telemetry and the event log ------------------------------------


def _snap(t: FcTelemetry, now_ms: int) -> MavSnapshot:
    return MavSnapshot(
        vehicle_id="sim-01",
        connected=True,
        telemetry=replace(t, attitude_mono_ms=now_ms),
        now_mono_ms=now_ms,
        in_flight=True,
        onboard_alive=True,
        airborne_since=None,
        manual=False,
        watchdog_phase="ok",
    )


def test_panel_telemetry_carries_the_wind_check(monkeypatch: pytest.MonkeyPatch) -> None:
    core = cast(
        GroundMav, SimpleNamespace(clock=FakeClock(), vehicle=None, energy_short=lambda _: None)
    )
    loop = asyncio.new_event_loop()
    mav = ThreadSafeMav(core, loop, WindWatch("until_calibrated"))
    monkeypatch.setattr(mav, "_wind_params", lambda: None)  # no flight controller here
    t = flying(pitch_deg=-25.0)
    out = None
    for now in range(0, 12_001, 250):
        monkeypatch.setattr(mav, "snapshot", lambda now=now: _snap(t, now))
        mav.update_wind("sim-01")
        out = mav.telemetry("sim-01")
    loop.close()
    assert out is not None and out.wind is not None
    assert out.wind.reasons == ["TILT_MARGIN"] and out.wind.wind_limit_ms == 8.0


def test_events_when_the_warning_starts_and_ends(
    ground: Ground, vehicle_link: FakeVehicleLink
) -> None:
    calm = telemetry(dt.datetime(2026, 1, 1, tzinfo=dt.UTC), alt=30.0, landed="in_air")
    check = WindWatch().check(flying(), 0)
    windy = check.model_copy(update={"warning": True, "reasons": ["TILT_MARGIN"]})
    sub = ground.bus.subscribe(["vehicle"])
    service = TelemetryService(ground)
    for value in (check, windy, windy, None, windy, check):
        vehicle_link.values["sim-01"] = calm.model_copy(update={"wind": value})
        service.publish()
    seen: list[str] = []
    while not sub.queue.empty():
        event = sub.queue.get_nowait()
        assert event is not None
        seen.append(event.type)
    assert seen == ["vehicle.wind_warning", "vehicle.wind_ok"] * 2
    logged = [e for e in ground.queued_events() if e["type"].startswith("vehicle.wind")]
    assert [e["severity"] for e in logged] == ["warning", "info"] * 2
    assert logged[0]["payload"]["reasons"] == ["TILT_MARGIN"]


# --- the flight controller's parameters (B3) -------------------------------------------

READ = {
    "ATC_ANGLE_MAX": 25.0,
    "RTL_SPEED_MS": 6.0,
    "WP_SPD": 10.0,
    "BATT_CAPACITY": 8000.0,
    "BATT_LOW_MAH": 2000.0,
}


def test_read_parameters_are_used() -> None:
    with structlog.testing.capture_logs() as logs:
        p = fc_params(READ)
    assert (p.angle_max_deg, p.rtl_speed_ms, p.capacity_mah, p.reserve_mah) == (
        25.0,
        6.0,
        8000.0,
        2000.0,
    )
    assert p.fallback == ()
    assert not [e for e in logs if e["event"] == "wind.param_fallback"]
    watch = WindWatch(params=p)
    t = flying(roll_deg=2.0)
    now = run(watch, t, 12.0)
    check = watch.check(t, now)
    assert check.tilt_warn_deg == 15.0  # 25 from the flight controller, not 30
    assert check.left_mah == 0.8 * 8000.0 and check.reserve_mah == 2000.0


def test_missing_parameters_fall_back_to_the_constants_with_a_log() -> None:
    with structlog.testing.capture_logs() as logs:
        p = fc_params({"ATC_ANGLE_MAX": None, "RTL_SPEED_MS": 6.0})
    assert p == FcParams(
        rtl_speed_ms=6.0, fallback=("ATC_ANGLE_MAX", "BATT_CAPACITY", "BATT_LOW_MAH")
    )
    fallbacks = {e["param"]: e for e in logs if e["event"] == "wind.param_fallback"}
    assert set(fallbacks) == {"ATC_ANGLE_MAX", "BATT_CAPACITY", "BATT_LOW_MAH"}
    assert fallbacks["ATC_ANGLE_MAX"]["used"] == FC_ANGLE_MAX_DEG
    assert all(e["log_level"] == "warning" for e in fallbacks.values())
    assert WindWatch(params=p).tilt_warn_deg == FC_ANGLE_MAX_DEG - 10.0


def test_nothing_read_is_all_constants() -> None:
    with structlog.testing.capture_logs() as logs:
        p = fc_params({})
    assert p == FcParams()
    assert len([e for e in logs if e["event"] == "wind.param_fallback"]) == 4


@pytest.mark.parametrize(
    ("values", "fallback"),
    [
        ({"ATC_ANGLE_MAX": 5.0}, "ATC_ANGLE_MAX"),  # outside ArduCopter's 10-80
        ({"ATC_ANGLE_MAX": math.nan}, "ATC_ANGLE_MAX"),
        ({"BATT_LOW_MAH": 0.0}, "BATT_LOW_MAH"),  # no mAh failsafe: keep the constant
        ({"BATT_CAPACITY": -1.0}, "BATT_CAPACITY"),
        ({"RTL_SPEED_MS": 0.0, "WP_SPD": None}, "RTL_SPEED_MS"),
    ],
)
def test_unusable_values_fall_back(values: dict[str, float | None], fallback: str) -> None:
    assert fallback in fc_params({**READ, **values}).fallback


def test_rtl_speed_0_flies_at_wp_spd() -> None:
    p = fc_params({**READ, "RTL_SPEED_MS": 0.0})
    assert p.rtl_speed_ms == 10.0 and "RTL_SPEED_MS" not in p.fallback


async def test_parameters_are_read_from_the_flight_controller() -> None:
    """Through ``ThreadSafeMav``: read on connect with the fence check's reading."""
    async with stack() as s:
        s.fc.params["ATC_ANGLE_MAX"] = 25.0
        s.fc.params["BATT_LOW_MAH"] = 2500.0
        mav = cast(ThreadSafeMav, s.actions.mav)
        mav.update_wind(VID)  # starts the read
        await s.run(10.0)  # WP_SPD is not answered: two tries of 2 s
        mav.update_wind(VID)
        out = mav.telemetry(VID)
        assert out is not None and out.wind is not None
        assert out.wind.tilt_warn_deg == 15.0
        assert out.wind.reserve_mah == 2500.0
        assert mav.wind.params.fallback == ()


async def test_unanswered_parameters_keep_the_constants_and_retry() -> None:
    async with stack() as s:
        for name in ("ATC_ANGLE_MAX", "RTL_SPEED_MS", "BATT_CAPACITY", "BATT_LOW_MAH"):
            s.fc.params.pop(name, None)
        mav = cast(ThreadSafeMav, s.actions.mav)
        with structlog.testing.capture_logs() as logs:
            mav.update_wind(VID)
            await s.run(25.0)
            mav.update_wind(VID)  # takes the finished read
        assert mav.wind.params == FcParams()
        assert {e["param"] for e in logs if e["event"] == "wind.param_fallback"} == {
            "ATC_ANGLE_MAX",
            "RTL_SPEED_MS",
            "BATT_CAPACITY",
            "BATT_LOW_MAH",
        }
        s.fc.params["ATC_ANGLE_MAX"] = 25.0  # answers on the next try, 30 s later
        mav.update_wind(VID)
        await s.run(5.0)
        mav.update_wind(VID)
        assert mav.wind.params.angle_max_deg == FC_ANGLE_MAX_DEG  # not yet asked again
        for _ in range(16):  # the callers' telemetry, every 5 s here
            await s.run(5.0)
            mav.update_wind(VID)
        assert mav.wind.params.angle_max_deg == 25.0


async def test_a_new_connection_starts_from_the_constants_and_reads_again() -> None:
    async with stack() as s:
        s.fc.params["ATC_ANGLE_MAX"] = 25.0
        mav = cast(ThreadSafeMav, s.actions.mav)
        mav.update_wind(VID)
        await s.run(10.0)
        mav.update_wind(VID)
        assert mav.wind.params.angle_max_deg == 25.0
        mav.wind.drag_k = 0.01
        snap = mav.snapshot()
        assert snap is not None
        lost = replace(snap, connected=False)
        mav.snapshot = lambda: lost  # the link drops for one step
        mav.update_wind(VID)
        del mav.snapshot
        assert mav.wind.params == FcParams() and mav.wind.drag_k is None
        assert mav.telemetry(VID) is not None and mav.telemetry(VID).wind is None
        s.fc.params["ATC_ANGLE_MAX"] = 40.0  # another flight controller
        mav.update_wind(VID)
        await s.run(10.0)
        mav.update_wind(VID)
        assert mav.wind.params.angle_max_deg == 40.0


# --- the energy home against the wind (B4) ---------------------------------------------


def test_ground_speed_home() -> None:
    # Calm, or within what the drone can fly: RTL holds its ground speed.
    assert home_ground_speed(8.0, 15.0, 0.0, 0.0) == 8.0
    assert home_ground_speed(8.0, 15.0, 6.0, 0.0) == 8.0
    # Airspeed minus the headwind component once the airspeed is at its top.
    assert home_ground_speed(8.0, 15.0, 10.0, 0.0) == pytest.approx(5.0)
    # Crosswind: crab out of it, then what is left after the headwind.
    assert home_ground_speed(8.0, 15.0, 9.0, 9.0) == pytest.approx(3.0)
    # A tailwind never makes RTL faster than its speed.
    assert home_ground_speed(8.0, 8.0, -6.0, 0.0) == 8.0
    # Too strong: does not get home.
    assert home_ground_speed(8.0, 15.0, 14.5, 0.0) is None
    assert home_ground_speed(8.0, 15.0, 0.0, 15.0) is None


def wind_check(
    watch: WindWatch,
    t: FcTelemetry,
    speed_ms: float,
    from_deg: float,
    seconds: float = 12.0,
) -> tuple[WindCheck, int]:
    now = 0
    for i in range(round(seconds * 4) + 1):
        now = i * 250
        watch.observe(
            replace(
                t,
                attitude_mono_ms=now,
                wind_mono_ms=now - now % 1000,
                wind_speed_ms=speed_ms,
                wind_from_deg=from_deg,
            ),
            now,
        )
    return watch.check(t, now), now


# 500 m north of home: home is to the south. SIM_WIND_DIR-style "from" directions.
FROM_SOUTH, FROM_NORTH, FROM_EAST = 180.0, 0.0, 90.0


def far(**kw: object) -> FcTelemetry:
    return flying(lat=FAR[0], lon=FAR[1], groundspeed_ms=5.0, **kw)


def test_headwind_costs_more_than_the_estimate_without_wind() -> None:
    before = home_mah(500.0, 20.0)
    check, _ = wind_check(WindWatch(), far(battery_pct=60.0), 6.0, FROM_SOUTH)
    assert check.home_mah is not None and check.home_mah > before
    # No drag measured: 8 m/s through the air, 2 m/s over the ground, at full lean.
    lean = (1.0 / math.cos(math.radians(FC_ANGLE_MAX_DEG))) ** 1.5
    assert check.home_mah == pytest.approx(500.0 / 2.0 * 20.0 * lean / 3.6 * 1.3, rel=1e-3)


def test_tailwind_never_costs_less_than_the_estimate_without_wind() -> None:
    # No drag measured: RTL speed over the ground, but at full lean (the safe side).
    check, _ = wind_check(WindWatch(), far(battery_pct=60.0), 6.0, FROM_NORTH)
    lean = (1.0 / math.cos(math.radians(FC_ANGLE_MAX_DEG))) ** 1.5
    assert check.home_mah == pytest.approx(home_mah(500.0, 20.0) * lean, rel=1e-3)
    # Drag measured and little lean needed: the estimate without the wind still holds.
    watch = WindWatch()
    watch.drag_k = 0.001
    check, _ = wind_check(watch, far(battery_pct=60.0), 6.0, FROM_NORTH)
    way = distance_m(HOME.lat, HOME.lon, *FAR)
    assert check.home_mah == pytest.approx(home_mah(way, 20.0), rel=1e-4)


def test_crosswind_costs_between_tail_and_head() -> None:
    head, _ = wind_check(WindWatch(), far(battery_pct=60.0), 6.0, FROM_SOUTH)
    cross, _ = wind_check(WindWatch(), far(battery_pct=60.0), 6.0, FROM_EAST)
    tail, _ = wind_check(WindWatch(), far(battery_pct=60.0), 6.0, FROM_NORTH)
    assert tail.home_mah is not None and cross.home_mah is not None
    assert head.home_mah is not None
    assert tail.home_mah < cross.home_mah < head.home_mah
    # sqrt(8**2 - 6**2) = 5.29 m/s over the ground.
    lean = (1.0 / math.cos(math.radians(FC_ANGLE_MAX_DEG))) ** 1.5
    expected = 500.0 / math.sqrt(28.0) * 20.0 * lean / 3.6 * 1.3
    assert cross.home_mah == pytest.approx(expected, rel=1e-3)


def test_headwind_warns_where_the_estimate_without_wind_did_not() -> None:
    # 500 m out, 30 %: without the wind 451 + 1500 < 3000, so no warning before.
    t = far(battery_pct=30.0)
    calm = WindWatch()
    now = run(calm, t, 12.0)
    assert "ENERGY_HOME" not in calm.check(t, now).reasons
    check, _ = wind_check(WindWatch(), t, 6.0, FROM_SOUTH)
    assert "ENERGY_HOME" in check.reasons


def test_too_strong_to_get_home_warns_at_once() -> None:
    """With the drag measured: 9 m/s against a top airspeed of 8.7 m/s."""
    watch = WindWatch()
    watch.drag_k = math.tan(math.radians(30.0)) / 8.7**2
    check, _ = wind_check(watch, far(battery_pct=95.0), 9.0, FROM_SOUTH)
    assert "ENERGY_HOME" in check.reasons and check.home_mah is None


def test_without_drag_a_strong_headwind_is_slow_not_impossible() -> None:
    """Without the drag the top airspeed is unknown (RTL speed is only a lower bound), so
    a full battery 500 m out in 9 m/s is not sent home; a low one still is."""
    full, _ = wind_check(WindWatch(), far(battery_pct=95.0), 9.0, FROM_SOUTH)
    assert "ENERGY_HOME" not in full.reasons and full.home_mah is not None
    low, _ = wind_check(WindWatch(), far(battery_pct=50.0), 9.0, FROM_SOUTH)
    assert "ENERGY_HOME" in low.reasons


def test_one_gust_does_not_count_as_the_wind_home() -> None:
    watch = WindWatch()
    t = far(battery_pct=30.0)
    now = 0
    for i in range(60):  # 59 s calm from the south, then one 12 m/s sample
        now = i * 1000
        speed = 12.0 if i == 59 else 1.0
        watch.observe(
            replace(
                t,
                attitude_mono_ms=now,
                wind_mono_ms=now,
                wind_speed_ms=speed,
                wind_from_deg=FROM_SOUTH,
            ),
            now,
        )
    assert "ENERGY_HOME" not in watch.check(t, now).reasons


def test_hovering_at_home_in_wind_is_not_short() -> None:
    watch = WindWatch()
    watch.drag_k = 0.01  # top airspeed 7.6 m/s, less than the wind
    check, _ = wind_check(watch, flying(battery_pct=90.0), 9.0, FROM_SOUTH)
    assert "ENERGY_HOME" not in check.reasons and check.home_mah == pytest.approx(0.0)


def test_upside_down_does_not_break_the_check() -> None:
    check, _ = wind_check(WindWatch(), far(battery_pct=60.0, roll_deg=170.0), 6.0, FROM_SOUTH)
    assert check.home_mah is not None and math.isfinite(check.home_mah)


def test_measured_drag_lets_the_drone_fly_faster_than_rtl_through_the_air() -> None:
    """Holding position in 6 m/s at 10 degrees: tan(10) / 36. The top airspeed at 30
    degrees is then 6 * sqrt(tan 30 / tan 10) = 10.9 m/s, so 9 m/s of headwind leaves
    1.9 m/s over the ground instead of not getting home."""
    watch = WindWatch()
    hold = flying(lat=FAR[0], lon=FAR[1], pitch_deg=-10.0, groundspeed_ms=0.2, battery_pct=95.0)
    check, _ = wind_check(watch, hold, 6.0, FROM_SOUTH)
    assert watch.drag_k == pytest.approx(math.tan(math.radians(10.0)) / 36.0)
    assert check.home_mah is not None
    moving = WindWatch()
    moving.drag_k = watch.drag_k
    strong, _ = wind_check(moving, replace(hold, groundspeed_ms=5.0), 9.0, FROM_SOUTH)
    assert moving.drag_k == watch.drag_k  # not holding position: not measured again
    assert strong.home_mah is not None
    top = 6.0 * math.sqrt(math.tan(math.radians(30.0)) / math.tan(math.radians(10.0)))
    assert strong.home_mah == pytest.approx(
        500.0
        / (top - 9.0)
        * 20.0
        * math.cos(math.radians(10.0)) ** 1.5
        / math.cos(math.radians(30.0)) ** 1.5
        / 3.6
        * 1.3,
        rel=1e-2,
    )


def test_landing_forgets_the_drag() -> None:
    watch = WindWatch()
    hold = flying(pitch_deg=-10.0, groundspeed_ms=0.2)
    wind_check(watch, hold, 6.0, FROM_SOUTH)
    assert watch.drag_k is not None
    watch.observe(replace(hold, armed=False, landed_state="on_ground"), 20_000)
    assert watch.drag_k is None


def test_the_energy_home_never_warns_later_than_before() -> None:
    """Whatever the wind, the parameters read and the drag: wherever the estimate
    without the wind (the constants, x 1.3) warned, the check still warns."""
    rng = random.Random(20261031)  # noqa: S311 - test data
    for _ in range(400):
        pct = rng.uniform(5.0, 60.0)
        current = rng.uniform(5.0, 40.0)
        lat = HOME.lat + rng.uniform(-0.008, 0.008)
        lon = HOME.lon + rng.uniform(-0.008, 0.008)
        params = FcParams(
            angle_max_deg=rng.uniform(15.0, 45.0),
            rtl_speed_ms=rng.uniform(3.0, 15.0),
            capacity_mah=rng.uniform(5000.0, 20000.0),
            reserve_mah=rng.uniform(100.0, 3000.0),
            fallback=(),
        )
        watch = WindWatch(params=params)
        if rng.random() < 0.5:
            watch.drag_k = rng.uniform(0.0005, 0.02)
        t = flying(
            lat=lat,
            lon=lon,
            battery_pct=pct,
            battery_current_a=current,
            pitch_deg=-rng.uniform(0.0, 25.0),
            groundspeed_ms=3.0,
        )
        check, _ = wind_check(watch, t, rng.uniform(0.0, 16.0), rng.uniform(0.0, 360.0))
        way = distance_m(HOME.lat, HOME.lon, lat, lon)
        before = home_mah(way, current) + wind.RESERVE_MAH > pct / 100.0 * wind.CAPACITY_MAH
        if before:
            assert "ENERGY_HOME" in check.reasons, (pct, current, way, params)


# --- the battery guard flies home on ENERGY_HOME ----------------------------------------


async def test_energy_home_starts_the_battery_guards_fly_home() -> None:
    async with stack() as s:
        s.fly()
        s.fc.battery_pct = 80.0  # far above the level
        await s.run(1.0)
        s.core.energy_short(True)
        await s.run(1.0)
        assert s.fc.mode in HOME_MODES
        latch = s.latches.get(VID)
        assert latch is not None and latch.action == "battery_rtl"
        guard = [e for e in s.g.events if e[0] == "vehicle.battery_rtl" and "alarm" in e[2]]
        assert [e[2]["reason"] for e in guard] == ["energy_home"]


async def test_a_stale_energy_home_never_flies_home() -> None:
    """When the callers of ``telemetry`` stop, their last ENERGY_HOME runs out."""
    async with stack() as s:
        s.fly()
        s.fc.battery_pct = 80.0
        await s.run(1.0)
        s.core.energy_short(True)
        s.core._energy_short_at = s.clock.monotonic() - 10.0  # pyright: ignore[reportPrivateUsage]
        await s.run(2.0)
        assert s.fc.mode not in HOME_MODES and s.latches.get(VID) is None


async def test_energy_home_leaves_the_pilots_take_over_alone() -> None:
    async with stack() as s:
        s.fly(mode=CopterMode.LOITER)
        s.fc.battery_pct = 80.0
        await s.run(1.0)
        s.actions.takeover(VID, client_action_id="t1", by=PILOT)
        await s.run(1.0)
        for _ in range(4):
            s.core.energy_short(True)
            await s.run(0.5)
        assert s.fc.mode == CopterMode.LOITER
        latch = s.latches.get(VID)
        assert latch is None or latch.action != "battery_rtl"


async def test_energy_ok_leaves_the_percentage_level_as_before() -> None:
    async with stack(rtl_pct=20.0) as s:
        s.fly()
        await s.run(1.0)
        s.core.energy_short(False)
        s.fc.battery_pct = 19.0
        await s.run(1.5)
        assert s.fc.mode in HOME_MODES


async def test_stop_and_take_over_after_an_energy_flight_home_keep_the_level() -> None:
    """ENERGY_HOME flies home at 60 %, the pilot presses STOP and takes over: at the
    percentage level the battery guard still flies home (the energy latch under the STOP
    blocks only the energy trigger)."""
    async with stack(rtl_pct=20.0) as s:
        s.fly()
        s.fc.battery_pct = 60.0
        await s.run(1.0)
        s.core.energy_short(True)
        await s.run(1.0)
        assert s.fc.mode in HOME_MODES
        s.stop()
        await s.run(1.0)
        latch = s.latches.get(VID)
        assert latch is not None and (latch.action, latch.underlying) == ("stop", "battery_rtl")
        s.actions.takeover(VID, client_action_id="t-energy", by=PILOT)
        await s.run(1.0)
        assert s.fc.mode == CopterMode.LOITER
        for _ in range(4):  # still short: the energy trigger stays blocked under the STOP
            s.core.energy_short(True)
            await s.run(0.5)
        assert s.fc.mode == CopterMode.LOITER
        s.fc.battery_pct = 19.0
        await s.run(1.5)
        assert s.fc.mode in HOME_MODES
        guard = [e for e in s.g.events if e[0] == "vehicle.battery_rtl" and "alarm" in e[2]]
        assert [e[2]["reason"] for e in guard] == ["energy_home", "battery_level"]


async def test_a_level_latch_under_a_stop_still_blocks_the_level() -> None:
    """As before: the percentage level's own latch under a STOP waits for Resume."""
    async with stack(rtl_pct=20.0) as s:
        s.fly()
        s.fc.battery_pct = 19.0
        await s.run(1.5)
        assert s.fc.mode in HOME_MODES
        s.stop()
        await s.run(1.0)
        s.fc.battery_pct = 18.0
        await s.run(1.5)
        assert s.fc.mode not in HOME_MODES
        guard = [e for e in s.g.events if e[0] == "vehicle.battery_rtl" and "alarm" in e[2]]
        assert len(guard) == 1


async def test_resume_and_the_alarm_name_the_energy_reason() -> None:
    async with stack() as s:
        s.fly()
        s.fc.battery_pct = 80.0
        await s.run(1.0)
        s.core.energy_short(True)
        await s.run(1.0)
        with pytest.raises(ActionRefusedError) as err:
            s.resume()
        assert err.value.code == "BATTERY_RTL_ACTIVE"
        assert "vinden" in str(err.value.detail_sv)
        alarms = [e[2] for e in s.g.events if e[0] == "alarm.BATTERY_RTL"]
        assert [a["reason"] for a in alarms] == ["energy_home"]
        assert "vinden" in alarms[0]["message_sv"]


async def test_the_level_alarm_still_says_the_battery_is_low() -> None:
    async with stack(rtl_pct=20.0) as s:
        s.fly()
        s.fc.battery_pct = 19.0
        await s.run(1.5)
        with pytest.raises(ActionRefusedError) as err:
            s.resume()
        assert "Batteriet är lågt" in str(err.value.detail_sv)
        alarms = [e[2] for e in s.g.events if e[0] == "alarm.BATTERY_RTL"]
        assert [a["reason"] for a in alarms] == ["battery_level"]


# --- which estimates may fly the drone home (energy_rtl) --------------------------------


def test_the_wind_estimate_without_drag_only_warns() -> None:
    check, _ = wind_check(watch := WindWatch(), far(battery_pct=30.0), 6.0, FROM_SOUTH)
    assert "ENERGY_HOME" in check.reasons and not watch.energy_rtl


def test_the_wind_estimate_with_measured_drag_flies_home() -> None:
    watch = WindWatch()
    watch.drag_k = math.tan(math.radians(10.0)) / 36.0  # top airspeed 10.9 m/s
    check, _ = wind_check(watch, far(battery_pct=30.0), 9.0, FROM_SOUTH)
    assert "ENERGY_HOME" in check.reasons and watch.energy_rtl


def test_too_strong_to_get_home_only_warns() -> None:
    watch = WindWatch()
    watch.drag_k = math.tan(math.radians(30.0)) / 8.7**2
    check, _ = wind_check(watch, far(battery_pct=95.0), 9.0, FROM_SOUTH)
    assert "ENERGY_HOME" in check.reasons and check.home_mah is None
    assert not watch.energy_rtl


def test_the_estimate_with_the_fc_numbers_flies_home() -> None:
    # 500 m, 20 A, no wind: 451 mAh + 1500 reserve > 15 % of 10000 mAh.
    watch = WindWatch()
    t = far(battery_pct=15.0)
    now = run(watch, t, 12.0)
    assert "ENERGY_HOME" in watch.check(t, now).reasons and watch.energy_rtl
    t = far(battery_pct=60.0)
    now = run(watch, t, 12.0, start_ms=now + 250)
    assert "ENERGY_HOME" not in watch.check(t, now).reasons and not watch.energy_rtl


def test_the_fc_capacity_decides_the_fly_home() -> None:
    """The estimate with the constants (10000 mAh) only warns; the flight controller's
    BATT_CAPACITY decides whether it flies home."""
    t = far(battery_pct=25.0)  # 2500 mAh by the constant: 451 + 1500 fits
    small = WindWatch(params=FcParams(capacity_mah=7000.0, fallback=()))  # 1750 mAh
    now = run(small, t, 12.0)
    assert "ENERGY_HOME" in small.check(t, now).reasons and small.energy_rtl
    large = WindWatch(params=FcParams(capacity_mah=16000.0, fallback=()))
    t = far(battery_pct=15.0)  # 1500 mAh by the constant: short, 2400 by the FC: fits
    now = run(large, t, 12.0)
    assert "ENERGY_HOME" in large.check(t, now).reasons and not large.energy_rtl


@pytest.mark.parametrize("groundspeed", [0.0, math.nan])
def test_no_ground_speed_is_not_holding_position(groundspeed: float) -> None:
    watch = WindWatch()
    hold = flying(pitch_deg=-10.0, groundspeed_ms=groundspeed)
    wind_check(watch, hold, 6.0, FROM_SOUTH)
    assert watch.drag_k is None


@pytest.mark.parametrize(
    ("values", "fallback"),
    [
        ({"RTL_SPEED_MS": 25.0}, "RTL_SPEED_MS"),
        ({"RTL_SPEED_MS": 0.0, "WP_SPD": 30.0}, "RTL_SPEED_MS"),
        ({"BATT_CAPACITY": 5e6}, "BATT_CAPACITY"),
        ({"BATT_LOW_MAH": 9000.0}, "BATT_LOW_MAH"),  # above the 8000 mAh capacity
    ],
)
def test_absurd_values_fall_back(values: dict[str, float | None], fallback: str) -> None:
    assert fallback in fc_params({**READ, **values}).fallback


def test_the_wind_direction_is_the_mean_of_the_window() -> None:
    watch = WindWatch()
    t = flying()
    now = 0
    for i, frm in enumerate((350.0, 10.0, 350.0, 10.0)):
        now = i * 1000
        watch.observe(replace(t, wind_mono_ms=now, wind_speed_ms=5.0, wind_from_deg=frm), now)
    assert watch.check(t, now).wind_from_deg == pytest.approx(0.0, abs=1e-6)


async def test_telemetry_reads_only() -> None:
    """``telemetry`` neither reads parameters nor hands ENERGY_HOME on."""
    async with stack() as s:
        s.fly()
        s.fc.battery_pct = 5.0
        mav = cast(ThreadSafeMav, s.actions.mav)
        for _ in range(20):
            assert mav.telemetry(VID) is not None
            await s.run(0.25)
        assert mav._params_read is None  # pyright: ignore[reportPrivateUsage]
        assert s.core._energy_short_at is None  # pyright: ignore[reportPrivateUsage]
