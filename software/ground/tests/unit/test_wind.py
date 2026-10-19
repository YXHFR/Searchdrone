"""The ground station's wind warning (``sd_ground.services.wind``): a warning only."""

from __future__ import annotations

import asyncio
import datetime as dt
import math
from dataclasses import replace
from types import SimpleNamespace
from typing import cast

import pytest
from conftest import FakeVehicleLink, telemetry  # pyright: ignore[reportMissingImports]

from sd_ground.context import Ground
from sd_ground.mav.thread import GroundMav, MavSnapshot, ThreadSafeMav
from sd_ground.services import wind
from sd_ground.services.telemetry import TelemetryService
from sd_ground.services.wind import WindWatch, home_mah, tilt_deg
from sd_mav.vehicle import FcTelemetry
from sd_safety.constants import FC_ANGLE_MAX_DEG
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
    core = cast(GroundMav, SimpleNamespace(clock=FakeClock()))
    loop = asyncio.new_event_loop()
    mav = ThreadSafeMav(core, loop, WindWatch("until_calibrated"))
    t = flying(pitch_deg=-25.0)
    out = None
    for now in range(0, 12_001, 250):
        monkeypatch.setattr(mav, "snapshot", lambda now=now: _snap(t, now))
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
