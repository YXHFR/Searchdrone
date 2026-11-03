"""The AI's "till hemgräns" minutes: seconds to the home limit from the ground's own
telemetry (level, the wind check's mean current, the pack's capacity), None without them."""

from __future__ import annotations

import asyncio
import dataclasses
import datetime as dt
import math
from types import SimpleNamespace
from typing import cast

import pytest

from sd_ground.mav.thread import GroundMav, MavSnapshot, ThreadSafeMav
from sd_ground.services.situation import (
    _vehicle,  # pyright: ignore[reportPrivateUsage]
    battery_to_reserve_s,
    build_situation,
)
from sd_ground.services.wind import CAPACITY_MAH, TILT_WINDOW_S, BatteryDrain, FcParams, WindWatch
from sd_mav.vehicle import FcTelemetry
from sd_schemas.api import Charter, Mission
from sd_schemas.base import LatLon, LatLonAlt, new_id
from sd_schemas.charter import AutonomyMode, MissionCharter
from sd_schemas.mission import MissionState
from sd_schemas.safety import SafetyLimits, home_limit_pct
from sd_testing import FakeClock

NOW = dt.datetime(2026, 1, 1, tzinfo=dt.UTC)

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


def test_nan_currents_are_left_out_of_the_mean() -> None:
    watch = WindWatch()
    now = _feed(watch, dataclasses.replace(FLYING, battery_current_a=math.nan), 0, 2.0)
    assert watch.drain(now) is None
    now = _feed(watch, FLYING, now + 250, 2.0)
    drain = watch.drain(now)
    assert drain is not None and drain.current_a == 30.0


def _mav(snap: MavSnapshot | None) -> ThreadSafeMav:
    core = cast(
        GroundMav, SimpleNamespace(clock=FakeClock(), vehicle=None, energy_short=lambda _: None)
    )
    mav = ThreadSafeMav(core, asyncio.new_event_loop(), WindWatch())
    mav._wind_params = lambda: None  # type: ignore[method-assign]  # no flight controller
    mav.snapshot = lambda: snap  # type: ignore[method-assign]
    return mav


def _flown(mav: ThreadSafeMav, snap: MavSnapshot) -> MavSnapshot:
    """Feed the wind check 2 s of flight; returns the last snapshot."""
    for now in range(snap.now_mono_ms, snap.now_mono_ms + 2_001, 250):
        snap = dataclasses.replace(
            snap,
            now_mono_ms=now,
            telemetry=dataclasses.replace(snap.telemetry, attitude_mono_ms=now),
        )
        mav.snapshot = lambda snap=snap: snap  # type: ignore[method-assign]
        mav.update_wind("sim-01")
    return snap


def test_the_thread_hands_out_the_drain_of_this_vehicle_and_link_only() -> None:
    mav = _mav(_snap(FLYING))
    snap = _flown(mav, _snap(FLYING))
    drain = mav.battery_drain("sim-01")
    assert drain is not None and drain.current_a == 30.0
    assert mav.battery_drain("sim-02") is None  # another vehicle
    mav.snapshot = lambda: dataclasses.replace(snap, connected=False)  # type: ignore[method-assign]
    assert mav.battery_drain("sim-01") is None  # no link
    mav.snapshot = lambda: None  # type: ignore[method-assign]
    assert mav.battery_drain("sim-01") is None
    # A reconnect (or reboot) the wind check has not taken yet: the window is the old link's.
    rebooted = dataclasses.replace(
        snap,
        telemetry=dataclasses.replace(
            snap.telemetry, fc_link_epoch=snap.telemetry.fc_link_epoch + 1
        ),
    )
    mav.snapshot = lambda: rebooted  # type: ignore[method-assign]
    assert mav.battery_drain("sim-01") is None
    mav.loop.close()


def _charter_row(mission: Mission, rtl_pct: float) -> Charter:
    body = MissionCharter(
        charter_id=new_id(),
        mission_id=mission.id,
        vehicle_id=mission.vehicle_id,
        version=1,
        area={
            "type": "Polygon",
            "coordinates": [
                [[149.165, -35.364], [149.166, -35.364], [149.166, -35.363], [149.165, -35.364]]
            ],
        },
        home=LatLonAlt(lat=-35.363261, lon=149.165230, alt_amsl_m=584.0),
        pilot_position=LatLon(lat=-35.363261, lon=149.165230),
        battery_rtl_pct=rtl_pct,
        allowed_skills=["goto"],
        autonomy_mode=AutonomyMode.WITHIN_CHARTER,
        sectors=[],
        fence_hash=None,
        approved_by=None,
        approved_at=None,
    )
    return Charter(
        id=body.charter_id,
        mission_id=mission.id,
        version=1,
        status="approved",
        body=body,
        proposed_by="operator:demo",
        approved_by=None,
        approved_at=None,
        install_status="installed",
        fence_hash=None,
        fence_verified_at=None,
        approval_id=None,
        created_at=NOW,
    )


@pytest.mark.parametrize(("charter_pct", "want_s"), [(None, 360), (40.0, 240), (20.0, 360)])
def test_the_situation_counts_down_to_the_charters_level(
    charter_pct: float | None, want_s: int
) -> None:
    mission = Mission(
        id=new_id(),
        vehicle_id="sim-01",
        title="test",
        status=MissionState.ACTIVE,
        paused_by=None,
        returning_cause=None,
        state_seq=1,
        limits=SafetyLimits(battery_rtl_pct=30.0),
        active_charter_id=None,
        autonomy_mode=None,
        ai_enabled=True,
        created_by="operator:demo",
        created_at=NOW,
        activated_at=None,
        landed_at=None,
        closed_at=None,
        closed_reason=None,
        notes=None,
    )
    s = build_situation(
        mission=mission,
        charter=None if charter_pct is None else _charter_row(mission, charter_pct),
        pending_proposal=None,
        pending_approvals=[],
        clearance=None,
        now=NOW,
        now_ms=1_000,
        latch=None,
        mav=_snap(FLYING),
        snapshot=None,
        sdlink_up=True,
        drain=DRAIN,
        active_command=None,
        sectors=[],
        annotations=[],
        recent_events=[],
    )
    # 60 % now: 30 % above the mission's 30 %, or 20 % above a charter's 40 %.
    assert s.vehicle.battery_to_reserve_s == want_s
