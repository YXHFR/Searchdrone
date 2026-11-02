"""The AI's "till hemgräns" minutes: seconds to the home limit from the ground's own
telemetry (level, measured current, the pack's capacity), None without them."""

from __future__ import annotations

import dataclasses
import math

import pytest

from sd_ground.mav.thread import MavSnapshot
from sd_ground.services.situation import (
    _vehicle,  # pyright: ignore[reportPrivateUsage]
    battery_to_reserve_s,
)
from sd_ground.services.wind import CAPACITY_MAH
from sd_mav.vehicle import FcTelemetry

FLYING = FcTelemetry(
    armed=True,
    landed_state="in_air",
    alt_rel_home_m=40.0,
    battery_pct=60.0,
    battery_current_a=30.0,
)


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
    # 30 % of the pack at 30 A: 0.30 * capacity mAh * 3.6 s / 30 A.
    want = round(0.30 * CAPACITY_MAH * 3.6 / 30.0)
    assert battery_to_reserve_s(60.0, 30.0, 30.0) == want
    if CAPACITY_MAH == 10_000:
        assert want == 360  # 6 min on the 10 Ah pack


def test_at_or_below_the_limit_it_is_zero() -> None:
    assert battery_to_reserve_s(30.0, 30.0, 30.0) == 0
    assert battery_to_reserve_s(25.0, 30.0, 30.0) == 0


@pytest.mark.parametrize(
    ("pct", "rtl", "current"),
    [
        (None, 30.0, 30.0),
        (60.0, None, 30.0),
        (60.0, 30.0, None),
        (60.0, 30.0, 0.0),  # no current measured
        (60.0, 30.0, 0.4),  # not a flight's current
        (60.0, 30.0, -5.0),
        (math.nan, 30.0, 30.0),
        (60.0, 30.0, math.inf),
    ],
)
def test_without_usable_data_it_is_none(
    pct: float | None, rtl: float | None, current: float | None
) -> None:
    assert battery_to_reserve_s(pct, rtl, current) is None


def test_the_ai_sees_the_seconds_in_flight() -> None:
    v = _vehicle(_snap(FLYING), None, True, None, 30.0)
    assert v.battery_to_reserve_s == battery_to_reserve_s(60.0, 30.0, 30.0)


@pytest.mark.parametrize(
    "t",
    [
        dataclasses.replace(FLYING, battery_current_a=None),
        dataclasses.replace(FLYING, battery_pct=None),
        dataclasses.replace(FLYING, landed_state="on_ground", alt_rel_home_m=0.0),
        dataclasses.replace(FLYING, armed=False),
    ],
)
def test_without_a_current_a_level_or_a_flight_the_ai_sees_none(t: FcTelemetry) -> None:
    assert _vehicle(_snap(t), None, True, None, 30.0).battery_to_reserve_s is None


def test_without_a_link_or_a_limit_the_ai_sees_none() -> None:
    assert (
        _vehicle(_snap(FLYING, connected=False), None, True, None, 30.0).battery_to_reserve_s
        is None
    )
    assert _vehicle(_snap(FLYING), None, True, None).battery_to_reserve_s is None
    assert _vehicle(None, None, True, None, 30.0).battery_to_reserve_s is None
