"""``to_telemetry`` forwards ATTITUDE only while it is fresh."""

from __future__ import annotations

import datetime as dt
import math

import pytest

from sd_ground.mav.thread import ATTITUDE_FRESH_S, MavSnapshot, to_telemetry
from sd_mav.vehicle import FcTelemetry

NOW = dt.datetime(2026, 1, 1, tzinfo=dt.UTC)


def _snap(t: FcTelemetry, now_mono_ms: int) -> MavSnapshot:
    return MavSnapshot(
        vehicle_id="sim-01",
        connected=True,
        telemetry=t,
        now_mono_ms=now_mono_ms,
        in_flight=False,
        onboard_alive=True,
        airborne_since=None,
        manual=False,
        watchdog_phase="ok",
    )


def test_fresh_attitude_is_forwarded_in_degrees() -> None:
    t = FcTelemetry(attitude_mono_ms=10_000, roll_deg=-12.5, pitch_deg=4.0, yaw_deg=-90.0)
    att = to_telemetry(_snap(t, 10_200), NOW).attitude
    assert att is not None
    assert (att.roll_deg, att.pitch_deg, att.yaw_deg) == (-12.5, 4.0, 270.0)


def test_no_attitude_before_the_first_message() -> None:
    assert to_telemetry(_snap(FcTelemetry(), 10_000), NOW).attitude is None


def test_stale_attitude_is_dropped_not_frozen() -> None:
    t = FcTelemetry(attitude_mono_ms=10_000, roll_deg=30.0)
    late = 10_000 + int(ATTITUDE_FRESH_S * 1000) + 1
    assert to_telemetry(_snap(t, late), NOW).attitude is None


def test_rounding_at_the_edges_stays_inside_the_schema() -> None:
    t = FcTelemetry(attitude_mono_ms=0, roll_deg=180.0000001, pitch_deg=90.0000001, yaw_deg=-1e-17)
    att = to_telemetry(_snap(t, 0), NOW).attitude
    assert att is not None
    assert att.roll_deg == 180.0 and att.pitch_deg == 90.0 and 0.0 <= att.yaw_deg < 360.0


@pytest.mark.parametrize("bad", [math.nan, math.inf, -math.inf])
@pytest.mark.parametrize("axis", ["roll_deg", "pitch_deg", "yaw_deg"])
def test_non_finite_attitude_is_dropped_not_clamped(axis: str, bad: float) -> None:
    t = FcTelemetry(attitude_mono_ms=0, **{axis: bad})
    assert to_telemetry(_snap(t, 0), NOW).attitude is None


@pytest.mark.parametrize(
    ("roll", "pitch"), [(200.0, 0.0), (-181.0, 0.0), (0.0, 95.0), (0.0, -120.0)]
)
def test_impossible_attitude_is_dropped_not_clamped(roll: float, pitch: float) -> None:
    t = FcTelemetry(attitude_mono_ms=0, roll_deg=roll, pitch_deg=pitch)
    assert to_telemetry(_snap(t, 0), NOW).attitude is None
