"""SpoofDetector (S1 jump, S2 height) and GpsHealth on synthetic series."""

from __future__ import annotations

import random

import pytest
from gate_fixtures import latlon

from sd_safety.gnss import GnssSample, GpsHealth, SpoofDetector

STEP_MS = 200  # 5 Hz


def sample(
    t_ms: int,
    north_off: float = 0.0,
    alt_off: float = 0.0,
    fix: int = 3,
    noise: float = 0.0,
    rnd: random.Random | None = None,
) -> GnssSample:
    ekf = latlon(0, 0)
    n = north_off + (rnd.gauss(0, noise) if rnd else 0.0)
    e = rnd.gauss(0, noise) if rnd else 0.0
    gps = latlon(e, n)
    return GnssSample(
        mono_ms=t_ms,
        fix_type=fix,
        satellites=14,
        hdop=0.8,
        gps_lat=gps[0],
        gps_lon=gps[1],
        gps_alt_msl_m=620.0 + alt_off + (rnd.gauss(0, noise) if rnd else 0.0),
        ekf_lat=ekf[0],
        ekf_lon=ekf[1],
        ekf_alt_msl_m=624.0,
    )


def feed(d: SpoofDetector, series: list[GnssSample]) -> list[tuple[int, str]]:
    out = []
    for s in series:
        f = d.update(s, armed=True)
        if f is not None:
            out.append((s.mono_ms, f.kind))
    return out


def baseline(seconds: float = 12.0) -> list[GnssSample]:
    return [sample(t) for t in range(0, int(seconds * 1000), STEP_MS)]


def test_spoof_jump_30_m_for_2_s() -> None:
    d = SpoofDetector()
    series = baseline() + [sample(12_000 + t, north_off=31) for t in range(0, 4000, STEP_MS)]
    found = feed(d, series)
    assert found == [(14_000, "jump")]
    assert d.finding is not None and d.finding.value_m == pytest.approx(31, abs=0.2)


def test_spoof_jump_29_m_never() -> None:
    d = SpoofDetector()
    assert (
        feed(d, baseline() + [sample(12_000 + t, north_off=29) for t in range(0, 20_000, 200)])
        == []
    )


def test_spoof_short_glitch_never() -> None:
    d = SpoofDetector()
    glitch = [sample(12_000 + t, north_off=45) for t in range(0, 1800, STEP_MS)]
    back = [sample(14_000 + t) for t in range(0, 5000, STEP_MS)]
    assert feed(d, baseline() + glitch + back) == []


def test_spoof_15_m_glitch_never() -> None:
    d = SpoofDetector()
    glitch = [sample(12_000 + t, north_off=15) for t in range(0, 5000, STEP_MS)]
    assert feed(d, baseline() + glitch) == []


def test_spoof_noise_never() -> None:
    rnd = random.Random(7)  # noqa: S311 - test data
    d = SpoofDetector()
    series = [sample(t, noise=3.0, rnd=rnd) for t in range(0, 300_000, STEP_MS)]
    assert feed(d, series) == []


def test_spoof_altitude_25_m_for_3_s() -> None:
    d = SpoofDetector()
    ramp = [sample(12_000 + t, alt_off=26) for t in range(0, 5000, STEP_MS)]
    assert feed(d, baseline() + ramp) == [(15_000, "altitude")]


def test_spoof_altitude_baseline_takes_the_takeoff_offset() -> None:
    """A GPS that always reads 30 m above the barometer is no spoof."""
    d = SpoofDetector()
    series = [sample(t, alt_off=30) for t in range(0, 60_000, STEP_MS)]
    assert feed(d, series) == []


def test_spoof_needs_a_3d_fix() -> None:
    d = SpoofDetector()
    series = baseline() + [
        sample(12_000 + t, north_off=100, alt_off=100, fix=2) for t in range(0, 10_000, STEP_MS)
    ]
    assert feed(d, series) == []


def test_spoof_fires_once_and_resets_after_landing() -> None:
    d = SpoofDetector()
    series = baseline() + [sample(12_000 + t, north_off=50) for t in range(0, 10_000, STEP_MS)]
    assert len(feed(d, series)) == 1
    d.reset()
    assert d.finding is None
    later = [sample(100_000 + s.mono_ms, north_off=50) for s in baseline()]
    assert len(feed(d, later)) == 1


def test_spoof_slow_ramp_with_altitude() -> None:
    """GP4: 1 m/s north and up; S2 catches it within 30 s of the start."""
    d = SpoofDetector()
    ramp = [
        sample(12_000 + t, north_off=t / 1000 * 0.0, alt_off=t / 1000)
        for t in range(0, 40_000, STEP_MS)
    ]
    found = feed(d, baseline() + ramp)
    assert found and found[0][1] == "altitude" and found[0][0] - 12_000 <= 30_000


def test_gps_health() -> None:
    h = GpsHealth()
    events = []
    for t in range(0, 30_000, 100):
        if t < 5000:
            fix, sats = 3, 12
        elif t < 10_000:
            fix, sats = 3, 5  # weak
        elif t < 15_000:
            fix, sats = 1, 0  # lost
        else:
            fix, sats = 3, 12
        new = h.update(fix_type=fix, satellites=sats, hdop=0.9, now_ms=t)
        if new:
            events.append((t, new))
    assert events == [(8000, "weak"), (12_000, "lost"), (20_000, "ok")]


def test_gps_health_hdop() -> None:
    h = GpsHealth()
    got = [h.update(fix_type=3, satellites=12, hdop=3.0, now_ms=t) for t in range(0, 3100, 100)]
    assert got[-1] == "weak"
