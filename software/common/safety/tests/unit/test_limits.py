"""Limits, the outer cap they may be raised to, and the flight controller's fence."""

from __future__ import annotations

import ast
from pathlib import Path

import pytest
from gate_fixtures import CHARTER

from sd_safety import constants, limits
from sd_safety.limits import (
    ABSOLUTE_MAX_ALT_AGL_M,
    ABSOLUTE_MAX_RANGE_FROM_PILOT_M,
    ABSOLUTE_MAX_SPEED_MS,
    ABSOLUTE_MIN_ALT_AGL_M,
    ABSOLUTE_MIN_BATTERY_RTL_PCT,
    DEFAULT_CAPS,
    DEFAULT_LIMITS,
    charter_violations,
    check_limits,
    fc_area_max_from_home_m,
    fc_fence_params,
)
from sd_schemas.safety import LimitCaps, SafetyLimits

REPO = Path(__file__).resolve().parents[5]


def test_owner_standard_values() -> None:
    assert SafetyLimits() == DEFAULT_LIMITS
    assert (DEFAULT_LIMITS.max_alt_agl_m, DEFAULT_LIMITS.max_speed_ms) == (120, 15)
    assert (DEFAULT_LIMITS.max_range_from_pilot_m, DEFAULT_LIMITS.battery_rtl_pct) == (500, 15)
    assert (ABSOLUTE_MAX_ALT_AGL_M, ABSOLUTE_MAX_SPEED_MS) == (120, 15)
    assert (ABSOLUTE_MAX_RANGE_FROM_PILOT_M, ABSOLUTE_MIN_BATTERY_RTL_PCT) == (500, 15)
    assert ABSOLUTE_MIN_ALT_AGL_M == 5


def test_limits_bounds_match() -> None:
    """sd_schemas cannot import sd_safety, so its lowest height must equal ours."""
    assert SafetyLimits.model_fields["max_alt_agl_m"].metadata[0].gt == ABSOLUTE_MIN_ALT_AGL_M
    assert LimitCaps().min_alt_agl_m == ABSOLUTE_MIN_ALT_AGL_M


def test_default_caps_are_the_owners_outer_cap() -> None:
    assert (DEFAULT_CAPS.max_alt_agl_m, DEFAULT_CAPS.max_speed_ms) == (300, 25)
    assert DEFAULT_CAPS.max_range_from_pilot_m == 2000
    assert check_limits(DEFAULT_LIMITS, DEFAULT_CAPS) == []


def test_raising_up_to_the_cap_is_allowed() -> None:
    top = SafetyLimits(max_alt_agl_m=300, max_speed_ms=25, max_range_from_pilot_m=2000)
    assert check_limits(top, DEFAULT_CAPS) == []


@pytest.mark.parametrize(
    ("field", "value"),
    [
        ("max_alt_agl_m", 300.1),
        ("max_speed_ms", 25.1),
        ("max_range_from_pilot_m", 2000.1),
    ],
)
def test_above_the_cap(field: str, value: float) -> None:
    out = check_limits(SafetyLimits(**{field: value}), DEFAULT_CAPS)
    assert [(v.field, v.code) for v in out] == [(field, "LIMIT_ABOVE_ABSOLUTE")]
    assert out[0].value == value


def test_battery_floor_of_the_cap() -> None:
    caps = LimitCaps(min_battery_rtl_pct=15)
    assert check_limits(SafetyLimits(battery_rtl_pct=15), caps) == []
    loose = SafetyLimits.model_construct(**{**DEFAULT_LIMITS.model_dump(), "battery_rtl_pct": 14.0})
    out = check_limits(loose, caps)
    assert [(v.field, v.code, v.limit) for v in out] == [
        ("battery_rtl_pct", "LIMIT_ABOVE_ABSOLUTE", 15)
    ]


def test_height_limit_must_stay_above_the_lowest_height() -> None:
    caps = LimitCaps(min_alt_agl_m=50)
    out = check_limits(SafetyLimits(max_alt_agl_m=50), caps)
    assert [(v.field, v.code) for v in out] == [("max_alt_agl_m", "VALIDATION")]
    assert check_limits(SafetyLimits(max_alt_agl_m=51), caps) == []


def test_fence_the_flight_controller_cannot_hold() -> None:
    wide = LimitCaps(max_alt_agl_m=5000, max_speed_ms=100, max_range_from_pilot_m=50_000)
    out = check_limits(
        SafetyLimits(max_alt_agl_m=996, max_speed_ms=36, max_range_from_pilot_m=9971), wide
    )
    assert {(v.field, v.code) for v in out} == {
        ("FENCE_ALT_MAX", "VALIDATION"),
        ("FENCE_RADIUS", "VALIDATION"),
        ("LOIT_SPEED_MS", "VALIDATION"),
    }
    slow = check_limits(SafetyLimits(max_speed_ms=0.1), DEFAULT_CAPS)
    assert [(v.field, v.limit) for v in slow] == [("LOIT_SPEED_MS", 0.2)]


def test_absolute_caps_match_fc_params() -> None:
    """The flight controller's fence sits just outside the standard limits, never inside."""
    fence = fc_fence_params(DEFAULT_LIMITS)
    expected = constants.FC_SAFETY_EXPECTED
    assert fence["FENCE_ALT_MAX"] == expected["FENCE_ALT_MAX"] == 125
    assert fence["FENCE_RADIUS"] == expected["FENCE_RADIUS"] == 530
    assert fence["LOIT_SPEED_MS"] == expected["LOIT_SPEED_MS"] == 15
    assert expected["FENCE_ALT_MAX"] >= ABSOLUTE_MAX_ALT_AGL_M + constants.FENCE_ALT_MARGIN_M
    assert (
        expected["FENCE_RADIUS"]
        >= ABSOLUTE_MAX_RANGE_FROM_PILOT_M + constants.FENCE_RADIUS_MARGIN_M
    )
    assert expected["LOIT_SPEED_MS"] >= ABSOLUTE_MAX_SPEED_MS
    assert expected["FENCE_MARGIN"] == constants.FENCE_MARGIN_M
    assert expected["FS_GCS_TIMEOUT"] == constants.FS_GCS_TIMEOUT_S
    parm = REPO / "simulation/sitl/params/searchdrone-safety.parm"
    if parm.exists():  # the parameter file arrives with the simulator tickets
        values = _parm(parm)
        for name, value in expected.items():
            assert values[name] == value, name


def _parm(path: Path) -> dict[str, float]:
    out: dict[str, float] = {}
    for line in path.read_text().splitlines():
        words = line.split("#")[0].split()
        if len(words) >= 2:
            out[words[0]] = float(words[1])
    return out


def test_area_limit_from_home() -> None:
    assert fc_area_max_from_home_m(DEFAULT_LIMITS) == 515
    assert fc_area_max_from_home_m(SafetyLimits(max_range_from_pilot_m=1000)) == 1015


def test_charter_violations() -> None:
    assert charter_violations(CHARTER, DEFAULT_LIMITS) == []
    strict = SafetyLimits(
        max_alt_agl_m=100, max_speed_ms=10, max_range_from_pilot_m=400, battery_rtl_pct=20
    )
    assert charter_violations(CHARTER, strict) == [
        "alt_max_agl_m",
        "max_speed_ms",
        "max_range_from_pilot_m",
        "battery_rtl_pct",
    ]


def test_caps_not_configurable() -> None:
    """The limits module reads no settings, files or environment."""
    tree = ast.parse(Path(limits.__file__).read_text())
    imported = {
        node.module or "" for node in ast.walk(tree) if isinstance(node, ast.ImportFrom)
    } | {
        alias.name
        for node in ast.walk(tree)
        if isinstance(node, ast.Import)
        for alias in node.names
    }
    assert imported <= {
        "__future__",
        "dataclasses",
        "typing",
        "sd_schemas.charter",
        "sd_schemas.safety",
        "sd_safety.constants",
    }
