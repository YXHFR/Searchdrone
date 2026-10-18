"""Base types, GeoJSON checks, and the mission limits with their outer cap."""

from __future__ import annotations

from typing import Any

import pytest
from pydantic import TypeAdapter, ValidationError

from sd_schemas.api import AnnotationCreate
from sd_schemas.base import GeoJsonLineString, GeoJsonPoint, GeoJsonPolygon, VehicleId
from sd_schemas.safety import LimitCaps, SafetyLimits

POLY = TypeAdapter(GeoJsonPolygon)
POINT = TypeAdapter(GeoJsonPoint)
LINE = TypeAdapter(GeoJsonLineString)
VID = TypeAdapter(VehicleId)

SQUARE = [[149.164, -35.364], [149.166, -35.364], [149.166, -35.362], [149.164, -35.364]]


def test_safety_limits_defaults_are_the_owner_values() -> None:
    limits = SafetyLimits()
    assert limits.max_alt_agl_m == 120
    assert limits.max_speed_ms == 15
    assert limits.max_range_from_pilot_m == 500
    assert limits.battery_rtl_pct == 15


@pytest.mark.parametrize(
    "bad",
    [
        {"max_alt_agl_m": 5},
        {"max_alt_agl_m": -1},
        {"max_speed_ms": 0},
        {"max_range_from_pilot_m": 0},
        {"battery_rtl_pct": 91},
    ],
)
def test_safety_limits_field_bounds(bad: dict[str, float]) -> None:
    with pytest.raises(ValidationError):
        SafetyLimits(**bad)


def test_limit_caps_defaults() -> None:
    caps = LimitCaps()
    assert (caps.max_alt_agl_m, caps.max_speed_ms, caps.max_range_from_pilot_m) == (
        300,
        25,
        2000,
    )
    assert caps.min_battery_rtl_pct == 15
    assert caps.min_alt_agl_m == 5


@pytest.mark.parametrize(
    "bad",
    [
        {"max_alt_agl_m": 100},  # below the standard 120 m
        {"max_speed_ms": 10},
        {"max_range_from_pilot_m": 400},
        {"min_battery_rtl_pct": 10},  # the battery floor can never be loosened
        {"min_alt_agl_m": 4},
        {"min_battery_rtl_pct": 20},  # would forbid the standard 15 %
    ],
)
def test_limit_caps_cannot_loosen_floors_or_undercut_standard(bad: dict[str, float]) -> None:
    with pytest.raises(ValidationError):
        LimitCaps(**bad)


def test_default_limits_fit_inside_default_caps() -> None:
    limits, caps = SafetyLimits(), LimitCaps()
    assert limits.max_alt_agl_m <= caps.max_alt_agl_m
    assert limits.max_speed_ms <= caps.max_speed_ms
    assert limits.max_range_from_pilot_m <= caps.max_range_from_pilot_m
    assert limits.battery_rtl_pct >= caps.min_battery_rtl_pct


def test_polygon_ok() -> None:
    POLY.validate_python({"type": "Polygon", "coordinates": [SQUARE]})


@pytest.mark.parametrize(
    "bad",
    [
        {"type": "Polygon", "coordinates": [SQUARE[:-1]]},  # not closed
        {"type": "Polygon", "coordinates": [SQUARE, SQUARE]},  # a hole
        {"type": "Polygon", "coordinates": [[SQUARE[0], SQUARE[1], SQUARE[0]]]},  # too few
        {"type": "Polygon", "coordinates": [[[200, 0], [1, 1], [1, 0], [200, 0]]]},  # range
        {"type": "LineString", "coordinates": [SQUARE]},
        {"type": "Polygon", "coordinates": [SQUARE], "bbox": []},
        {"type": "Polygon", "coordinates": [[[1, 1], [1, 1], [2, 2], [1, 1]]]},  # 2 corners
    ],
)
def test_polygon_rejected(bad: dict[str, Any]) -> None:
    with pytest.raises(ValidationError):
        POLY.validate_python(bad)


def test_polygon_point_limit() -> None:
    ring = [[149.0 + i * 1e-5, -35.3 + (i % 2) * 1e-5] for i in range(201)]
    ring[-1] = ring[0]
    POLY.validate_python({"type": "Polygon", "coordinates": [ring]})
    ring.insert(1, [149.5, -35.5])
    with pytest.raises(ValidationError):
        POLY.validate_python({"type": "Polygon", "coordinates": [ring]})


def test_point() -> None:
    POINT.validate_python({"type": "Point", "coordinates": [149.1652, -35.3632]})
    with pytest.raises(ValidationError):
        POINT.validate_python({"type": "Point", "coordinates": [-35.3632]})
    with pytest.raises(ValidationError):
        POINT.validate_python({"type": "Point", "coordinates": [True, 1]})


@pytest.mark.parametrize("good", ["sim-01", "abc", "a" * 32])
def test_vehicle_id_ok(good: str) -> None:
    VID.validate_python(good)


@pytest.mark.parametrize("bad", ["SD01", "ab", "-sim", "a" * 33, "sim_01"])
def test_vehicle_id_rejected(bad: str) -> None:
    with pytest.raises(ValidationError):
        VID.validate_python(bad)


LINE_OK = {"type": "LineString", "coordinates": [[149.164, -35.364], [149.165, -35.363]]}


def test_line_string_accepts_a_drawn_path() -> None:
    assert LINE.validate_python(LINE_OK) == LINE_OK
    fifty = [[149.164 + i * 1e-4, -35.364] for i in range(50)]
    LINE.validate_python({"type": "LineString", "coordinates": fifty})


@pytest.mark.parametrize(
    "bad",
    [
        {"type": "LineString", "coordinates": [[149.164, -35.364]]},  # one point
        {"type": "LineString", "coordinates": [[149.164 + i * 1e-4, -35.364] for i in range(51)]},
        {"type": "LineString", "coordinates": [[149.164, -35.364], [149.164, -35.364]]},
        {"type": "LineString", "coordinates": [[-35.364, 149.164], [-35.363, 149.165]]},
        {"type": "Polygon", "coordinates": [SQUARE]},
        {"type": "LineString", "coordinates": [[149.164, -35.364], [149.165, -35.363]], "x": 1},
    ],
)
def test_line_string_rejects(bad: dict[str, Any]) -> None:
    with pytest.raises(ValidationError):
        LINE.validate_python(bad)


def test_path_annotation_needs_a_line_and_a_pa_label() -> None:
    assert AnnotationCreate(kind="path", label="pa-1", geometry=LINE_OK).kind == "path"
    with pytest.raises(ValidationError):
        AnnotationCreate(kind="path", label="P1", geometry=LINE_OK)
    with pytest.raises(ValidationError):
        AnnotationCreate(kind="area", label="ar-1", geometry=LINE_OK)
    with pytest.raises(ValidationError):
        AnnotationCreate(
            kind="path", label="pa-1", geometry={"type": "Point", "coordinates": [149.1, -35.3]}
        )
