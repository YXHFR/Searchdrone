"""The gate's geometry: geodesic distance, the local plane and exact polygon checks."""

from __future__ import annotations

import math

import pytest
from gate_fixtures import HOME_LAT, HOME_LON, L_AREA, PLANE, latlon

from sd_safety.targets import Plane, Polygon, distance_m, polygon_from_geojson


def _dms(d: int, m: int, s: float) -> float:
    sign = -1 if d < 0 else 1
    return sign * (abs(d) + m / 60 + s / 3600)


def test_vincenty_reference_line() -> None:
    """Flinders Peak to Buninyong, the classic test of Vincenty's formula: 54 972.271 m."""
    a = (_dms(-37, 57, 3.72030), _dms(144, 25, 29.52440))
    b = (_dms(-37, 39, 10.15610), _dms(143, 55, 35.38390))
    assert distance_m(*a, *b) == pytest.approx(54_972.271, abs=0.001)


def test_one_degree_along_the_meridian_and_the_equator() -> None:
    assert distance_m(0, 0, 1, 0) == pytest.approx(110_574.389, abs=0.01)
    assert distance_m(0, 0, 0, 1) == pytest.approx(111_319.491, abs=0.01)


def test_distance_edge_cases() -> None:
    assert distance_m(HOME_LAT, HOME_LON, HOME_LAT, HOME_LON) == 0.0
    assert distance_m(10, 180, 10, -180) == 0.0  # the same point written twice
    assert distance_m(0, 0, 0.5, 179.7) == math.inf  # nearly antipodal: never converges


def test_plane_matches_geodesic_distance_nearby() -> None:
    for east, north, tolerance in (
        (500, 0, 0.01),
        (0, 500, 0.01),
        (-353.6, 353.6, 0.01),
        (1500, -1500, 0.2),
    ):
        lat, lon = latlon(east, north)
        planar = math.hypot(*PLANE.xy(lat, lon))
        assert planar == pytest.approx(distance_m(HOME_LAT, HOME_LON, lat, lon), abs=tolerance)
    lat, lon = latlon(120, -80)
    x, y = PLANE.xy(lat, lon)
    assert (x, y) == (pytest.approx(120, abs=1e-6), pytest.approx(-80, abs=1e-6))


def test_plane_across_the_antimeridian() -> None:
    plane = Plane(0.0, 179.999)
    east, _ = plane.xy(0.0, -179.999)
    assert east == pytest.approx(222.6, abs=0.1)


SQUARE = Polygon(((0, 0), (10, 0), (10, 10), (0, 10)))
L = polygon_from_geojson(L_AREA, PLANE)


def test_contains_is_closed() -> None:
    assert SQUARE.contains((5, 5))
    assert SQUARE.contains((10, 5))  # on an edge
    assert SQUARE.contains((0, 0))  # on a corner
    assert not SQUARE.contains((10.01, 5))
    assert not SQUARE.contains((-1, -1))


def test_legs_in_a_concave_polygon() -> None:
    assert L.leg_outside_by((-250, -250), (250, -250)) == 0.0
    assert L.leg_outside_by((250, 50), (50, 250)) == pytest.approx(50, abs=1e-6)
    assert L.leg_outside_by((-300, -300), (300, -300)) == 0.0  # along an edge
    assert L.leg_outside_by((0, 0), (0, 0)) == 0.0  # no length
    assert L.leg_outside_by((300, 100), (100, 300)) > 0  # corner to corner across the notch
    assert L.leg_outside_by((100, 100), (100, 300)) == 0.0  # along the notch edge


def test_polygon_inside_polygon() -> None:
    inner = Polygon(((-200, -200), (0, -200), (0, 0), (-200, 0)))
    assert L.outside_by(inner) == 0.0
    across = Polygon(((50, 50), (150, 50), (150, 150), (50, 150)))
    assert L.outside_by(across) == pytest.approx(50, abs=1e-6)


@pytest.mark.parametrize(
    ("corners", "simple"),
    [
        (((0, 0), (10, 0), (10, 10), (0, 10)), True),
        (((0, 0), (10, 10), (10, 0), (0, 10)), False),  # bow tie
        (((0, 0), (10, 0), (20, 0)), False),  # no area
        (((0, 0), (10, 0), (10, 10), (5, 0), (0, 10)), False),  # touches its own edge
        (((0, 0), (0, 0), (10, 0), (10, 10)), False),  # repeated corner
        (((0, 0), (10, 0), (10, 10), (0, 10), (0, 2), (0, 5)), False),  # folds back on an edge
        (((0, 0), (10, 0), (10, 10), (0, 10), (0, 5), (0, 2)), True),  # extra corners on an edge
    ],
)
def test_simple_polygons(corners: tuple[tuple[float, float], ...], simple: bool) -> None:
    assert Polygon(corners).is_simple() is simple
    assert L.is_simple()


@pytest.mark.parametrize("east_north", [(0.0, 0.0), (250.0, -400.0), (-1500.0, 1200.0)])
def test_plane_latlon_inverts_xy(east_north: tuple[float, float]) -> None:
    lat, lon = PLANE.latlon(*east_north)
    assert PLANE.xy(lat, lon) == pytest.approx(east_north, abs=1e-6)


def test_plane_latlon_across_the_antimeridian() -> None:
    plane = Plane(10.0, 179.999)
    lat, lon = plane.latlon(500.0, 0.0)
    assert lat == pytest.approx(10.0) and -180 <= lon < -179.99
    assert plane.xy(lat, lon) == pytest.approx((500.0, 0.0), abs=1e-6)
