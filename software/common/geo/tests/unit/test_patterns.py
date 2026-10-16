"""The lawnmower pattern and the coverage measure."""

from __future__ import annotations

import math
from itertools import pairwise

import pytest
from hypothesis import given, settings
from hypothesis import strategies as st
from shapely import Point
from shapely import Polygon as ShapelyPolygon

from sd_geo import PatternError, coverage_fraction, lawnmower

SQUARE = [(0.0, 0.0), (100.0, 0.0), (100.0, 100.0), (0.0, 100.0)]


def _approx(points: list[tuple[float, float]]) -> list[object]:
    return [pytest.approx(p, abs=0.01) for p in points]


def test_lawnmower_rect() -> None:
    third, two_thirds = 38.333, 61.667
    assert lawnmower(SQUARE, 30, 90, (0, 0)) == _approx(
        [
            (0, 15),
            (100, 15),
            (100, third),
            (0, third),
            (0, two_thirds),
            (100, two_thirds),
            (100, 85),
            (0, 85),
        ]
    )


def test_lawnmower_survey_small() -> None:
    small = [(60.0, 20.0), (140.0, 20.0), (140.0, 100.0), (60.0, 100.0)]
    points = lawnmower(small, 30, 90, (0, 0))
    assert sorted({round(y, 3) for _, y in points}) == [35, 60, 85]
    assert points[0] == pytest.approx((60, 35))


@pytest.mark.parametrize(
    ("start", "first"),
    [((0, 0), (0, 15)), ((100, 0), (100, 15)), ((0, 100), (0, 85)), ((100, 100), (100, 85))],
)
def test_starts_at_the_nearest_corner(
    start: tuple[float, float], first: tuple[float, float]
) -> None:
    points = lawnmower(SQUARE, 30, 90, start)
    assert points[0] == pytest.approx(first, abs=0.01)
    ys = [round(y, 3) for _, y in points]
    assert len(set(ys)) == 4  # every line flown once


def test_heading_north_runs_lines_along_y() -> None:
    points = lawnmower(SQUARE, 30, 0, (0, 0))
    assert points[0] == pytest.approx((15, 0), abs=0.01)
    assert points[1] == pytest.approx((15, 100), abs=0.01)


def test_heading_none_follows_the_longest_side() -> None:
    long_thin = [(0.0, 0.0), (0.0, 300.0), (60.0, 300.0), (60.0, 0.0)]
    points = lawnmower(long_thin, 30, None, (0, 0))
    xs = {round(x, 3) for x, _ in points}
    assert xs == {15.0, 45.0}  # two lines along the long (north) side


def test_lawnmower_concave_stays_inside() -> None:
    l_shape = [(0.0, 0.0), (100.0, 0.0), (100.0, 40.0), (40.0, 40.0), (40.0, 100.0), (0.0, 100.0)]
    shape = ShapelyPolygon(l_shape)
    points = lawnmower(l_shape, 20, 90, (0, 0))
    assert all(shape.buffer(1e-6).contains(Point(p)) for p in points)
    track = _fly(points, 40.0)
    assert coverage_fraction(l_shape, track, 40.0, 20) >= 0.95


def test_concave_line_cut_in_two_flies_both_pieces() -> None:
    u_shape = [(0.0, 0.0), (100.0, 0.0), (100.0, 100.0), (70.0, 100.0), (70.0, 30.0),
               (30.0, 30.0), (30.0, 100.0), (0.0, 100.0)]  # fmt: skip
    points = lawnmower(u_shape, 20, 90, (0, 0))
    upper = [p for p in points if p[1] > 30]
    assert any(x < 30 for x, _ in upper) and any(x > 70 for x, _ in upper)


@pytest.mark.parametrize(
    "polygon",
    [
        [(float(i), float(i % 2)) for i in range(21)],  # 21 corners
        [(0.0, 0.0), (10.0, 0.0)],
        [(0.0, 0.0), (100.0, 100.0), (100.0, 0.0), (0.0, 100.0)],  # crosses itself
    ],
)
def test_bad_polygons_are_refused(polygon: list[tuple[float, float]]) -> None:
    with pytest.raises(PatternError):
        lawnmower(polygon, 30, 90, (0, 0))


def test_bad_spacing_is_refused() -> None:
    with pytest.raises(PatternError):
        lawnmower(SQUARE, 0, 90, (0, 0))


def test_narrow_polygon_gives_one_line_in_the_middle() -> None:
    narrow = [(0.0, 0.0), (100.0, 0.0), (100.0, 20.0), (0.0, 20.0)]
    points = lawnmower(narrow, 30, 90, (0, 0))
    assert points == _approx([(0, 10), (100, 10)])


def _fly(points: list[tuple[float, float]], alt: float) -> list[tuple[float, float, float]]:
    """A perfect track sampled every metre along the waypoints."""
    track: list[tuple[float, float, float]] = []
    for (x1, y1), (x2, y2) in pairwise(points):
        n = max(1, math.ceil(math.dist((x1, y1), (x2, y2))))
        track += [(x1 + (x2 - x1) * k / n, y1 + (y2 - y1) * k / n, alt) for k in range(n + 1)]
    return track


def test_perfect_track_covers_the_square() -> None:
    points = lawnmower(SQUARE, 30, 90, (0, 0))
    assert coverage_fraction(SQUARE, _fly(points, 40), 40, 30) >= 0.99


def test_coverage_fraction_known_track() -> None:
    # One line along y = 50 reaches 17 m: rows with centres 37.5..62.5 (6 of 20 rows).
    track = [(float(x), 50.0, 40.0) for x in range(101)]
    assert coverage_fraction(SQUARE, track, 40, 30) == pytest.approx(6 / 20, abs=0.005)


def test_samples_at_the_wrong_height_do_not_count() -> None:
    track = [(float(x), 50.0, 50.0) for x in range(101)]
    assert coverage_fraction(SQUARE, track, 40, 30) == 0.0
    assert coverage_fraction(SQUARE, [], 40, 30) == 0.0


def test_coverage_needs_an_area() -> None:
    with pytest.raises(ValueError, match="no area"):
        coverage_fraction([(0.0, 0.0), (1.0, 1.0), (2.0, 2.0)], [], 40, 30)


@settings(max_examples=200, deadline=None)
@given(
    w=st.floats(20, 400),
    h=st.floats(20, 400),
    spacing=st.floats(10, 100),
    heading=st.floats(0, 180),
)
def test_pattern_stays_inside(w: float, h: float, spacing: float, heading: float) -> None:
    rect = [(0.0, 0.0), (w, 0.0), (w, h), (0.0, h)]
    points = lawnmower(rect, spacing, heading, (0, 0))
    shape = ShapelyPolygon(rect).buffer(1e-6)
    assert all(shape.contains(Point(p)) for p in points)


@settings(max_examples=100, deadline=None)
@given(
    w=st.floats(60, 400),
    h=st.floats(60, 400),
    spacing=st.floats(10, 40),
    heading=st.sampled_from([0.0, 90.0, 180.0]),
)
def test_pattern_along_the_sides_covers(w: float, h: float, spacing: float, heading: float) -> None:
    rect = [(0.0, 0.0), (w, 0.0), (w, h), (0.0, h)]
    points = lawnmower(rect, spacing, heading, (0, 0))
    assert coverage_fraction(rect, _fly(points, 40), 40, spacing) >= 0.95
