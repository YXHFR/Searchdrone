"""The covered part of a sector: height band, reach, clipping and growing step by step."""

from __future__ import annotations

from shapely.geometry import box

from sd_geo.coverage import REACH_EXTRA_M
from sd_ground.services import coverage
from sd_safety.targets import Plane

PLANE = Plane(-35.363261, 149.165230)
SECTOR = box(0.0, 0.0, 200.0, 200.0)


def at(east: float, north: float, alt: float) -> tuple[float, float, float]:
    lat, lon = PLANE.latlon(east, north)
    return lat, lon, alt


def test_points_outside_the_height_band_do_not_count() -> None:
    points = [at(100, 100, 40.0 + 3.5), at(100, 100, 40.0 - 3.5)]
    assert coverage.covered_area(points, SECTOR, PLANE, alt_m=40, spacing_m=20) is None


def test_reach_is_half_the_spacing_plus_the_shared_extra() -> None:
    seen = coverage.covered_area([at(100, 100, 41.0)], SECTOR, PLANE, alt_m=40, spacing_m=20)
    assert seen is not None
    minx, _, maxx, _ = seen.bounds
    reach = 20 / 2 + REACH_EXTRA_M
    assert abs((maxx - minx) - 2 * reach) < 0.5


def test_clipped_to_the_sector() -> None:
    seen = coverage.covered_area([at(0, 0, 40.0)], SECTOR, PLANE, alt_m=40, spacing_m=20)
    assert seen is not None
    assert SECTOR.buffer(0.01).covers(seen)
    assert seen.area < 3.2 * (10 + REACH_EXTRA_M) ** 2 / 4 * 1.1  # one quarter of the disc


def test_grows_with_new_points_only() -> None:
    first = coverage._grow(None, [at(50, 50, 40.0)], SECTOR, PLANE, 40.0, 20.0)  # pyright: ignore[reportPrivateUsage]
    assert first is not None
    same = coverage._grow(first, [], SECTOR, PLANE, 40.0, 20.0)  # pyright: ignore[reportPrivateUsage]
    assert same is first
    more = coverage._grow(first, [at(150, 150, 40.0)], SECTOR, PLANE, 40.0, 20.0)  # pyright: ignore[reportPrivateUsage]
    assert more is not None and more.area > first.area * 1.9
    assert more.covers(first)


def test_state_is_per_mission() -> None:
    state = coverage.Coverage()
    state.of("a").since = None
    state.of("a").finished.add("x")
    assert state.of("b").finished == set()
    state.forget("a")
    assert state.of("a").finished == set()


SPACING = 30.0


def _lawnmower(skip: tuple[float, float, float] | None = None) -> list[tuple[float, float, float]]:
    """Track points every 4 m (8 m/s at 2 Hz) along a lawnmower over ``SECTOR`` at 30 m
    spacing. ``skip`` (north of the leg, from east, to east): that stretch of the leg is
    flown 40 m north of it instead, as a way around a cylinder there would."""
    points: list[tuple[float, float, float]] = []
    for i, north in enumerate(range(15, 200, 30)):
        east = list(range(15, 186, 4))
        for e in east if i % 2 == 0 else reversed(east):
            n = float(north)
            if skip is not None and skip[0] == north and skip[1] <= e <= skip[2]:
                n += 40.0
            points.append(at(e, n, 40.0))
    return points


def _gap(points: list[tuple[float, float, float]]) -> float:
    seen = coverage.covered_area(points, SECTOR, PLANE, alt_m=40, spacing_m=SPACING)
    return coverage.gap_m2(SECTOR, seen)


def test_a_full_search_leaves_no_gap() -> None:
    """The slivers a full search leaves at the edges and corners are below the floor."""
    assert _gap(_lawnmower()) == 0.0


def test_a_leg_skipped_for_90_m_behind_the_drone_is_a_gap() -> None:
    """The #234 case: the way around a cylinder cut 90 m of a leg; the cylinder was lifted
    after the drone had passed, so nothing searched it again. The part the neighbouring
    legs do not reach is about 26 m wide."""
    m2 = _gap(_lawnmower(skip=(105, 60, 150)))
    width = 2 * SPACING - 2 * (SPACING / 2 + REACH_EXTRA_M)
    assert 0.6 * 90 * width < m2 < 1.2 * 90 * width, m2


def test_the_floor_is_the_width_and_the_area_both() -> None:
    """A part counts when a disc ``GAP_MIN_WIDTH_M`` wide fits and it is at least
    ``GAP_MIN_M2``: tested on both sides of each."""
    w, a = coverage.GAP_MIN_WIDTH_M, coverage.GAP_MIN_M2

    def left(hole: object) -> float:
        return coverage.gap_m2(SECTOR, SECTOR.difference(hole))  # type: ignore[arg-type]

    side = a**0.5
    assert left(box(50, 50, 50 + side + 0.5, 50 + side + 0.5)) > 0  # just over the area
    assert left(box(50, 50, 50 + side - 0.5, 50 + side - 0.5)) == 0  # just under it
    long = 2 * a / w
    assert left(box(50, 50, 50 + long, 50 + w + 0.2)) > 0  # just wider than the floor
    assert left(box(50, 50, 50 + long, 50 + w - 0.2)) == 0  # a sliver: just narrower


def test_nothing_seen_is_all_gap() -> None:
    assert coverage.gap_m2(SECTOR, None) == SECTOR.area
