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
