"""Property tests for the height floor (spec "Titta närmare" §17, acceptance tests 2 and 3).

Random small packs (trees, other nodata values, slopes, old, unknown and future scans),
random points and random laser states. The check is independent of the module: the
trees within 10 m are found here by cell centre (a subset of the cells the module
counts) with the owner's margins written out again.
"""

from __future__ import annotations

import math
from datetime import date

import numpy as np
from floor_fixtures import ORIGIN, PLANE, at
from hypothesis import given, settings
from hypothesis import strategies as st
from hypothesis.extra.numpy import arrays

from sd_safety.floor import FloorPack, FloorResult, LaserState, floor_at

SIZE = 40
FLIGHT = date(2026, 6, 1)


@st.composite
def packs(draw: st.DrawFn) -> FloorPack:
    heights = st.one_of(st.integers(-1, 400), st.sampled_from([-2, -50, -32768]))
    chm = draw(arrays(np.int16, (SIZE, SIZE), elements=heights))
    slope = draw(st.floats(-0.5, 0.5))
    ground = 100 * (200 + slope * np.arange(SIZE, dtype=np.float64))[None, :].repeat(SIZE, 0)
    dtm = np.round(ground).astype(np.int32)
    scans = st.sampled_from([0, 2010, 2016, 2019, 2026, 2030])
    years = draw(arrays(np.uint16, (2, 2), elements=scans))
    return FloorPack(
        name="random",
        sha256="0" * 64,
        origin=ORIGIN,
        dtm_cm=dtm,
        chm_dm=chm,
        scan_year=years,
        scan_cell_m=20,
    )


lasers = st.one_of(
    st.none(),
    st.builds(
        LaserState,
        confirmed=st.just(False),
        surface_amsl_m=st.none(),
        min_range_m=st.sampled_from([0.0, 10.0]),
    ),
    st.builds(
        LaserState,
        confirmed=st.just(True),
        surface_amsl_m=st.floats(150.0, 400.0),
        min_range_m=st.sampled_from([0.0, 10.0]),
    ),
)
points = st.tuples(st.floats(-19.9, 19.9), st.floats(-19.9, 19.9))


def spec_trees(pack: FloorPack, east: float, north: float) -> tuple[float, bool]:
    """The adjusted highest tree within 10 m (by cell centre), and whether an unknown
    cell is among them: the owner's margins written out again. Unknown (no height, a scan
    too old or after the flight, outside the pack) is 30 m forest at the least."""
    rows, cols = pack.dtm_cm.shape
    best, unknown = 0.0, False
    for r in range(-12, rows + 12):
        for c in range(-12, cols + 12):
            ce, cn = c - cols / 2 + 0.5, rows / 2 - r - 0.5
            if math.hypot(ce - east, cn - north) > 10.0:
                continue
            inside = 0 <= r < rows and 0 <= c < cols
            year = int(pack.scan_year[r // 20, c // 20]) if inside else 0
            height = int(pack.chm_dm[r, c]) if inside else -1
            if height < 0 or (year != 0 and not 0 <= FLIGHT.year - year <= 10):
                own = height / 10 + 2 + 5 if height >= 0 else 0.0
                best, unknown = max(best, 30 + 2 + 5, own), True
            else:
                growth = 5.0 if year == 0 else min(5.0, max(0.0, 0.5 * (FLIGHT.year - year)))
                best = max(best, height / 10 + 2 + growth)
    return best, unknown


def result(
    pack: FloorPack, p: tuple[float, float], laser: LaserState | None, lost: bool = False
) -> FloorResult:
    q = at(*p)
    return floor_at(
        pack, q.lat, q.lon, flight_date=FLIGHT, laser=laser, start_alt_amsl_m=0.0, laser_lost=lost
    )


@settings(max_examples=150, deadline=None)
@given(packs(), points, lasers, st.booleans())
def test_the_floor_never_below_the_margins(
    pack: FloorPack, p: tuple[float, float], laser: LaserState | None, lost: bool
) -> None:
    r = result(pack, p, laser, lost)
    assert r.ground_amsl_m is not None and r.tree_m is not None
    ground = r.ground_amsl_m
    tree, unknown = spec_trees(pack, *p)
    eps = 1e-6
    assert r.tree_m >= tree - eps
    assert r.floor_lowest_amsl_m >= ground + tree + 5 - eps
    if not r.laser_confirmed:
        assert r.floor_lowest_amsl_m >= ground + tree + 15 - eps
    assert r.floor_auto_amsl_m >= ground + 30 - eps
    assert r.floor_auto_amsl_m >= r.floor_lowest_amsl_m
    if unknown:
        assert r.floor_lowest_amsl_m >= ground + 37 + (5 if r.laser_confirmed else 15) - eps


@settings(max_examples=150, deadline=None)
@given(packs(), points, lasers)
def test_the_laser_only_raises_the_floor_over_the_maps_lowest(
    pack: FloorPack, p: tuple[float, float], laser: LaserState | None
) -> None:
    with_laser = result(pack, p, laser)
    without = result(pack, p, None)
    assert with_laser.ground_amsl_m is not None and with_laser.tree_m is not None
    map_lowest = with_laser.ground_amsl_m + with_laser.tree_m + 5
    assert with_laser.floor_lowest_amsl_m >= map_lowest - 1e-6
    assert with_laser.floor_auto_amsl_m >= without.floor_auto_amsl_m
    if laser is not None and laser.confirmed:
        assert with_laser.laser_confirmed
        assert laser.surface_amsl_m is not None
        laser_floor = laser.surface_amsl_m + max(5.0, laser.min_range_m + 2.0)
        assert with_laser.floor_lowest_amsl_m >= laser_floor - 1e-6
    else:
        assert with_laser.floor_lowest_amsl_m == without.floor_lowest_amsl_m


def test_plane_matches_the_fixture() -> None:
    """The fixtures place points in the same plane the pack reads them in."""
    pack = FloorPack(
        name="x",
        sha256="0" * 64,
        origin=ORIGIN,
        dtm_cm=np.zeros((2, 2), np.int32),
        chm_dm=np.zeros((2, 2), np.int16),
        scan_year=np.zeros((1, 1), np.uint16),
        scan_cell_m=20,
    )
    assert pack.plane == PLANE
