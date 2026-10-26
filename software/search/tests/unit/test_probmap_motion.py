"""The probability map with movement (``[motion]``): the person may walk while we search."""

from __future__ import annotations

import datetime as dt
import time

import numpy as np
import pytest
from search_helpers import T0, at, profile  # pyright: ignore[reportMissingImports]

from sd_search.probmap.build import MapBuilder, MapInputs
from sd_search.probmap.pod import Still
from sd_search.probmap.terrain import NoTerrain, SimTerrainSource


def square(east: float, north: float, half: float) -> tuple[tuple[float, float], ...]:
    pts = [
        at(east - half, north - half),
        at(east + half, north - half),
        at(east + half, north + half),
        at(east - half, north + half),
    ]
    return tuple((lon, lat) for lat, lon in pts)


def ring(east: float, north: float, half: float) -> list[list[float]]:
    c = [list(p) for p in square(east, north, half)]
    return [*c, c[0]]


def survey(
    east: float, north: float, half: float, *, start_min: float, order: str, seq0: int
) -> list[Still]:
    """Stills over a square, one a minute, taken ``start_min`` after last seen."""
    out = []
    step = 50.0
    k = 0
    for y in np.arange(north - half + step / 2, north + half, step):
        for x in np.arange(east - half + step / 2, east + half, step):
            out.append(
                Still(
                    f"{order}-{k}",
                    seq0 + k,
                    order,
                    square(float(x), float(y), 30),
                    120.0,
                    True,
                    T0 + dt.timedelta(minutes=start_min + k / 6),
                )
            )
            k += 1
    return out


def inputs(now_min: float | None, **kw: object) -> MapInputs:
    base: dict[str, object] = {
        "profile": profile(subject_category="dementia"),
        "area": ring(0, 0, 900),
        "sectors": [("A", ring(-450, 0, 440)), ("B", ring(450, 0, 440))],
        "areas": [],
        "stills": [],
        "candidates": [],
        "alt_m": 40.0,
        "speed_ms": 5.0,
        "now_utc": None if now_min is None else T0 + dt.timedelta(minutes=now_min),
    }
    return MapInputs(**(base | kw))  # type: ignore[arg-type]


def _ipp(b: MapBuilder, i: MapInputs) -> tuple[int, int]:
    cell = b.grid_for(i).cell_of(0.0, 0.0)
    assert cell is not None
    return cell


def test_movement_off_is_exactly_the_old_map() -> None:
    stills = survey(0, 0, 100, start_min=90, order="p1", seq0=1)
    old = MapBuilder(NoTerrain()).build(inputs(None, stills=stills))
    off = MapBuilder(NoTerrain(), motion=False).build(inputs(180, stills=stills))
    assert np.array_equal(old.state.p, off.state.p)
    assert old.state.row == off.state.row
    assert off.state.motion is None


def test_the_map_spreads_with_time() -> None:
    b = MapBuilder(NoTerrain())
    early = b.build(inputs(30)).state
    late = MapBuilder(NoTerrain()).build(inputs(240)).state
    ipp = _ipp(b, inputs(30))
    assert early.motion is not None and late.motion is not None
    assert early.total() == pytest.approx(1.0, abs=1e-5)
    assert late.total() == pytest.approx(1.0, abs=1e-5)
    assert late.p[ipp] < early.p[ipp]
    assert late.motion.t_s == pytest.approx(240 * 60)


def test_a_searched_area_refills_only_from_walkers() -> None:
    b = MapBuilder(NoTerrain())
    before = b.build(inputs(89)).state
    stills = survey(0, 0, 150, start_min=90, order="p1", seq0=1)
    end = 90 + len(stills) / 6
    just_after = MapBuilder(NoTerrain()).build(inputs(end, stills=stills)).state
    later = MapBuilder(NoTerrain()).build(inputs(end + 30, stills=stills)).state
    seen = just_after.pod_cum > 0.5
    assert just_after.p[seen].sum() < 0.5 * before.p[seen].sum()
    # Walkers leave the searched area as well as enter it; only those who stop there add
    # to the stopped layer, and someone who sat down where we looked is still not there.
    assert later.motion is not None and just_after.motion is not None
    stopped_after = just_after.motion.stopped[seen].sum()
    assert stopped_after < later.motion.stopped[seen].sum() < 0.5 * before.p[seen].sum()


def test_someone_lying_down_does_not_walk() -> None:
    lying = profile(subject_category="hiker", likely_state=["lying"])
    state = MapBuilder(NoTerrain()).build(inputs(120, profile=lying)).state
    assert state.motion is not None
    assert state.motion.moving.sum() == 0.0


def test_reduced_mobility_stays_closer() -> None:
    def near(mobility: str) -> float:
        b = MapBuilder(NoTerrain())
        i = inputs(120, profile=profile(subject_category="hiker", mobility=mobility))
        cx, cy = b.grid_for(i).centers
        return float(b.build(i).state.p[np.hypot(cx, cy) < 500].sum())

    assert near("reduced") > 1.2 * near("good")


def test_nobody_walks_into_a_lake() -> None:
    corners = [at(200, 100), at(600, 100), at(600, 500), at(200, 500)]
    lake = [[lon, lat] for lat, lon in [*corners, corners[0]]]
    terrain = SimTerrainSource(
        [{"type": "Feature", "properties": {"kind": "lake"},
          "geometry": {"type": "Polygon", "coordinates": [lake]}}]
    )  # fmt: skip
    b = MapBuilder(terrain)
    i = inputs(180)
    state = b.build(i).state
    cx, cy = b.grid_for(i).centers
    inside = (cx > 215) & (cx < 585) & (cy > 115) & (cy < 485)
    assert state.p[inside].sum() == 0.0


def test_a_rebuild_with_a_later_clock_walks_on_from_the_cache() -> None:
    stills = survey(0, 0, 100, start_min=60, order="p1", seq0=1)
    b = MapBuilder(NoTerrain())
    b.build(inputs(100, stills=stills))
    t0 = time.perf_counter()
    cached = b.build(inputs(101, stills=stills)).state
    quick = time.perf_counter() - t0
    fresh = MapBuilder(NoTerrain()).build(inputs(101, stills=stills)).state
    assert np.allclose(cached.p, fresh.p, atol=1e-6)
    assert quick < 0.5


def test_a_six_hour_search_on_the_largest_map_fits_the_budget() -> None:
    area = ring(0, 0, 1500)  # 3 x 3 km plus the margin: the 160 x 160 cell maximum
    stills = survey(-600, 300, 200, start_min=120, order="p1", seq0=1) + survey(
        500, -400, 200, start_min=200, order="p2", seq0=1000
    )
    b = MapBuilder(NoTerrain())
    i = inputs(360, area=area, stills=stills, profile=profile(subject_category="hiker"))
    assert b.grid_for(i).shape == (160, 160)
    t0 = time.perf_counter()
    state = b.build(i).state
    seconds = time.perf_counter() - t0
    assert state.total() == pytest.approx(1.0, abs=1e-4)
    # The spec's budget is 2 s on a laptop; shared CI runners are slower.
    assert seconds < 4.0
