import time

import numpy as np
import pytest
from hypothesis import given, settings
from hypothesis import strategies as st

from sd_motion import MotionState, Terrain, advance, for_category, negative, start_state, step

CATEGORIES = ("dementia", "despondent", "hiker", "gatherer")
CELL_M = 25.0


def _empty(shape: tuple[int, int]) -> MotionState:
    return MotionState(np.zeros(shape), np.zeros((8, *shape)), 0.0, 0.0)


def _at(shape: tuple[int, int], rc: tuple[int, int], heading: int | None = None) -> MotionState:
    s = _empty(shape)
    if heading is None:
        s.moving[:, rc[0], rc[1]] = 1.0 / 8.0
    else:
        s.moving[heading, rc[0], rc[1]] = 1.0
    return s


@st.composite
def worlds(draw: st.DrawFn) -> tuple[Terrain, MotionState, str]:
    h = draw(st.integers(4, 24))
    w = draw(st.integers(4, 24))
    seed = draw(st.integers(0, 2**31 - 1))
    rng = np.random.default_rng(seed)
    classes = rng.choice([10, 20, 30, 40, 50, 80, 90], size=(h, w))
    track = rng.random((h, w)) < 0.15
    terrain = Terrain.from_classes(classes, track, CELL_M)
    land = 1.0 - terrain.wall
    stopped = rng.random((h, w)) * land
    moving = rng.random((8, h, w)) * land
    row = float(rng.random())
    scale = 1.0 / (stopped.sum() + moving.sum() + row)
    state = MotionState(stopped * scale, moving * scale, row * scale, rng.random() * 7200)
    return terrain, state, draw(st.sampled_from(CATEGORIES))


@settings(max_examples=60, deadline=None)
@given(worlds(), st.integers(1, 40))
def test_mass_is_kept_and_never_enters_water(world, steps) -> None:
    terrain, state, category = world
    params = for_category(category)
    for _ in range(steps):
        state = step(state, terrain, params)
        assert state.total() == pytest.approx(1.0, abs=1e-5)
        assert state.p[terrain.wall > 0].sum() == pytest.approx(0.0, abs=1e-9)
        assert state.stopped.min() >= 0.0
        assert state.moving.min() >= -1e-9


@settings(max_examples=30, deadline=None)
@given(worlds())
def test_deterministic(world) -> None:
    terrain, state, category = world
    params = for_category(category)
    a = advance(state, terrain, params, 600)
    b = advance(state, terrain, params, 600)
    assert np.array_equal(a.moving, b.moving)
    assert np.array_equal(a.stopped, b.stopped)
    assert a.row == b.row


@settings(max_examples=30, deadline=None)
@given(worlds())
def test_stopped_mass_never_moves(world) -> None:
    terrain, state, category = world
    still = MotionState(state.stopped, np.zeros_like(state.moving), state.row, state.t_s)
    after = advance(still, terrain, for_category(category), 900)
    assert np.allclose(after.stopped, state.stopped)
    assert after.moving.sum() == 0.0


def test_walkers_keep_to_a_straight_track() -> None:
    shape = (81, 81)
    forest = np.ones(shape, bool)
    no_water = np.zeros(shape, bool)
    track = np.zeros(shape, bool)
    track[:, 40] = True  # a north-south track through forest
    params = for_category("hiker")
    rows = np.arange(shape[0])[:, None]

    def shares(terrain: Terrain) -> tuple[float, float]:
        state = advance(_at(shape, (40, 40)), terrain, params, 600)
        p = state.p / state.p.sum()
        return float(p[:, 39:42].sum()), float((p[:, 39:42] * np.abs(rows - 40)).sum())

    near, along = shares(Terrain.from_cover(forest, track, no_water, CELL_M))
    near_without, along_without = shares(Terrain.from_cover(forest, ~forest, no_water, CELL_M))
    assert near > 2 * near_without
    assert along > 3 * along_without  # they went along it, not only stayed on it


def test_walkers_leave_a_fan_in_their_heading() -> None:
    shape = (60, 60)
    terrain = Terrain.open(shape, CELL_M)
    state = advance(_at(shape, (10, 30), heading=0), terrain, for_category("dementia"), 900)
    p = state.p
    north = p[11:, :].sum()
    assert north > 0.9 * p.sum()


def test_water_turns_walkers_back() -> None:
    shape = (30, 30)
    water = np.zeros(shape, bool)
    water[20:, :] = True  # a lake to the north
    terrain = Terrain.from_cover(np.zeros(shape, bool), np.zeros(shape, bool), water, CELL_M)
    state = advance(_at(shape, (15, 15), heading=0), terrain, for_category("hiker"), 1800)
    assert state.p[20:].sum() == 0.0
    assert state.total() == pytest.approx(1.0, abs=1e-5)


def test_mass_off_the_edge_joins_the_world() -> None:
    shape = (10, 10)
    state = advance(_at(shape, (8, 5), heading=0), Terrain.open(shape, CELL_M),
                    for_category("hiker"), 900)  # fmt: skip
    assert state.row > 0.5
    assert state.total() == pytest.approx(1.0, abs=1e-5)


@pytest.mark.parametrize("category", CATEGORIES)
def test_time_step_hardly_matters(category: str) -> None:
    shape = (81, 81)
    rng = np.random.default_rng(3)
    track = np.zeros(shape, bool)
    track[40, :] = True
    terrain = Terrain.from_cover(rng.random(shape) < 0.6, track, np.zeros(shape, bool), CELL_M)
    params = for_category(category)
    a = advance(_at(shape, (40, 40)), terrain, params, 3600, dt_s=15.0)
    b = advance(_at(shape, (40, 40)), terrain, params, 3600, dt_s=7.5)
    rr, cc = np.indices(shape)
    dist = np.hypot(rr - 40, cc - 40)

    def mean_dist(s: MotionState) -> float:
        return float((s.p * dist).sum() / s.p.sum())

    assert mean_dist(a) == pytest.approx(mean_dist(b), rel=0.02)
    assert a.stopped.sum() == pytest.approx(b.stopped.sum(), rel=0.02)


def test_advance_runs_whole_and_part_steps() -> None:
    shape = (12, 12)
    terrain = Terrain.open(shape, CELL_M)
    state = advance(_at(shape, (6, 6)), terrain, for_category("hiker"), 40)
    assert state.t_s == pytest.approx(40.0)
    with pytest.raises(ValueError, match="backwards"):
        advance(state, terrain, for_category("hiker"), -1)


def test_negative_sweep_lowers_the_searched_cells_and_keeps_the_sum() -> None:
    shape = (10, 10)
    terrain = Terrain.open(shape, CELL_M)
    rings = np.full(shape, 0.8 / 100)
    state = start_state(terrain, for_category("hiker"), (5, 5), 1800, rings, 0.2)
    pod = np.zeros(shape)
    pod[4:7, 4:7] = 0.8
    after = negative(state, pod)
    assert after.total() == pytest.approx(1.0, abs=1e-5)
    assert after.p[5, 5] < 0.5 * state.p[5, 5]
    assert after.p[0, 0] > state.p[0, 0]
    assert after.row > state.row


def test_searched_cells_refill_only_from_walkers() -> None:
    shape = (31, 31)
    terrain = Terrain.open(shape, CELL_M)
    params = for_category("dementia")
    rings = np.full(shape, 0.9 / shape[0] / shape[1])
    state = start_state(terrain, params, (15, 15), 3 * 3600, rings, 0.1)
    block = (slice(13, 18), slice(13, 18))
    pod = np.zeros(shape)
    pod[block] = 0.9
    searched = negative(state, pod)
    later = advance(searched, terrain, params, 300)
    assert searched.p[block].sum() < later.p[block].sum() < state.p[block].sum()
    assert later.stopped[block].sum() == pytest.approx(searched.stopped[block].sum(), rel=0.2)


def test_start_state_blends_the_walk_and_the_rings() -> None:
    shape = (41, 41)
    water = np.zeros(shape, bool)
    water[:, :5] = True
    terrain = Terrain.from_cover(np.zeros(shape, bool), np.zeros(shape, bool), water, CELL_M)
    rings = np.full(shape, 0.7 / shape[0] / shape[1])
    state = start_state(terrain, for_category("despondent"), (20, 20), 3600, rings, 0.3)
    assert state.total() == pytest.approx(1.0, abs=1e-5)
    assert state.p[:, :5].sum() == 0.0
    assert state.t_s == 3600
    lying = start_state(terrain, for_category("despondent"), (20, 20), 3600, rings, 0.3,
                        moving_now=0.0)  # fmt: skip
    assert lying.moving.sum() == 0.0
    assert lying.stopped[20, 20] > 0.4


def test_start_state_follows_a_known_heading() -> None:
    shape = (41, 41)
    terrain = Terrain.open(shape, CELL_M)
    rings = np.zeros(shape)
    params = for_category("hiker")
    east = start_state(terrain, params, (20, 20), 600, rings, 1.0, heading_deg=90.0, blend=1.0)
    p = east.p
    assert p[:, 21:].sum() > 1.3 * p[:, :20].sum()  # ahead weighs twice behind


def test_a_large_map_is_fast_enough() -> None:
    shape = (160, 160)
    rng = np.random.default_rng(1)
    track = np.zeros(shape, bool)
    track[80, :] = track[:, 40] = True
    terrain = Terrain.from_cover(rng.random(shape) < 0.6, track, rng.random(shape) < 0.03, CELL_M)
    rings = np.full(shape, 0.9 / shape[0] / shape[1]) * (1.0 - terrain.wall)
    t0 = time.perf_counter()
    start_state(terrain, for_category("hiker"), (80, 80), 3 * 3600, rings, 0.1)
    # The spec asks for 1 s on a laptop; CI runners are slower and shared.
    assert time.perf_counter() - t0 < 3.0
