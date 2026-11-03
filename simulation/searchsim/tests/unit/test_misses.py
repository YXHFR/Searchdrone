"""Correlated misses: a repeat sweep of the same cell is worth less than the first."""

from __future__ import annotations

import numpy as np
import pytest
from hypothesis import given, settings
from hypothesis import strategies as st

from sd_searchsim.config import SimConfig
from sd_searchsim.env import Chooser, Episode, EpisodeResult, pod_after, split_pod
from sd_searchsim.policies import BASELINES
from sd_searchsim.scenario import Scenario, make_scenario


def _fly(ep: Episode, choose: Chooser) -> tuple[EpisodeResult, list[np.ndarray]]:
    seen = []
    while not ep.done:
        obs = ep.observe()
        seen.append(obs.features)
        ep.step(choose(obs))
    return ep.result(), seen


@pytest.mark.parametrize("seed", range(4))
def test_no_correlation_gives_the_old_numbers_bit_for_bit(
    seed: int, independent_episode: type[Episode]
) -> None:
    cfg = SimConfig(miss_correlation=0.0)
    sc = make_scenario(seed, cfg)
    for name, make in BASELINES.items():
        new, old = Episode(sc, cfg), independent_episode(sc, cfg)
        new_result, new_seen = _fly(new, make())
        old_result, old_seen = _fly(old, make())
        assert new_result == old_result, name
        assert len(new_seen) == len(old_seen), name
        assert all(np.array_equal(a, b) for a, b in zip(new_seen, old_seen, strict=True)), name
        assert np.array_equal(new.belief, old.belief), name


def _standing_in_the_trees(cfg: SimConfig) -> tuple[Scenario, int]:
    """A scenario whose person stands still in the most wooded cell of the home sector."""
    sc = make_scenario(5, cfg)
    k, s = cfg.sector_cells, cfg.n_sectors
    row, col = int(sc.home_m[1] // cfg.sector_m), int(sc.home_m[0] // cfg.sector_m)
    block = sc.canopy[row * k : (row + 1) * k, col * k : (col + 1) * k]
    r, c = np.unravel_index(int(np.argmax(block)), block.shape)
    assert block[r, c] > 0.3
    centre = np.array([(col * k + c + 0.5) * cfg.cell_m, (row * k + r + 0.5) * cfg.cell_m])
    path = np.tile(centre, (len(sc.path_m), 1))
    return type(sc)(**{**sc.__dict__, "path_m": path}), row * s + col


def _chance_per_sweep(cfg: SimConfig, sweeps: int) -> list[float]:
    """Each sweep's chance to find the person, given the sweeps before it did not."""
    sc, sector = _standing_in_the_trees(cfg)
    ep = Episode(sc, cfg)
    out = []
    for _ in range(sweeps):
        before = ep.survive
        ep.step(sector)
        out.append((before - ep.survive) / before)
    return out


def test_a_repeat_sweep_of_the_same_cell_adds_less_than_the_first() -> None:
    chance = _chance_per_sweep(SimConfig(), 3)
    assert chance[0] > 0
    assert chance[1] < chance[0]
    assert chance[2] < chance[1]
    # The first sweep is the same as without the correlation; there every sweep is.
    flat = _chance_per_sweep(SimConfig(miss_correlation=0.0), 3)
    assert chance[0] == pytest.approx(flat[0])
    assert flat[1] == pytest.approx(flat[0])
    assert flat[2] == pytest.approx(flat[0])


def test_the_belief_expects_less_from_a_repeat_sweep() -> None:
    cfg = SimConfig()
    sc, sector = _standing_in_the_trees(cfg)
    ep = Episode(sc, cfg)
    first = ep.pod_belief_now()
    assert first == pytest.approx(sc.pod_belief)
    pos_first = ep.observe().features[sector, 0]
    ep.step(sector)
    second = ep.pod_belief_now()
    k, s = cfg.sector_cells, cfg.n_sectors
    swept = np.zeros(first.shape, dtype=bool)
    row, col = divmod(sector, s)
    swept[row * k : (row + 1) * k, col * k : (col + 1) * k] = True
    wooded = swept & (sc.canopy > 0)
    assert wooded.any()
    assert (second[wooded] < first[wooded]).all()
    assert np.array_equal(second[~swept], first[~swept])
    # The planners see it: the sector's expected detection drops more than the Bayes
    # update alone would make it.
    flat = Episode(sc, SimConfig(miss_correlation=0.0))
    assert flat.observe().features[sector, 0] == pytest.approx(pos_first)
    flat.step(sector)
    assert ep.observe().features[sector, 0] < flat.observe().features[sector, 0]


@settings(max_examples=50, deadline=None)
@given(st.integers(0, 2**31 - 1), st.floats(0.0, 1.0))
def test_split_and_repeat_chances_stay_probabilities(seed: int, m: float) -> None:
    rng = np.random.default_rng(seed)
    canopy = rng.uniform(0.0, 1.0, (10, 10))
    pod = rng.uniform(0.0, 1.0) * (1.0 - canopy)
    visible, per_pass = split_pod(pod, canopy, m)
    assert ((visible >= 0) & (visible <= 1) & (per_pass >= 0) & (per_pass <= 1)).all()
    assert visible * per_pass == pytest.approx(pod)
    last = pod_after(visible, per_pass, np.zeros(pod.shape, dtype=int))
    assert last == pytest.approx(pod)
    for n in range(1, 6):
        now = pod_after(visible, per_pass, np.full(pod.shape, n))
        assert ((now >= 0) & (now <= last + 1e-15)).all()
        last = now


@pytest.mark.parametrize("m", [0.5, 1.0])
def test_with_correlation_the_belief_and_the_score_stay_sane(m: float) -> None:
    cfg = SimConfig(miss_correlation=m)
    for seed in range(4):
        ep = Episode(make_scenario(seed, cfg), cfg)
        choose = BASELINES["greedy_psr"]()
        while not ep.done:
            pod = ep.pod_belief_now()
            assert ((pod >= 0) & (pod <= 1)).all()
            ep.step(choose(ep.observe()))
            assert ep.belief.sum() == pytest.approx(1.0)
            assert (ep.belief >= 0).all()
            assert 0.0 <= ep.survive <= 1.0
        r = ep.result()
        assert 0.0 <= r.p_find <= 1.0
        assert 0.0 <= r.score <= 1.0


def test_miss_correlation_must_be_a_share() -> None:
    with pytest.raises(ValueError, match="miss_correlation"):
        SimConfig(miss_correlation=1.5)
