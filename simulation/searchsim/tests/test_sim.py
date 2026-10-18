"""The scenario generator and the search episode."""

from __future__ import annotations

import numpy as np
import pytest
from hypothesis import given, settings
from hypothesis import strategies as st

from sd_searchsim.config import SimConfig
from sd_searchsim.env import Episode, Observation, diffuse, run_episode
from sd_searchsim.policies import BASELINES
from sd_searchsim.scenario import make_scenario

CFG = SimConfig()


def test_same_seed_same_scenario() -> None:
    a, b = make_scenario(7), make_scenario(7)
    for name in ("canopy", "prior", "allowed", "path_m", "pod_true", "home_m"):
        assert np.array_equal(getattr(a, name), getattr(b, name)), name
    assert not np.array_equal(a.path_m, make_scenario(8).path_m)


@pytest.mark.parametrize("seed", range(20))
def test_scenario_is_sane(seed: int) -> None:
    sc = make_scenario(seed)
    size = CFG.grid_cells * CFG.cell_m
    assert sc.prior.sum() == pytest.approx(1.0)
    assert (sc.prior >= 0).all()
    assert ((sc.path_m >= 0) & (sc.path_m < size)).all()
    assert ((sc.pod_true >= 0) & (sc.pod_true <= 1)).all()
    # Home is never in a no-fly sector.
    home_sector = (int(sc.home_m[1] // CFG.sector_m), int(sc.home_m[0] // CFG.sector_m))
    assert sc.allowed[home_sector]


@settings(max_examples=30, deadline=None)
@given(st.integers(0, 2**31 - 1), st.floats(0.0, 6.0))
def test_diffuse_keeps_mass_and_sign(seed: int, sigma: float) -> None:
    p = np.random.default_rng(seed).random((20, 20))
    q = diffuse(p, sigma)
    assert q.sum() == pytest.approx(p.sum())
    assert (q >= 0).all()


def test_step_updates_the_belief_like_bayes() -> None:
    ep = Episode(make_scenario(3))
    obs = ep.observe()
    sector = int(np.flatnonzero(obs.feasible)[0])
    before = ep._sector_sum(ep.belief)[sector] / ep.belief.sum()
    ep.step(sector)
    after = ep._sector_sum(ep.belief)[sector]
    assert ep.belief.sum() == pytest.approx(1.0)
    assert after < before
    assert ep.swept[sector] == 1


def test_infeasible_sector_is_refused() -> None:
    sc = next(make_scenario(s) for s in range(100) if not make_scenario(s).allowed.all())
    ep = Episode(sc)
    forbidden = int(np.flatnonzero(~sc.allowed.ravel())[0])
    assert not ep.observe().feasible[forbidden]
    with pytest.raises(ValueError, match="not permitted"):
        ep.step(forbidden)


@pytest.mark.parametrize("name", sorted(BASELINES))
def test_every_flight_keeps_the_reserve_and_the_fence(name: str) -> None:
    for seed in range(6):
        sc = make_scenario(seed)
        ep = Episode(sc)
        choose = BASELINES[name]()
        reserve = CFG.reserve_frac * CFG.battery_min
        while not ep.done:
            sector = choose(ep.observe())
            assert sc.allowed.ravel()[sector]
            ep.step(sector)
            if ep.done:
                break
            # After every survey the drone can still get home above the reserve.
            assert ep.battery - ep.home_min[sector] >= reserve - 1e-9
        r = ep.result()
        assert 0.0 <= r.p_find <= 1.0
        assert 0.0 <= r.score <= 1.0
        assert r.expected_minutes <= CFG.horizon_min + 1e-9
        assert ep.t <= CFG.horizon_min + CFG.battery_min


def test_a_person_standing_in_the_first_sector_is_likely_found() -> None:
    sc = make_scenario(5)
    ep = Episode(sc)
    sector = int(np.flatnonzero(ep.observe().feasible)[0])
    k, s = CFG.sector_cells, CFG.n_sectors
    r0, c0 = (sector // s) * k, (sector % s) * k
    centre = np.array([(c0 + k / 2) * CFG.cell_m, (r0 + k / 2) * CFG.cell_m])
    path = np.tile(centre, (len(sc.path_m), 1))
    pod = np.ones_like(sc.pod_true)
    standing = type(sc)(**{**sc.__dict__, "path_m": path, "pod_true": pod})
    greedy = BASELINES["greedy_psr"]()

    def choose(obs: Observation) -> int:
        return sector if not obs.swept.any() else greedy(obs)

    r = run_episode(standing, choose)
    assert r.p_find == pytest.approx(1.0)
    assert r.expected_minutes < 10
