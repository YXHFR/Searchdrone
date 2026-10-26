"""world=real: places from ``just rl-maps`` and the lost-person statistics, with made-up
place files (no data, no network)."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest

from sd_searchsim.config import EVAL_SEED_BASE, VAL_SEED_BASE, SimConfig
from sd_searchsim.env import run_episode
from sd_searchsim.policies import BASELINES
from sd_searchsim.realworld import (
    CATEGORIES,
    Category,
    category_map,
    in_split,
    load_sites,
    split_of,
)
from sd_searchsim.scenario import _distance_to, make_scenario

N = SimConfig().grid_cells


@pytest.fixture
def sites_dir(tmp_path: Path) -> Path:
    rng = np.random.default_rng(0)
    entries = []
    for i in range(20):
        track = np.zeros((N, N), dtype=bool)
        track[N // 2, :] = True
        track[:, int(rng.integers(10, N - 10))] = True
        water = np.zeros((N, N), dtype=bool)
        water[:10, :10] = True
        np.savez_compressed(
            tmp_path / f"{i:04d}.npz",
            canopy=rng.uniform(0, 0.75, (N, N)).astype(np.float32),
            track=track,
            water=water,
            built=np.zeros((N, N), dtype=np.float32),
        )
        entries.append({"name": f"{i:04d}", "lat": 60.0, "lon": 15.0})
    (tmp_path / "sites.json").write_text(json.dumps({"sites": entries}))
    return tmp_path


def _cfg(sites_dir: Path) -> SimConfig:
    return SimConfig(world="real", sites_dir=str(sites_dir))


def test_splits_never_share_a_place(sites_dir: Path) -> None:
    sites = load_sites(str(sites_dir), N)
    names = {s: {x.name for x in in_split(sites, s)} for s in ("train", "val", "eval")}
    assert len(names["train"]) == 14 and len(names["val"]) == 2 and len(names["eval"]) == 4
    assert not names["train"] & names["val"]
    assert not names["train"] & names["eval"]
    assert not names["val"] & names["eval"]
    assert split_of(0) == "train"
    assert split_of(VAL_SEED_BASE) == "val"
    assert split_of(EVAL_SEED_BASE + 5) == "eval"


@pytest.mark.parametrize("seed", [0, 1, 2, VAL_SEED_BASE, EVAL_SEED_BASE, EVAL_SEED_BASE + 1])
def test_real_scenario_is_sane(sites_dir: Path, seed: int) -> None:
    cfg = _cfg(sites_dir)
    sc = make_scenario(seed, cfg)
    assert sc.prior.sum() == pytest.approx(1.0)
    assert (sc.prior >= 0).all()
    assert sc.strategy in {c.name for c in CATEGORIES}
    # The person stays at the find location, which is dry land in the grid.
    assert (sc.path_m == sc.path_m[0]).all()
    size = cfg.grid_cells * cfg.cell_m
    assert ((sc.path_m >= 0) & (sc.path_m < size)).all()
    a, b = make_scenario(seed, cfg), make_scenario(seed, cfg)
    assert np.array_equal(a.prior, b.prior) and np.array_equal(a.path_m, b.path_m)
    for make in BASELINES.values():
        r = run_episode(sc, make(), cfg)
        assert 0.0 <= r.score <= 1.0


def test_nobody_in_water(sites_dir: Path) -> None:
    cfg = _cfg(sites_dir)
    for seed in range(10):
        sc = make_scenario(seed, cfg)
        r, c = (sc.path_m[0] // cfg.cell_m).astype(int)[::-1]
        # Water has no mass in the belief or the truth, so the find location has some.
        assert sc.prior[r, c] > 0


@pytest.mark.parametrize("cat", CATEGORIES, ids=lambda c: c.name)
def test_category_map_follows_the_on_track_share(cat: Category) -> None:
    cfg = SimConfig()
    track = np.zeros((N, N), dtype=bool)
    track[N // 2, :] = True
    track[:, N // 2] = True
    dist = _distance_to(track, cfg.cell_m)
    ipp = np.array([N / 2 * cfg.cell_m, N / 2 * cfg.cell_m])
    p = category_map(cfg, ipp, cat, cat.q50_km * 1000, dist, np.zeros((N, N), dtype=bool))
    assert p.sum() == pytest.approx(1.0)
    assert p[track].sum() == pytest.approx(cat.on_track)
    off = ~track
    near = off & (dist <= 100.0)
    expected = cat.within_100m - cat.on_track
    # Not exact: the exponential is spread over cells with the distance rings on top.
    assert p[near].sum() == pytest.approx(expected, abs=0.12)


def test_statistics_are_ordered() -> None:
    for c in CATEGORIES:
        assert c.q25_km < c.q50_km < c.q75_km
        assert 0 < c.on_track < c.within_100m < 1
        assert c.sigma > 0 and c.offset_scale_m > 0
    assert sum(c.share for c in CATEGORIES) == pytest.approx(1.0)


def test_missing_sites_say_what_to_run(tmp_path: Path) -> None:
    with pytest.raises(FileNotFoundError, match="rl-maps"):
        load_sites(str(tmp_path / "none"), N)
