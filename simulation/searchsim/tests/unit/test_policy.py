"""Baselines, the learned scorer, training, evaluation and the command line."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest

from sd_searchsim import cli, policies
from sd_searchsim.config import EVAL_SEED_BASE, SIZES, VAL_SEED_BASE, RunSize
from sd_searchsim.env import Episode, run_episode
from sd_searchsim.evaluate import evaluate
from sd_searchsim.policies import (
    BASELINES,
    initial_params,
    lawnmower_order,
    scorer,
    spiral_order,
)
from sd_searchsim.scenario import make_scenario
from sd_searchsim.train import MAX_SEED, train, train_seeds

TINY = RunSize(
    iterations=2, population=4, train_scenarios=2, val_scenarios=3, eval_scenarios=3, val_every=1
)


@pytest.mark.parametrize("corner", range(4))
def test_lawnmower_visits_every_sector_by_neighbour_steps(corner: int) -> None:
    order = lawnmower_order(8, corner)
    assert sorted(order.tolist()) == list(range(64))
    rows, cols = order // 8, order % 8
    assert (np.abs(np.diff(rows)) + np.abs(np.diff(cols)) == 1).all()


def test_spiral_starts_at_the_centre_and_covers_all() -> None:
    order = spiral_order(8, (2, 5))
    assert order[0] == 2 * 8 + 5
    assert sorted(order.tolist()) == list(range(64))


def test_untrained_scorer_is_greedy_psr() -> None:
    sc = make_scenario(11)
    a = run_episode(sc, scorer(initial_params()))
    b = run_episode(sc, BASELINES["greedy_psr"]())
    assert a == b


def test_random_scorer_only_picks_feasible_sectors() -> None:
    theta = np.random.default_rng(0).normal(size=initial_params().size) * 3
    for seed in range(5):
        ep = Episode(make_scenario(seed))
        choose = scorer(theta)
        while not ep.done:
            obs = ep.observe()
            sector = choose(obs)
            assert obs.feasible[sector]
            ep.step(sector)


def test_save_and_load(tmp_path: Path) -> None:
    theta = np.arange(initial_params().size, dtype=float)
    path = tmp_path / "p.json"
    policies.save(path, theta, {"note": "test"})
    assert np.array_equal(policies.load(path), theta)
    doc = json.loads(path.read_text())
    doc["features"] = ["other"]
    path.write_text(json.dumps(doc))
    with pytest.raises(ValueError, match="other features"):
        policies.load(path)


def test_training_is_deterministic_and_never_worse_than_greedy() -> None:
    a, log_a = train(TINY, seed=1, workers=1)
    b, _ = train(TINY, seed=1, workers=1)
    assert np.array_equal(a, b)
    assert log_a.best_val >= log_a.start_val
    assert len(log_a.iterations) == TINY.iterations


def test_evaluation_compares_on_the_same_scenarios() -> None:
    res = evaluate(3, initial_params(), workers=1)
    assert set(res["choosers"]) == {*BASELINES, "policy"}
    # The untrained policy is greedy_psr, so it ties on every scenario.
    assert res["policy_vs"]["greedy_psr"]["ties"] == 3
    assert res["policy_vs"]["greedy_psr"]["score_diff"]["mean"] == 0
    assert evaluate(3, initial_params(), workers=1) == res


def test_cli_train_then_eval(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setitem(SIZES, "ci", TINY)
    pol, out = tmp_path / "policy.json", tmp_path / "eval.json"
    assert cli.main(["train", "size=ci", "workers=1", f"out={pol}"]) == 0
    assert cli.main(["eval", "size=ci", "workers=1", f"policy={pol}", f"out={out}"]) == 0
    doc = json.loads(out.read_text())
    assert doc["scenarios"] == 3
    assert "policy_vs" in doc


def test_cli_eval_without_policy_compares_baselines(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setitem(SIZES, "ci", TINY)
    monkeypatch.chdir(tmp_path)
    assert cli.main(["eval", "size=ci", "workers=1", "out=e.json"]) == 0
    doc = json.loads((tmp_path / "e.json").read_text())
    assert set(doc["choosers"]) == set(BASELINES)
    assert doc["policy_file"] is None


@pytest.mark.parametrize(
    "argv",
    [
        ["train", "foo=1"],
        ["train", "size=huge"],
        ["eval", "nokey"],
        ["fly"],
        ["train", "seed=x"],
        ["eval", "miss_correlation=x"],
        ["eval", "miss_correlation=1.5"],
    ],
)
def test_cli_rejects_bad_arguments(argv: list[str]) -> None:
    with pytest.raises(SystemExit) as exc:
        cli.main(argv)
    assert exc.value.code == 2


def test_cli_eval_missing_named_policy(tmp_path: Path) -> None:
    with pytest.raises(SystemExit):
        cli.main(["eval", "size=ci", f"policy={tmp_path / 'none.json'}"])


@pytest.mark.parametrize("name", list(SIZES))
def test_training_seeds_never_reach_validation_or_evaluation(name: str) -> None:
    size = SIZES[name]
    used = {
        s
        for seed in (0, MAX_SEED)
        for it in (1, size.iterations)
        for s in train_seeds(size, seed, it)
    }
    val = range(VAL_SEED_BASE, VAL_SEED_BASE + size.val_scenarios)
    ev = range(EVAL_SEED_BASE, EVAL_SEED_BASE + size.eval_scenarios)
    assert max(used) < VAL_SEED_BASE
    assert max(val) < EVAL_SEED_BASE
    assert not used & set(val) and not used & set(ev)


def test_a_seed_outside_the_training_range_is_refused(capsys: pytest.CaptureFixture[str]) -> None:
    with pytest.raises(ValueError, match="seed"):
        train_seeds(SIZES["ci"], MAX_SEED + 1, 1)
    with pytest.raises(SystemExit) as exc:
        cli.main(["train", "size=ci", f"seed={MAX_SEED + 1}"])
    assert exc.value.code == 2
    assert "seed" in capsys.readouterr().err
