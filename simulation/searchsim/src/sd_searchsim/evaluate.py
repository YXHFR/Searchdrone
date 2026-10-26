"""Compare the learned policy with the patterns and greedy rules on the same scenarios.

Every chooser flies the same held-out seeded scenarios (never used in training or
validation). The JSON has per-chooser means with 95 % intervals and, for the policy,
the paired difference to each baseline (same scenario, so the scenario's own luck
cancels out) with a 95 % interval and wins/ties/losses.
"""

from __future__ import annotations

import math
from collections.abc import Callable
from concurrent.futures import ProcessPoolExecutor
from typing import Any

import numpy as np

from sd_searchsim.config import EVAL_SEED_BASE, SimConfig
from sd_searchsim.env import Chooser, EpisodeResult, run_episode
from sd_searchsim.policies import BASELINES, scorer
from sd_searchsim.scenario import make_scenario

METRICS = ("score", "p_find", "expected_minutes")


def _choosers(theta: np.ndarray | None) -> dict[str, Callable[[], Chooser]]:
    out: dict[str, Callable[[], Chooser]] = dict(BASELINES)
    if theta is not None:
        out["policy"] = lambda: scorer(theta)
    return out


def _run_seed(
    args: tuple[int, np.ndarray | None, SimConfig],
) -> tuple[str, dict[str, EpisodeResult]]:
    seed, theta, cfg = args
    sc = make_scenario(seed, cfg)
    return sc.strategy, {
        name: run_episode(sc, make(), cfg) for name, make in _choosers(theta).items()
    }


def _mean_ci(x: np.ndarray) -> dict[str, float]:
    mean = float(x.mean())
    half = 1.96 * float(x.std(ddof=1)) / math.sqrt(len(x)) if len(x) > 1 else float("nan")
    return {"mean": mean, "ci95_low": mean - half, "ci95_high": mean + half}


def evaluate(
    n_scenarios: int,
    theta: np.ndarray | None,
    cfg: SimConfig | None = None,
    workers: int = 1,
) -> dict[str, Any]:
    cfg = cfg or SimConfig()
    seeds = [EVAL_SEED_BASE + i for i in range(n_scenarios)]
    tasks = [(s, theta, cfg) for s in seeds]
    if workers > 1:
        with ProcessPoolExecutor(workers) as pool:
            done = list(pool.map(_run_seed, tasks, chunksize=4))
    else:
        done = [_run_seed(t) for t in tasks]
    kinds = [k for k, _ in done]
    rows = [r for _, r in done]
    names = list(rows[0])
    values = {
        name: {m: np.array([getattr(r[name], m) for r in rows]) for m in METRICS} for name in names
    }
    choosers: dict[str, dict[str, Any]] = {}
    for name in names:
        entry: dict[str, Any] = {m: _mean_ci(values[name][m]) for m in METRICS}
        entry["sectors_surveyed_mean"] = float(np.mean([r[name].sectors_surveyed for r in rows]))
        choosers[name] = entry
    out: dict[str, Any] = {
        "scenarios": n_scenarios,
        "seed_first": seeds[0],
        "seed_last": seeds[-1],
        "horizon_min": cfg.horizon_min,
        "choosers": choosers,
        "ranking_by_score": sorted(names, key=lambda n: -choosers[n]["score"]["mean"]),
    }
    if cfg.world == "real":
        # Mean score per lost-person category (who is missing changes where to look).
        out["by_category"] = {
            kind: {
                name: float(
                    np.mean([values[name]["score"][i] for i, k in enumerate(kinds) if k == kind])
                )
                for name in names
            }
            for kind in sorted(set(kinds))
        }
    if "policy" in values:
        paired: dict[str, Any] = {}
        pol = values["policy"]["score"]
        for name in names:
            if name == "policy":
                continue
            d = pol - values[name]["score"]
            paired[name] = {
                "score_diff": _mean_ci(d),
                "wins": int((d > 1e-9).sum()),
                "ties": int((np.abs(d) <= 1e-9).sum()),
                "losses": int((d < -1e-9).sum()),
            }
        out["policy_vs"] = paired
    return out
