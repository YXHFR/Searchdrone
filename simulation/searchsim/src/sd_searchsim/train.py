"""Train the sector scorer with evolution strategies (Salimans et al. 2017).

Why ES and not PPO: the policy has about 200 weights, an episode is cheap, and the
score is known only at the end. ES needs no value function and no gradients through
the simulator, only many episodes, which spread over CPU cores. A GPU would not help:
the time goes into the simulator, not the network.

Each iteration draws antithetic pairs ``theta +- sigma * eps`` and scores every
candidate on the *same* fresh training scenarios (common random numbers), turns the
scores into centred ranks and takes an Adam step. Every ``val_every`` iterations the
current weights are scored on a fixed validation set, and the best ones are kept. The
starting point is the greedy POS-per-minute rule, so the best checkpoint is never
worse than greedy on the validation set.
"""

from __future__ import annotations

import os
import time
from collections.abc import Callable, Sequence
from concurrent.futures import Executor, ProcessPoolExecutor
from dataclasses import dataclass, field

import numpy as np

from sd_searchsim.config import TRAIN_SEED_BASE, VAL_SEED_BASE, RunSize, SimConfig
from sd_searchsim.env import run_episode
from sd_searchsim.policies import initial_params, scorer
from sd_searchsim.scenario import make_scenario


def _score_seed(args: tuple[int, np.ndarray, SimConfig]) -> np.ndarray:
    """Score every candidate on one scenario (built once per task)."""
    seed, thetas, cfg = args
    sc = make_scenario(seed, cfg)
    return np.array([run_episode(sc, scorer(t), cfg).score for t in thetas])


def score_candidates(
    pool: Executor | None, thetas: np.ndarray, seeds: Sequence[int], cfg: SimConfig
) -> np.ndarray:
    """Mean score per candidate over the seeds, shape (len(thetas),)."""
    tasks = [(s, thetas, cfg) for s in seeds]
    rows = list(pool.map(_score_seed, tasks)) if pool else [_score_seed(t) for t in tasks]
    return np.mean(np.stack(rows), axis=0)


def _centred_ranks(x: np.ndarray) -> np.ndarray:
    ranks = np.empty(len(x))
    ranks[np.argsort(x)] = np.arange(len(x))
    return ranks / (len(x) - 1) - 0.5


@dataclass
class TrainLog:
    iterations: list[dict[str, float]] = field(default_factory=list)
    best_val: float = -np.inf
    start_val: float = float("nan")
    best_iteration: int = 0
    seconds: float = 0.0


def train(
    size: RunSize,
    cfg: SimConfig | None = None,
    seed: int = 0,
    workers: int | None = None,
    sigma: float = 0.05,
    lr: float = 0.02,
    progress: Callable[[str], None] | None = None,
) -> tuple[np.ndarray, TrainLog]:
    cfg = cfg or SimConfig()
    if size.population % 2:
        raise ValueError("population must be even (antithetic pairs)")
    rng = np.random.default_rng(seed)
    theta = initial_params()
    m = np.zeros_like(theta)
    v = np.zeros_like(theta)
    b1, b2, eps_adam = 0.9, 0.999, 1e-8
    val_seeds = [VAL_SEED_BASE + i for i in range(size.val_scenarios)]
    log = TrainLog()
    t0 = time.monotonic()
    workers = workers or os.cpu_count() or 1
    pool = ProcessPoolExecutor(workers) if workers > 1 else None
    try:
        best = theta.copy()
        log.start_val = log.best_val = float(score_candidates(pool, theta[None], val_seeds, cfg)[0])
        half = size.population // 2
        for it in range(1, size.iterations + 1):
            eps = rng.standard_normal((half, theta.size))
            cands = np.concatenate([theta + sigma * eps, theta - sigma * eps])
            base = TRAIN_SEED_BASE + (seed * 1_000_000) + it * size.train_scenarios
            seeds = range(base, base + size.train_scenarios)
            fit = score_candidates(pool, cands, seeds, cfg)
            ranks = _centred_ranks(fit)
            grad = (ranks[:half] - ranks[half:]) @ eps / (size.population * sigma)
            # Adam ascent
            m = b1 * m + (1 - b1) * grad
            v = b2 * v + (1 - b2) * grad**2
            mh, vh = m / (1 - b1**it), v / (1 - b2**it)
            theta = theta + lr * mh / (np.sqrt(vh) + eps_adam)
            row = {"iteration": it, "train_mean": float(fit.mean()), "train_max": float(fit.max())}
            if it % size.val_every == 0 or it == size.iterations:
                val = float(score_candidates(pool, theta[None], val_seeds, cfg)[0])
                row["val"] = val
                if val > log.best_val:
                    log.best_val, log.best_iteration, best = val, it, theta.copy()
            log.iterations.append(row)
            if progress:
                progress(
                    f"varv {it}/{size.iterations}: träning {row['train_mean']:.4f}"
                    + (
                        f", validering {row['val']:.4f} (bäst {log.best_val:.4f})"
                        if "val" in row
                        else ""
                    )
                )
    finally:
        if pool:
            pool.shutdown()
    log.seconds = time.monotonic() - t0
    return best, log
