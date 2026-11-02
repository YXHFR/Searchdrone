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

With ``checkpoint`` set, the whole state (weights, Adam moments, the random generator
and the log) is written there after every validation, and a run that finds one picks
up after its last saved iteration. A resumed run ends with the same weights as one
that was never stopped. The checkpoint names the run (size, world, seed, step sizes),
not the code version: resume with the code it was started with. ``train`` leaves the
file in place when done; the command line removes it.
"""

from __future__ import annotations

import dataclasses
import json
import os
import time
from collections.abc import Callable, Sequence
from concurrent.futures import Executor, ProcessPoolExecutor
from dataclasses import dataclass, field
from pathlib import Path

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


SEED_STRIDE = 1_000_000  # training scenarios per run seed, so runs never share scenarios
MAX_SEED = (VAL_SEED_BASE - TRAIN_SEED_BASE) // SEED_STRIDE - 1


def train_seeds(size: RunSize, seed: int, it: int) -> range:
    """Scenario seeds for ES iteration ``it`` (1-based); never in the validation range."""
    if not 0 <= seed <= MAX_SEED:
        raise ValueError(f"seed must be 0..{MAX_SEED}")
    if (size.iterations + 1) * size.train_scenarios > SEED_STRIDE:
        raise ValueError("run size too large for the training seed range")
    base = TRAIN_SEED_BASE + seed * SEED_STRIDE + it * size.train_scenarios
    return range(base, base + size.train_scenarios)


def _run_id(size: RunSize, cfg: SimConfig, seed: int, sigma: float, lr: float) -> str:
    """What a checkpoint must match to be resumed: the same run, not just the same file."""
    doc = {
        "size": dataclasses.asdict(size),
        "cfg": dataclasses.asdict(cfg),
        "seed": seed,
        "sigma": sigma,
        "lr": lr,
    }
    return json.dumps(doc, sort_keys=True, default=str)


def _save_checkpoint(
    path: Path,
    run_id: str,
    it: int,
    arrays: dict[str, np.ndarray],
    rng: np.random.Generator,
    log: TrainLog,
) -> None:
    meta = {
        "run_id": run_id,
        "iteration": it,
        "rng": rng.bit_generator.state,
        "log": dataclasses.asdict(log),
    }
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name + ".tmp")
    with tmp.open("wb") as f:
        np.savez(
            f,
            meta=np.array(json.dumps(meta)),
            theta=arrays["theta"],
            m=arrays["m"],
            v=arrays["v"],
            best=arrays["best"],
        )
    tmp.replace(path)  # a stop mid-write leaves the previous checkpoint whole


def _load_checkpoint(path: Path, run_id: str) -> tuple[int, dict[str, np.ndarray], dict, TrainLog]:
    with np.load(path) as z:
        meta = json.loads(str(z["meta"]))
        arrays = {k: z[k] for k in ("theta", "m", "v", "best")}
    if meta["run_id"] != run_id:
        raise ValueError(
            f"{path} är från en annan körning (storlek, värld, seed eller inställning skiljer); "
            "ta bort den eller välj en annan checkpoint."
        )
    return int(meta["iteration"]), arrays, meta["rng"], TrainLog(**meta["log"])


def train(
    size: RunSize,
    cfg: SimConfig | None = None,
    seed: int = 0,
    workers: int | None = None,
    sigma: float = 0.05,
    lr: float = 0.02,
    progress: Callable[[str], None] | None = None,
    checkpoint: Path | None = None,
) -> tuple[np.ndarray, TrainLog]:
    cfg = cfg or SimConfig()
    if size.population % 2:
        raise ValueError("population must be even (antithetic pairs)")
    train_seeds(size, seed, size.iterations)  # fail before any work on a bad seed
    rng = np.random.default_rng(seed)
    theta = initial_params()
    m = np.zeros_like(theta)
    v = np.zeros_like(theta)
    b1, b2, eps_adam = 0.9, 0.999, 1e-8
    val_seeds = [VAL_SEED_BASE + i for i in range(size.val_scenarios)]
    log = TrainLog()
    run_id = _run_id(size, cfg, seed, sigma, lr)
    first, resumed = 1, None
    if checkpoint is not None and checkpoint.exists():
        done, arrays, rng_state, log = _load_checkpoint(checkpoint, run_id)
        theta, m, v = arrays["theta"], arrays["m"], arrays["v"]
        rng.bit_generator.state = rng_state
        resumed, first = arrays["best"], done + 1
        if progress:
            progress(f"fortsätter från varv {done}/{size.iterations} ({checkpoint})")
    t0 = time.monotonic() - log.seconds
    workers = workers or os.cpu_count() or 1
    pool = ProcessPoolExecutor(workers) if workers > 1 else None
    try:
        if resumed is not None:
            best = resumed
        else:
            best = theta.copy()
            start = float(score_candidates(pool, theta[None], val_seeds, cfg)[0])
            log.start_val = log.best_val = start
        half = size.population // 2
        for it in range(first, size.iterations + 1):
            eps = rng.standard_normal((half, theta.size))
            cands = np.concatenate([theta + sigma * eps, theta - sigma * eps])
            fit = score_candidates(pool, cands, train_seeds(size, seed, it), cfg)
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
            if checkpoint is not None and "val" in row:
                log.seconds = time.monotonic() - t0
                arrays = {"theta": theta, "m": m, "v": v, "best": best}
                _save_checkpoint(checkpoint, run_id, it, arrays, rng, log)
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
