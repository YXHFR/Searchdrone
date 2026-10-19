"""Who picks the next sector: the fixed patterns, the greedy rules and the learned scorer.

Every chooser sees the same ``Observation`` and may only return a feasible sector
(permitted and fits the battery); ``Episode.step`` refuses anything else. The learned
policy is a small scorer applied to each sector's features: a linear part plus a
16-unit hidden layer, so it works on any number of sectors and needs only numpy to run.
It starts out equal to ``greedy_psr`` (all weight on the POS-per-minute feature), so
training only has to find what improves on that rule.
"""

from __future__ import annotations

import json
import math
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from sd_searchsim.env import FEATURES, N_FEATURES, Chooser, Observation

HIDDEN = 16
FORMAT = "sd-searchsim-policy/1"


def _first_in_order(order: np.ndarray) -> Chooser:
    """Fly a fixed order; skip what does not fit now; least-surveyed sectors first."""

    def choose(obs: Observation) -> int:
        ranked = order[np.argsort(obs.swept[order], kind="stable")]
        for sector in ranked:
            if obs.feasible[sector]:
                return int(sector)
        raise ValueError("no feasible sector")

    return choose


def lawnmower_order(n: int, start_corner: int = 0) -> np.ndarray:
    """Boustrophedon over the sector grid: row by row, every other row reversed."""
    rows = [list(range(r * n, r * n + n)) for r in range(n)]
    order = [s for i, row in enumerate(rows) for s in (row if i % 2 == 0 else row[::-1])]
    arr = np.array(order)
    if start_corner & 1:  # start from the east side
        arr = (arr // n) * n + (n - 1 - arr % n)
    if start_corner & 2:  # start from the north side
        arr = (n - 1 - arr // n) * n + arr % n
    return arr


def spiral_order(n: int, centre: tuple[int, int]) -> np.ndarray:
    """Square spiral outwards from the sector that holds the last seen point."""
    r, c = centre
    order = [r * n + c]
    seen = {order[0]}
    step, d = 1, 0
    dirs = ((0, 1), (1, 0), (0, -1), (-1, 0))
    while len(seen) < n * n:
        for _ in range(2):
            dr, dc = dirs[d % 4]
            for _ in range(step):
                r, c = r + dr, c + dc
                if 0 <= r < n and 0 <= c < n and r * n + c not in seen:
                    seen.add(r * n + c)
                    order.append(r * n + c)
            d += 1
        step += 1
    return np.array(order)


def _sector_side(obs: Observation) -> tuple[int, float]:
    n = math.isqrt(len(obs.feasible))
    side = float(obs.sector_centres_m[1, 0] - obs.sector_centres_m[0, 0])
    return n, side


def lawnmower() -> Callable[[Observation], int]:
    """Lawnmower from the grid corner nearest home (chosen on the first call)."""
    state: dict[str, Chooser] = {}

    def choose(obs: Observation) -> int:
        if "c" not in state:
            n, side = _sector_side(obs)
            size = n * side
            corner = int(obs.home_m[0] > size / 2) | (int(obs.home_m[1] > size / 2) << 1)
            state["c"] = _first_in_order(lawnmower_order(n, corner))
        return state["c"](obs)

    return choose


def spiral() -> Callable[[Observation], int]:
    state: dict[str, Chooser] = {}

    def choose(obs: Observation) -> int:
        if "c" not in state:
            n, side = _sector_side(obs)
            centre = (
                min(int(obs.ipp_m[1] // side), n - 1),
                min(int(obs.ipp_m[0] // side), n - 1),
            )
            state["c"] = _first_in_order(spiral_order(n, centre))
        return state["c"](obs)

    return choose


def _argmax_feasible(values: np.ndarray, feasible: np.ndarray) -> int:
    return int(np.argmax(np.where(feasible, values, -np.inf)))


def greedy_poa() -> Chooser:
    """Greedy by probability: the sector with the most belief mass."""
    i = FEATURES.index("poa")
    return lambda obs: _argmax_feasible(obs.features[:, i], obs.feasible)


def greedy_psr() -> Chooser:
    """Greedy by expected detection per minute of transit and survey."""
    i = FEATURES.index("psr")
    return lambda obs: _argmax_feasible(obs.features[:, i], obs.feasible)


@dataclass(frozen=True)
class ScorerShape:
    n_features: int = N_FEATURES
    hidden: int = HIDDEN

    @property
    def size(self) -> int:
        f, h = self.n_features, self.hidden
        return f + h * f + h + h


def initial_params(shape: ScorerShape | None = None) -> np.ndarray:
    shape = shape or ScorerShape()
    theta = np.zeros(shape.size)
    theta[FEATURES.index("psr")] = 1.0
    return theta


def scores(theta: np.ndarray, features: np.ndarray, shape: ScorerShape | None = None) -> np.ndarray:
    shape = shape or ScorerShape()
    f, h = shape.n_features, shape.hidden
    v = theta[:f]
    w1 = theta[f : f + h * f].reshape(h, f)
    b1 = theta[f + h * f : f + h * f + h]
    w2 = theta[f + h * f + h :]
    return features @ v + np.tanh(features @ w1.T + b1) @ w2


def scorer(theta: np.ndarray) -> Chooser:
    """The learned policy: the feasible sector with the highest score."""
    return lambda obs: _argmax_feasible(scores(theta, obs.features), obs.feasible)


BASELINES: dict[str, Callable[[], Chooser]] = {
    "lawnmower": lawnmower,
    "spiral": spiral,
    "greedy_poa": greedy_poa,
    "greedy_psr": greedy_psr,
}


def save(path: Path, theta: np.ndarray, meta: dict[str, object]) -> None:
    """Weights as JSON next to how they were made; kept out of git (build/)."""
    path.parent.mkdir(parents=True, exist_ok=True)
    doc = {
        "format": FORMAT,
        "features": list(FEATURES),
        "hidden": HIDDEN,
        "theta": [float(x) for x in theta],
        "meta": meta,
    }
    path.write_text(json.dumps(doc, indent=1) + "\n", encoding="utf-8")


def load(path: Path) -> np.ndarray:
    doc = json.loads(path.read_text(encoding="utf-8"))
    if doc.get("format") != FORMAT or doc.get("features") != list(FEATURES):
        raise ValueError(f"{path}: policy was trained for other features; train again")
    if doc.get("hidden") != HIDDEN:
        raise ValueError(f"{path}: policy has another network size; train again")
    theta = np.asarray(doc["theta"], dtype=float)
    if theta.shape != (ScorerShape().size,) or not np.all(np.isfinite(theta)):
        raise ValueError(f"{path}: broken weights")
    return theta
