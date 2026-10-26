"""One seeded search scenario: terrain, the prior map, the true path of the missing person.

Everything is drawn from ``numpy.random.default_rng(seed)``, so a seed always gives
the same scenario. The belief (the prior the searcher plans with) and the truth (where
the person really starts and how they move) are drawn from *different* settings, on
purpose: a real probability map is never exactly right, and a policy that only works
when it is would be useless.

The motion model is a mixture of the lost-person strategies in Hashimoto et al. (2022,
"An agent-based model reveals lost person behavior based on data from wilderness
search and rescue", Sci. Rep. 12:5873): stay put, random walk, route travel (follow a
track) and direction travel. Most people stop at some point. The mixture weights here
are round defaults, not the paper's fitted values.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np

from sd_searchsim.config import SimConfig

STRATEGIES = ("stay", "random_walk", "route", "direction")
_STRATEGY_P = np.array([0.25, 0.35, 0.25, 0.15])


@dataclass(frozen=True)
class Scenario:
    seed: int
    canopy: np.ndarray  # (n, n) 0..0.9, share of the ground hidden under trees
    track_dist_m: np.ndarray  # (n, n) distance to the nearest track
    prior: np.ndarray  # (n, n) the belief at the start, sums to 1
    allowed: np.ndarray  # (s, s) bool, sectors inside the permitted search area
    ipp_m: np.ndarray  # (2,) last seen point (x east, y north)
    home_m: np.ndarray  # (2,) take-off and landing point
    path_m: np.ndarray  # (horizon + 1, 2) true position at each whole minute
    pod_true: np.ndarray  # (n, n) real chance a sweep sees a person in the cell
    pod_belief: np.ndarray  # (n, n) what the searcher believes that chance is
    strategy: str


def cell_centres(cfg: SimConfig) -> tuple[np.ndarray, np.ndarray]:
    c = (np.arange(cfg.grid_cells) + 0.5) * cfg.cell_m
    yy, xx = np.meshgrid(c, c, indexing="ij")  # row = y (north), column = x (east)
    return xx, yy


def _smooth_noise(rng: np.random.Generator, n: int, coarse: int) -> np.ndarray:
    """Values in 0..1 that change smoothly over the grid (bilinear upsampling)."""
    low = rng.random((coarse + 1, coarse + 1))
    pos = np.linspace(0, coarse, n)
    i0 = np.minimum(pos.astype(int), coarse - 1)
    f = pos - i0
    rows = low[i0] * (1 - f)[:, None] + low[i0 + 1] * f[:, None]
    return rows[:, i0] * (1 - f)[None, :] + rows[:, i0 + 1] * f[None, :]


def _tracks(rng: np.random.Generator, n: int) -> np.ndarray:
    """1-3 tracks as wandering lines from one edge of the grid to another."""
    mask = np.zeros((n, n), dtype=bool)
    for _ in range(int(rng.integers(1, 4))):
        edge = int(rng.integers(4))
        start = rng.uniform(0, n - 1)
        pos = {
            0: np.array([0.0, start]),
            1: np.array([n - 1.0, start]),
            2: np.array([start, 0.0]),
            3: np.array([start, n - 1.0]),
        }[edge]
        heading = {0: 0.5, 1: -0.5, 2: 0.0, 3: 1.0}[edge] * math.pi + rng.normal(0, 0.3)
        for _ in range(4 * n):
            r, c = round(float(pos[0])), round(float(pos[1]))
            if not (0 <= r < n and 0 <= c < n):
                break
            mask[r, c] = True
            heading += rng.normal(0, 0.15)
            pos = pos + 0.5 * np.array([math.sin(heading), math.cos(heading)])
    return mask


def _distance_to(mask: np.ndarray, cell_m: float) -> np.ndarray:
    n = mask.shape[0]
    pts = np.argwhere(mask).astype(float)
    if len(pts) == 0:
        return np.full((n, n), np.inf)
    rr, cc = np.meshgrid(np.arange(n, dtype=float), np.arange(n, dtype=float), indexing="ij")
    grid = np.stack([rr.ravel(), cc.ravel()], axis=1)
    best = np.full(len(grid), np.inf)
    for chunk in np.array_split(pts, max(1, len(pts) // 64)):
        d2 = ((grid[:, None, :] - chunk[None, :, :]) ** 2).sum(-1).min(1)
        best = np.minimum(best, d2)
    return np.sqrt(best).reshape(n, n) * cell_m


def ring_prior(
    cfg: SimConfig,
    ipp_m: np.ndarray,
    r50_m: float,
    track_dist_m: np.ndarray,
    track_weight: float,
) -> np.ndarray:
    """Distance-ring prior (log-normal distance from the last seen point) times terrain."""
    xx, yy = cell_centres(cfg)
    r = np.hypot(xx - ipp_m[0], yy - ipp_m[1])
    r = np.maximum(r, cfg.cell_m / 2)
    sigma = 0.8
    radial = np.exp(-((np.log(r / r50_m)) ** 2) / (2 * sigma**2)) / (r * sigma)
    density = radial / (2 * math.pi * r)  # spread each ring over its area
    weight = 1.0 + track_weight * np.exp(-track_dist_m / 50.0)
    p = density * weight
    return p / p.sum()


def _sample_cell(rng: np.random.Generator, p: np.ndarray, cfg: SimConfig) -> np.ndarray:
    idx = int(rng.choice(p.size, p=p.ravel()))
    r, c = divmod(idx, p.shape[1])
    return np.array([(c + rng.random()) * cfg.cell_m, (r + rng.random()) * cfg.cell_m])


def _walk(
    rng: np.random.Generator,
    cfg: SimConfig,
    start: np.ndarray,
    strategy: str,
    track_dist_m: np.ndarray,
) -> np.ndarray:
    """True position per whole minute over the horizon."""
    steps = int(cfg.horizon_min)
    size = cfg.grid_cells * cfg.cell_m
    out = np.empty((steps + 1, 2))
    pos = start.copy()
    out[0] = pos
    stop_at = rng.exponential(rng.uniform(20, 120))
    speed = rng.uniform(0.2, 0.7) * 60.0  # metres per minute while moving
    heading = rng.uniform(0, 2 * math.pi)
    gy, gx = np.gradient(track_dist_m)
    for t in range(1, steps + 1):
        if strategy != "stay" and t < stop_at:
            if strategy == "random_walk":
                heading = rng.uniform(0, 2 * math.pi)
            elif strategy == "direction":
                heading += rng.normal(0, 0.1)
            else:  # route: head down the distance field onto a track, then follow it
                r = min(int(pos[1] // cfg.cell_m), cfg.grid_cells - 1)
                c = min(int(pos[0] // cfg.cell_m), cfg.grid_cells - 1)
                if track_dist_m[r, c] > cfg.cell_m and np.isfinite(track_dist_m[r, c]):
                    heading = math.atan2(-gy[r, c], -gx[r, c]) + rng.normal(0, 0.3)
                else:
                    heading += rng.normal(0, 0.3)
            pos = pos + speed * np.array([math.cos(heading), math.sin(heading)])
            for k in range(2):  # reflect at the grid edge
                if pos[k] < 0:
                    pos[k] = -pos[k]
                    heading = math.pi - heading if k == 0 else -heading
                elif pos[k] >= size:
                    pos[k] = 2 * size - pos[k] - 1e-6
                    heading = math.pi - heading if k == 0 else -heading
        out[t] = pos
    return out


def no_fly(rng: np.random.Generator, s: int) -> np.ndarray:
    """Permitted sectors: sometimes a no-fly band (another airspace user, a road
    closure) along one edge."""
    allowed = np.ones((s, s), dtype=bool)
    if rng.random() < 0.4:
        side = int(rng.integers(4))
        if side == 0:
            allowed[0, :] = False
        elif side == 1:
            allowed[-1, :] = False
        elif side == 2:
            allowed[:, 0] = False
        else:
            allowed[:, -1] = False
    return allowed


def make_scenario(seed: int, cfg: SimConfig | None = None) -> Scenario:
    cfg = cfg or SimConfig()
    if cfg.world == "real":
        from sd_searchsim.realworld import make_real_scenario  # here: it imports this module

        return make_real_scenario(seed, cfg)
    rng = np.random.default_rng(seed)
    n = cfg.grid_cells
    size = n * cfg.cell_m

    tracks = _tracks(rng, n)
    track_dist = _distance_to(tracks, cfg.cell_m)
    canopy = np.clip(_smooth_noise(rng, n, 6) * 1.3 - 0.2, 0.0, 0.9)
    canopy[tracks] *= 0.3  # a track is mostly open

    ipp = rng.uniform(0.3 * size, 0.7 * size, 2)
    # Home stays clear of the no-fly band below, so every straight transit between
    # permitted sectors and home stays inside the permitted (convex) area.
    home = np.clip(ipp + rng.normal(0, 150, 2), 300.0, size - 300.0)

    # Belief: a category median distance; truth: off by up to about 1.5x either way.
    r50_belief = float(rng.uniform(300, 900))
    r50_true = r50_belief * float(np.exp(rng.normal(0, 0.3)))
    prior = ring_prior(cfg, ipp, r50_belief, track_dist, cfg.track_weight)
    truth = ring_prior(cfg, ipp, r50_true, track_dist, float(rng.uniform(0.5, 4.0)))
    start = _sample_cell(rng, truth, cfg)
    strategy = str(rng.choice(STRATEGIES, p=_STRATEGY_P))
    path = _walk(rng, cfg, start, strategy, track_dist)

    p_det_true = float(rng.uniform(0.55, 0.9))
    pod_true = p_det_true * (1.0 - canopy)
    pod_belief = cfg.belief_p_det * (1.0 - canopy)

    allowed = no_fly(rng, cfg.n_sectors)

    return Scenario(
        seed=seed,
        canopy=canopy,
        track_dist_m=track_dist,
        prior=prior,
        allowed=allowed,
        ipp_m=ipp,
        home_m=home,
        path_m=path,
        pod_true=pod_true,
        pod_belief=pod_belief,
        strategy=strategy,
    )
