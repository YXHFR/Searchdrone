"""The search episode: pick a sector, fly there, survey it, update the map, repeat.

One decision = one sector. The drone flies straight to the sector centre, surveys the
sector with a lawnmower pattern, and the belief map gets the Bayes update for a sweep
that found nothing. A sector is only *feasible* when it is permitted (inside the search
area, outside no-fly bands), and the transit, the survey and the flight home fit in the
battery above the reserve, and in the mission time. When nothing fits, the drone flies
home and changes battery. The episode ends at the mission horizon.

Scoring uses the true path. Given the path, the outcome of each sweep is a coin flip
with the true detection chance, and the searcher's choices do not depend on the coins
(a find ends the search; until then every sweep came back empty). So the episode runs
once as if nothing was found, and the score is the exact expectation over the coins:

- ``p_find``: chance the person was found within the horizon.
- ``expected_minutes``: expected time to find, counted as the horizon when not found.
- ``score``: ``1 - expected_minutes / horizon``; higher is better, and it rewards
  finding early, not only finding.
"""

from __future__ import annotations

import functools
from collections.abc import Callable
from dataclasses import dataclass

import numpy as np

from sd_searchsim.config import SimConfig
from sd_searchsim.scenario import Scenario

# Per-sector features, in this order (see ``Episode.observe``).
FEATURES = (
    "pos",  # expected detection if surveyed now (belief), / max over sectors
    "psr",  # pos per minute of transit + survey, / max
    "poa",  # belief mass in the sector, / max
    "neigh_pos",  # pos of the 8 neighbouring sectors, / max
    "pod_mean",  # believed mean detection chance in the sector
    "transit",  # transit minutes / battery minutes
    "home",  # minutes from the sector centre home / battery minutes
    "swept",  # log(1 + times surveyed)
    "battery",  # battery left / battery minutes (same for every sector)
    "time",  # mission time used / horizon (same for every sector)
)
N_FEATURES = len(FEATURES)


@dataclass(frozen=True)
class Observation:
    features: np.ndarray  # (n_sectors**2, N_FEATURES)
    feasible: np.ndarray  # (n_sectors**2,) bool
    swept: np.ndarray  # (n_sectors**2,) int
    sector_centres_m: np.ndarray  # (n_sectors**2, 2)
    position_m: np.ndarray  # (2,)
    ipp_m: np.ndarray  # (2,)
    home_m: np.ndarray  # (2,)


Chooser = Callable[[Observation], int]


@dataclass(frozen=True)
class EpisodeResult:
    seed: int
    score: float
    p_find: float
    expected_minutes: float
    sectors_surveyed: int
    sorties: int


@functools.lru_cache(maxsize=256)
def _blur_matrix(n: int, sigma_cells: float) -> np.ndarray:
    """1D Gaussian blur as an (n, n) matrix, reflecting at the edges; columns sum to 1."""
    radius = max(1, int(np.ceil(3 * sigma_cells)))
    offsets = np.arange(-radius, radius + 1)
    k = np.exp(-(offsets.astype(float) ** 2) / (2 * sigma_cells**2))
    k /= k.sum()
    g = np.zeros((n, n))
    for j in range(n):
        i = j + offsets
        while (i < 0).any() or (i >= n).any():  # reflect until inside (wide kernels)
            i = np.where(i < 0, -i - 1, i)
            i = np.where(i >= n, 2 * n - i - 1, i)
        np.add.at(g[:, j], i, k)
    g.setflags(write=False)
    return g


def diffuse(p: np.ndarray, sigma_cells: float) -> np.ndarray:
    """Gaussian spread with reflecting edges; the total mass is kept."""
    if sigma_cells < 0.05:
        return p
    g = _blur_matrix(p.shape[0], round(float(sigma_cells), 2))
    return g @ p @ g.T


class Episode:
    def __init__(self, scenario: Scenario, cfg: SimConfig | None = None) -> None:
        self.cfg = cfg = cfg or SimConfig()
        self.sc = scenario
        s = cfg.n_sectors
        self.n_sec = s * s
        c = (np.arange(s) + 0.5) * cfg.sector_m
        yy, xx = np.meshgrid(c, c, indexing="ij")
        self.centres = np.stack([xx.ravel(), yy.ravel()], axis=1)
        self.allowed = scenario.allowed.ravel()
        self.home_min = np.linalg.norm(self.centres - scenario.home_m, axis=1) / (
            cfg.transit_speed_mps * 60.0
        )
        self.belief = scenario.prior.copy()
        self.pos = scenario.home_m.astype(float).copy()
        self.t = 0.0
        self.battery = cfg.battery_min
        self.swept = np.zeros(self.n_sec, dtype=int)
        self.survive = 1.0  # chance the person is still not found
        self.expected_minutes = 0.0
        self.sorties = 1
        self.surveyed = 0
        self.done = False
        self._ensure_feasible()

    # --- geometry -----------------------------------------------------------------

    def _sector_sum(self, grid: np.ndarray) -> np.ndarray:
        s, k = self.cfg.n_sectors, self.cfg.sector_cells
        return grid.reshape(s, k, s, k).sum(axis=(1, 3)).ravel()

    def _transit_min(self, to: np.ndarray) -> np.ndarray:
        return np.linalg.norm(to - self.pos, axis=-1) / (self.cfg.transit_speed_mps * 60.0)

    def feasible(self) -> np.ndarray:
        cfg = self.cfg
        transit = self._transit_min(self.centres)
        need = transit + cfg.sweep_min + self.home_min
        usable = self.battery - cfg.reserve_frac * cfg.battery_min
        in_time = self.t + transit + cfg.sweep_min <= cfg.horizon_min
        return self.allowed & (need <= usable) & in_time

    # --- the map ------------------------------------------------------------------

    def _advance(self, minutes: float) -> None:
        """Move the clock; the belief spreads (variances add, so one blur per call)."""
        sigma_cells = self.cfg.belief_diffusion_m * np.sqrt(minutes) / self.cfg.cell_m
        self.belief = diffuse(self.belief, sigma_cells)
        self.t += minutes

    def _true_cell(self, t: float) -> tuple[int, int]:
        path = self.sc.path_m
        i = min(int(t), len(path) - 2)
        f = min(max(t - i, 0.0), 1.0)
        x, y = path[i] * (1 - f) + path[i + 1] * f
        n = self.cfg.grid_cells
        return (
            min(int(y // self.cfg.cell_m), n - 1),
            min(int(x // self.cfg.cell_m), n - 1),
        )

    def _sector_of(self, cell: tuple[int, int]) -> int:
        k, s = self.cfg.sector_cells, self.cfg.n_sectors
        return (cell[0] // k) * s + cell[1] // k

    # --- the loop -----------------------------------------------------------------

    def observe(self) -> Observation:
        cfg = self.cfg
        pod_b = self.sc.pod_belief
        pos = self._sector_sum(self.belief * pod_b)
        poa = self._sector_sum(self.belief)
        transit = self._transit_min(self.centres)
        psr = pos / (transit + cfg.sweep_min)
        s = cfg.n_sectors
        padded = np.pad(pos.reshape(s, s), 1)
        neigh = np.sum(
            [
                padded[1 + dr : 1 + dr + s, 1 + dc : 1 + dc + s]
                for dr in (-1, 0, 1)
                for dc in (-1, 0, 1)
                if dr or dc
            ],
            axis=0,
        ).ravel()
        k2 = cfg.sector_cells**2
        feats = np.stack(
            [
                pos / max(pos.max(), 1e-12),
                psr / max(psr.max(), 1e-12),
                poa / max(poa.max(), 1e-12),
                neigh / max(neigh.max(), 1e-12),
                self._sector_sum(pod_b) / k2,
                transit / cfg.battery_min,
                self.home_min / cfg.battery_min,
                np.log1p(self.swept),
                np.full(self.n_sec, self.battery / cfg.battery_min),
                np.full(self.n_sec, self.t / cfg.horizon_min),
            ],
            axis=1,
        )
        return Observation(
            features=feats,
            feasible=self.feasible(),
            swept=self.swept.copy(),
            sector_centres_m=self.centres,
            position_m=self.pos.copy(),
            ipp_m=self.sc.ipp_m,
            home_m=self.sc.home_m,
        )

    def step(self, sector: int) -> None:
        if self.done:
            raise RuntimeError("episode is over")
        if not self.feasible()[sector]:
            # The shield: an infeasible or forbidden sector is never flown.
            raise ValueError(f"sector {sector} is not permitted or does not fit the battery")
        cfg = self.cfg
        transit = float(self._transit_min(self.centres[sector]))
        # Truth: where the person is in the middle of the survey.
        t_mid = self.t + transit + cfg.sweep_min / 2
        cell = self._true_cell(t_mid)
        if self._sector_of(cell) == sector:
            q = float(self.sc.pod_true[cell])
            found_now = self.survive * q
            self.expected_minutes += found_now * t_mid
            self.survive -= found_now
        self._advance(transit + cfg.sweep_min)
        # Belief: a sweep that found nothing (Bayes).
        k, s = cfg.sector_cells, cfg.n_sectors
        r0, c0 = (sector // s) * k, (sector % s) * k
        b = self.belief.copy()
        b[r0 : r0 + k, c0 : c0 + k] *= 1.0 - self.sc.pod_belief[r0 : r0 + k, c0 : c0 + k]
        self.belief = b / b.sum()
        self.battery -= transit + cfg.sweep_min
        self.pos = self.centres[sector].copy()
        self.swept[sector] += 1
        self.surveyed += 1
        self._ensure_feasible()

    def _ensure_feasible(self) -> None:
        """Fly home and change battery when no sector fits; end at the horizon."""
        if self.feasible().any():
            return
        cfg = self.cfg
        back = float(self._transit_min(self.sc.home_m))
        self._advance(back + cfg.swap_min)
        self.pos = self.sc.home_m.astype(float).copy()
        self.battery = cfg.battery_min
        self.sorties += 1
        if not self.feasible().any():
            self.done = True

    def result(self) -> EpisodeResult:
        h = self.cfg.horizon_min
        expected = self.expected_minutes + self.survive * h
        return EpisodeResult(
            seed=self.sc.seed,
            score=1.0 - expected / h,
            p_find=1.0 - self.survive,
            expected_minutes=expected,
            sectors_surveyed=self.surveyed,
            sorties=self.sorties - (1 if self.done else 0),
        )


def run_episode(scenario: Scenario, choose: Chooser, cfg: SimConfig | None = None) -> EpisodeResult:
    ep = Episode(scenario, cfg)
    while not ep.done:
        ep.step(choose(ep.observe()))
    return ep.result()
