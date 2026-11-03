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

Misses are not independent. The first sweep of a cell sees a person there with ``pod =
p_det * (1 - canopy)``, split into two coins: is the person *visible* from the air in this
cell (``1 - miss_correlation * canopy``; drawn once per scenario and cell, so the trees
hide them the same way on every pass) and, if so, does this sweep see them (the rest,
fresh each pass). After ``n`` empty sweeps with the person in the cell the next sweep's
chance is ``per_pass * P(visible | n misses)`` (``pod_after``), so a repeat sweep is worth
less than the first. The cells' coins are independent of each other, so the truth stays an
exact expectation: it counts the empty sweeps per cell while the person was in it. The
belief cannot know when the person was where, so it counts the sweeps of the sector, as if
the person had stayed put.
"""

from __future__ import annotations

import functools
from collections.abc import Callable
from dataclasses import dataclass

import numpy as np

from sd_motion import MotionState, advance, negative
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


def split_pod(
    pod: np.ndarray, canopy: np.ndarray, miss_correlation: float
) -> tuple[np.ndarray, np.ndarray]:
    """The first sweep's detection chance as ``(visible, per_pass)``, with ``visible *
    per_pass == pod``. ``visible`` lasts for the scenario, ``per_pass`` is fresh each sweep.
    With ``miss_correlation=0`` every person is visible and ``per_pass`` is ``pod`` itself."""
    visible = np.maximum(1.0 - miss_correlation * canopy, 1e-12)
    per_pass = pod / visible
    # A pod above the visible share (only when set by hand): seen whenever visible.
    over = per_pass > 1.0
    return np.where(over, pod, visible), np.where(over, 1.0, per_pass)


def pod_after(visible: np.ndarray, per_pass: np.ndarray, misses: np.ndarray) -> np.ndarray:
    """Chance the next sweep sees the person, after ``misses`` empty sweeps of the cell with
    them in it: ``per_pass`` times the chance they are visible at all, given those misses."""
    missed_if_visible = (1.0 - per_pass) ** misses
    hidden = 1.0 - visible
    still_visible = np.divide(
        visible * missed_if_visible,
        hidden + visible * missed_if_visible,
        out=np.ones_like(missed_if_visible, dtype=float),
        where=hidden > 0,  # always visible: the misses tell nothing
    )
    return per_pass * still_visible


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
        self.motion: MotionState | None = None
        if cfg.belief_motion == "lpb":
            if scenario.motion is None:
                raise ValueError("belief_motion=lpb needs a scenario made with it")
            self.motion = scenario.motion.start
            self.belief = self.motion.p
        self.pos = scenario.home_m.astype(float).copy()
        self.t = 0.0
        self.battery = cfg.battery_min
        self.swept = np.zeros(self.n_sec, dtype=int)
        m = cfg.miss_correlation
        self._true_split = split_pod(scenario.pod_true, scenario.canopy, m)
        self._belief_split = split_pod(scenario.pod_belief, scenario.canopy, m)
        # Truth: empty sweeps of each cell while the person was in it.
        self._true_misses = np.zeros_like(scenario.pod_true, dtype=int)
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
        """Move the clock; the belief spreads (variances add, so one blur per call) or, with
        ``belief_motion="lpb"``, walks by the motion model."""
        if self.motion is not None and self.sc.motion is not None:
            m = self.sc.motion
            self.motion = advance(self.motion, m.terrain, m.params, minutes * 60.0)
            self.belief = self.motion.p
        else:
            sigma_cells = self.cfg.belief_diffusion_m * np.sqrt(minutes) / self.cfg.cell_m
            self.belief = diffuse(self.belief, sigma_cells)
        self.t += minutes

    def _true_cell(self, t: float) -> tuple[int, int] | None:
        """The person's cell at minute ``t``; None once they have left the map."""
        path = self.sc.path_m
        i = min(int(t), len(path) - 2)
        f = min(max(t - i, 0.0), 1.0)
        x, y = path[i] * (1 - f) + path[i + 1] * f
        if np.isnan(x):
            return None
        n = self.cfg.grid_cells
        return (
            min(int(y // self.cfg.cell_m), n - 1),
            min(int(x // self.cfg.cell_m), n - 1),
        )

    def _sector_of(self, cell: tuple[int, int]) -> int:
        k, s = self.cfg.sector_cells, self.cfg.n_sectors
        return (cell[0] // k) * s + cell[1] // k

    def pod_belief_now(self) -> np.ndarray:
        """What the belief expects the next sweep of each cell to see, after the sweeps of
        its sector so far (as if the person had stayed in the cell through them)."""
        k, s = self.cfg.sector_cells, self.cfg.n_sectors
        misses = self.swept.reshape(s, s).repeat(k, axis=0).repeat(k, axis=1)
        # In the scenario's memory order, so sums over it add up in the same order as
        # before (miss_correlation=0 keeps the old numbers to the last bit).
        out = np.empty_like(self.sc.pod_belief)
        out[...] = pod_after(*self._belief_split, misses)
        return out

    # --- the loop -----------------------------------------------------------------

    def observe(self) -> Observation:
        cfg = self.cfg
        pod_b = self.pod_belief_now()
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
        if cell is not None and self._sector_of(cell) == sector:
            visible, per_pass = (x[cell] for x in self._true_split)
            q = float(pod_after(visible, per_pass, self._true_misses[cell]))
            found_now = self.survive * q
            self.expected_minutes += found_now * t_mid
            self.survive -= found_now
            self._true_misses[cell] += 1
        self._advance(transit + cfg.sweep_min)
        # Belief: a sweep that found nothing (Bayes), with this sweep's detection chance.
        k, s = cfg.sector_cells, cfg.n_sectors
        r0, c0 = (sector // s) * k, (sector % s) * k
        block = (slice(r0, r0 + k), slice(c0, c0 + k))
        pod_block = self.pod_belief_now()[block]
        if self.motion is not None:
            pod = np.zeros_like(self.belief)
            pod[block] = pod_block
            self.motion = negative(self.motion, pod)
            self.belief = self.motion.p
        else:
            b = self.belief.copy()
            b[block] *= 1.0 - pod_block
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
