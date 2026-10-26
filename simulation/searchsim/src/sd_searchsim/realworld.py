"""Real places and real lost-person statistics (``world=real``).

The places are 2 x 2 km of Swedish forest and open land made by ``just rl-maps``: land
cover from ESA WorldCover 2021 (canopy, water) and every road and track from
OpenStreetMap, one ``.npz`` per place (``tools/rlmaps/grid.py`` writes them). This module
only reads them, with numpy.

Who is missing and where they are found follows the Norwegian search dog statistics
(Lie & Skår, ICAR 2016; about 400 searches with a find, 2009-2015), the nearest data to
Sweden: per category the distance from the last seen point (quartiles, fitted with a
log-normal) and the share found *on* a road or track and within 100 m of one. The person
stays at the find location: the statistics are of where people were found, not of how
they walked. Nobody is found in open water. Children are left out: their numbers are
only in Koester's book (*Lost Person Behavior*), which the project does not have.

The belief uses the category's numbers as published; the truth's distance is off by
about 1.35x either way (real places do not follow the tables exactly; Doke 2012 found
that ISRID's did not fit Yosemite). The places are split for good: 70 % train, 10 %
validation, 20 % the final comparison, so the comparison is on places the policy never
saw. Each scenario also turns or mirrors its place (8 ways) at random.
"""

from __future__ import annotations

import functools
import json
import math
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from sd_searchsim.config import EVAL_SEED_BASE, VAL_SEED_BASE, SimConfig
from sd_searchsim.scenario import Scenario, _distance_to, _sample_cell, cell_centres, no_fly

_Z75 = 0.6745  # the standard normal's 75th percentile


@dataclass(frozen=True)
class Category:
    name: str
    share: float  # how common the category is among searches (a default)
    q25_km: float
    q50_km: float
    q75_km: float
    on_track: float  # found on a road, path or track
    within_100m: float  # found within 100 m of one

    @property
    def sigma(self) -> float:
        """Log-normal spread fitted to the quartiles."""
        return math.log(self.q75_km / self.q25_km) / (2 * _Z75)

    @property
    def offset_scale_m(self) -> float:
        """Exponential scale of the distance to the track for those found off it."""
        q = (self.within_100m - self.on_track) / (1.0 - self.on_track)
        return -100.0 / math.log(1.0 - q)


# Lie & Skår (2016), as read in research report 19. Shares: about a quarter of Swedish
# missing person reports are people with dementia (Swedish police, 2020), the rest are
# round defaults.
CATEGORIES = (
    Category("dementia", 0.35, 0.27, 0.80, 3.15, 0.64, 0.94),
    Category("despondent", 0.30, 0.37, 0.98, 2.50, 0.42, 0.86),
    Category("hiker", 0.20, 0.64, 1.71, 3.35, 0.33, 0.69),
    Category("berry_picker", 0.15, 1.26, 1.69, 2.78, 0.15, 0.46),
)
_CATEGORY_P = np.array([c.share for c in CATEGORIES])


@dataclass(frozen=True)
class Site:
    name: str
    canopy: np.ndarray  # (n, n)
    track: np.ndarray  # (n, n) bool, a road or track passes through the cell
    water: np.ndarray  # (n, n) bool, open water


@functools.cache
def load_sites(sites_dir: str, grid_cells: int) -> tuple[Site, ...]:
    root = Path(sites_dir)
    index = root / "sites.json"
    if not index.exists():
        raise FileNotFoundError(f"inga platser i {root} (kör just rl-maps)")
    doc = json.loads(index.read_text(encoding="utf-8"))
    sites = []
    for entry in doc["sites"]:
        with np.load(root / f"{entry['name']}.npz") as d:
            site = Site(
                entry["name"], d["canopy"].astype(float), d["track"].astype(bool), d["water"]
            )
        if site.canopy.shape != (grid_cells, grid_cells):
            raise ValueError(f"plats {site.name} har fel storlek {site.canopy.shape}")
        sites.append(site)
    return tuple(sites)


def split_of(seed: int) -> str:
    if seed >= EVAL_SEED_BASE:
        return "eval"
    return "val" if seed >= VAL_SEED_BASE else "train"


def in_split(sites: tuple[Site, ...], split: str) -> tuple[Site, ...]:
    """Every tenth place in turn: 0-6 train, 7 validation, 8-9 the final comparison."""
    keep = {"train": range(7), "val": (7,), "eval": (8, 9)}[split]
    out = tuple(s for i, s in enumerate(sites) if i % 10 in keep)
    if not out:
        raise ValueError(f"för få platser för {split} (minst 10 behövs)")
    return out


def category_map(
    cfg: SimConfig,
    ipp_m: np.ndarray,
    cat: Category,
    median_m: float,
    track_dist_m: np.ndarray,
    water: np.ndarray,
) -> np.ndarray:
    """Where the category is found: distance rings times the distance to a track."""
    xx, yy = cell_centres(cfg)
    r = np.maximum(np.hypot(xx - ipp_m[0], yy - ipp_m[1]), cfg.cell_m / 2)
    s = cat.sigma
    radial = np.exp(-((np.log(r / median_m)) ** 2) / (2 * s * s)) / (r * s)
    density = radial / (2 * math.pi * r)
    land = ~water
    on = (track_dist_m <= cfg.cell_m / 2) & land
    off = ~on & land
    a = density * on
    b = density * np.exp(-track_dist_m / cat.offset_scale_m) * off
    p = np.zeros_like(density)
    if a.sum() > 0:
        p += cat.on_track * a / a.sum()
    if b.sum() > 0:
        p += (1.0 - cat.on_track) * b / b.sum()
    return p / p.sum()


def _turn(x: np.ndarray, k: int) -> np.ndarray:
    x = np.rot90(x, k % 4)
    return np.fliplr(x) if k >= 4 else x


def make_real_scenario(seed: int, cfg: SimConfig) -> Scenario:
    rng = np.random.default_rng(seed)
    n = cfg.grid_cells
    size = n * cfg.cell_m
    sites = in_split(load_sites(cfg.sites_dir, n), split_of(seed))
    site = sites[int(rng.integers(len(sites)))]
    k = int(rng.integers(8))
    canopy = _turn(site.canopy, k).copy()
    track = _turn(site.track, k)
    water = _turn(site.water, k)
    track_dist = _distance_to(track, cfg.cell_m)

    cat = CATEGORIES[int(rng.choice(len(CATEGORIES), p=_CATEGORY_P))]
    # The last seen point: on a road or track near the middle (a car park, a house, a
    # path), else any dry cell there.
    lo, hi = int(0.3 * n), int(0.7 * n)
    middle = np.zeros((n, n), dtype=bool)
    middle[lo:hi, lo:hi] = True
    options = np.argwhere(middle & track & ~water)
    if len(options) == 0:
        options = np.argwhere(middle & ~water)
    r, c = options[int(rng.integers(len(options)))]
    ipp = np.array([(c + rng.random()) * cfg.cell_m, (r + rng.random()) * cfg.cell_m])
    # Home stays clear of the no-fly band (as in the made-up world).
    home = np.clip(ipp + rng.normal(0, 150, 2), 300.0, size - 300.0)

    median_m = cat.q50_km * 1000.0
    prior = category_map(cfg, ipp, cat, median_m, track_dist, water)
    truth = category_map(
        cfg, ipp, cat, median_m * float(np.exp(rng.normal(0, 0.3))), track_dist, water
    )
    start = _sample_cell(rng, truth, cfg)
    path = np.repeat(start[None, :], int(cfg.horizon_min) + 1, axis=0)

    p_det_true = float(rng.uniform(0.55, 0.9))
    return Scenario(
        seed=seed,
        canopy=canopy,
        track_dist_m=track_dist,
        prior=prior,
        allowed=no_fly(rng, cfg.n_sectors),
        ipp_m=ipp,
        home_m=home,
        path_m=path,
        pod_true=p_det_true * (1.0 - canopy),
        pod_belief=cfg.belief_p_det * (1.0 - canopy),
        strategy=cat.name,
    )
