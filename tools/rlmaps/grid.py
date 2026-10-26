"""The numpy half of ``just rl-maps``: a place's 2 x 2 km grid, its land cover and tracks.

The other half (``build.py``) reads ESA WorldCover and OpenStreetMap and needs rasterio
and osmium; everything here needs only numpy, so the workspace's tests check it.

The grid is the simulator's (``sd_searchsim.config.SimConfig``): 80 x 80 cells of 25 m,
row 0 at the south edge, column 0 at the west edge. A place is put on it with a local
flat projection around its centre, which is good to well under a metre over 2 km.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np

CELL_M = 25.0
GRID_CELLS = 80
SIZE_M = CELL_M * GRID_CELLS
SUB = 5  # land cover samples per cell side (every 5 m; WorldCover has 10 m pixels)
M_PER_DEG_LAT = 110_574.0
M_PER_DEG_LON_EQ = 111_320.0

# ESA WorldCover 2021 v200 classes.
TREE, SHRUB, GRASS, CROP, BUILT, BARE, SNOW, WATER, WETLAND, MANGROVE, MOSS = (
    10, 20, 30, 40, 50, 60, 70, 80, 90, 95, 100,
)  # fmt: skip
# Share of the ground a camera straight down cannot see, per class. Model defaults, not
# measured: a Swedish forest hides most of the ground, a bush some, a building a little.
CANOPY = {TREE: 0.75, SHRUB: 0.35, BUILT: 0.3, MANGROVE: 0.6, WETLAND: 0.05}
NATURAL = (TREE, SHRUB, GRASS, BARE, WETLAND, MOSS)

# A place is used when it is mostly forest and open land, not a lake or a town, and a
# road or track passes near the middle (a last seen point is nearly always on one).
MIN_NATURAL = 0.5
MAX_WATER = 0.3
MAX_BUILT = 0.15


@dataclass(frozen=True)
class Place:
    lat: float
    lon: float

    def m_per_deg_lon(self) -> float:
        return M_PER_DEG_LON_EQ * math.cos(math.radians(self.lat))

    def bounds(self, margin_m: float = 0.0) -> tuple[float, float, float, float]:
        """west, south, east, north in degrees, for the grid plus a margin."""
        half = SIZE_M / 2 + margin_m
        dlat, dlon = half / M_PER_DEG_LAT, half / self.m_per_deg_lon()
        return self.lon - dlon, self.lat - dlat, self.lon + dlon, self.lat + dlat

    def to_grid_m(self, lon: np.ndarray, lat: np.ndarray) -> np.ndarray:
        """Degrees to the grid's metres (x east, y north, origin at the south-west corner)."""
        x = (np.asarray(lon) - self.lon) * self.m_per_deg_lon() + SIZE_M / 2
        y = (np.asarray(lat) - self.lat) * M_PER_DEG_LAT + SIZE_M / 2
        return np.stack([x, y], axis=-1)

    def sample_points(self) -> tuple[np.ndarray, np.ndarray]:
        """lon, lat of the land cover samples, shape (GRID_CELLS * SUB,) * 2 as a mesh."""
        step = CELL_M / SUB
        c = (np.arange(GRID_CELLS * SUB) + 0.5) * step - SIZE_M / 2
        lat = self.lat + c / M_PER_DEG_LAT
        lon = self.lon + c / self.m_per_deg_lon()
        lat_m, lon_m = np.meshgrid(lat, lon, indexing="ij")  # row = north
        return lon_m, lat_m


def class_shares(classes: np.ndarray) -> dict[int, np.ndarray]:
    """Per cell, the share of samples in each class; ``classes`` is (80*SUB, 80*SUB)."""
    n = GRID_CELLS
    blocks = classes.reshape(n, SUB, n, SUB)
    return {int(k): (blocks == k).mean(axis=(1, 3)) for k in np.unique(classes)}


def canopy(shares: dict[int, np.ndarray]) -> np.ndarray:
    out = np.zeros((GRID_CELLS, GRID_CELLS))
    for k, hide in CANOPY.items():
        if k in shares:
            out += hide * shares[k]
    return np.clip(out, 0.0, 0.9)


def share(shares: dict[int, np.ndarray], *classes: int) -> np.ndarray:
    out = np.zeros((GRID_CELLS, GRID_CELLS))
    for k in classes:
        out += shares.get(k, 0.0)
    return out


def rasterise(lines_m: list[np.ndarray]) -> np.ndarray:
    """Cells a polyline passes through (points every metre), shape (80, 80) bool."""
    mask = np.zeros((GRID_CELLS, GRID_CELLS), dtype=bool)
    for line in lines_m:
        if len(line) == 1:
            pts = line
        else:
            seg = np.diff(line, axis=0)
            steps = np.maximum(np.ceil(np.hypot(seg[:, 0], seg[:, 1])), 1).astype(int)
            pts = np.concatenate(
                [
                    a + np.linspace(0.0, 1.0, k, endpoint=False)[:, None] * d
                    for a, d, k in zip(line[:-1], seg, steps, strict=True)
                ]
                + [line[-1:]]
            )
        col = np.floor(pts[:, 0] / CELL_M).astype(int)
        row = np.floor(pts[:, 1] / CELL_M).astype(int)
        inside = (col >= 0) & (col < GRID_CELLS) & (row >= 0) & (row < GRID_CELLS)
        mask[row[inside], col[inside]] = True
    return mask


def acceptable(shares: dict[int, np.ndarray], track: np.ndarray) -> str | None:
    """Why the place is not used, or None when it is."""
    if share(shares, *NATURAL).mean() < MIN_NATURAL:
        return "för lite skog och mark"
    if share(shares, WATER).mean() > MAX_WATER:
        return "för mycket vatten"
    if share(shares, BUILT).mean() > MAX_BUILT:
        return "för mycket bebyggelse"
    lo, hi = int(0.3 * GRID_CELLS), int(0.7 * GRID_CELLS)
    if not track[lo:hi, lo:hi].any():
        return "ingen väg eller stig nära mitten"
    return None


def site_arrays(shares: dict[int, np.ndarray], track: np.ndarray) -> dict[str, np.ndarray]:
    """What the simulator reads (``sd_searchsim.realworld``)."""
    return {
        "canopy": canopy(shares).astype(np.float32),
        "track": track,
        "water": share(shares, WATER) > 0.5,
        "built": share(shares, BUILT).astype(np.float32),
    }
