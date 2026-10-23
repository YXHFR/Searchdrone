"""Distance rings: how far from the place last seen people of a category were found.

``rings.toml`` holds the 25/50/75/95 % distances (km, straight line) and the terrain
weights per category. The distribution function ``F(r)`` is piecewise linear through
``(0, 0)``, the four points and ``(2 * r95, 1)``; the mass of a thin ring is spread evenly
over its area. A cell's mass is the integral over the cell, taken on ``SUB x SUB`` points,
each the mean density of a ring as wide as a sub-cell (finite at the IPP itself).
"""

from __future__ import annotations

import math
import tomllib
from dataclasses import dataclass
from functools import cache
from importlib import resources
from typing import Any, Final

import numpy as np

from sd_schemas.search import SubjectCategory
from sd_search.probmap.grid import SUB, Floats, Grid

__all__ = ["CHILD_CATEGORIES", "Rings", "TerrainWeights", "ring_mass", "rings_for"]

CHILD_CATEGORIES: Final = frozenset({"child_1_6", "child_7_12", "child_13_15"})


@dataclass(frozen=True)
class TerrainWeights:
    track: float  # a_track: near a road, path or track
    water: float  # a_water: within 30 m of water
    building: float  # a_build: within 20 m of a building


@dataclass(frozen=True)
class Rings:
    category: str
    radii_m: tuple[float, float, float, float]  # 25, 50, 75, 95 %
    uncertain: bool
    source: str
    weights: TerrainWeights

    @property
    def knots(self) -> tuple[tuple[float, ...], tuple[float, ...]]:
        r = (0.0, *self.radii_m, 2 * self.radii_m[3])
        return r, (0.0, 0.25, 0.50, 0.75, 0.95, 1.0)

    def cdf(self, r: Floats) -> Floats:
        xs, ys = self.knots
        return np.interp(r, xs, ys)

    def mean_density(self, r: Floats, half: float) -> Floats:
        """Mass per m² averaged over the ring ``[r - half, r + half]``."""
        lo = np.maximum(r - half, 0.0)
        hi = r + half
        area = math.pi * (hi * hi - lo * lo)
        return (self.cdf(hi) - self.cdf(lo)) / area


@cache
def _table() -> dict[str, Rings]:
    raw = tomllib.loads(
        resources.files("sd_search.probmap").joinpath("rings.toml").read_text(encoding="utf-8")
    )

    def one(name: str, row: dict[str, Any]) -> Rings:
        km = [float(x) for x in row["rings_km"]]
        if len(km) != 4 or km != sorted(km) or km[0] <= 0:
            raise ValueError(f"rings.toml: {name} needs four rising distances")
        w = row["weights"]
        if not isinstance(w, dict):
            raise ValueError(f"rings.toml: {name} needs weights")
        return Rings(
            category=name,
            radii_m=(km[0] * 1000, km[1] * 1000, km[2] * 1000, km[3] * 1000),
            uncertain=bool(row["uncertain"]),
            source=str(row["source"]),
            weights=TerrainWeights(
                track=float(w["track"]), water=float(w["water"]), building=float(w["building"])
            ),
        )

    out = {"default": one("default", raw["default"])}
    for name, row in raw["category"].items():
        out[name] = one(name, row)
    return out


def rings_for(category: SubjectCategory) -> Rings:
    """The category's rings, or the general default (children, autism, other)."""
    table = _table()
    return table.get(category, table["default"])


def ring_mass(grid: Grid, rings: Rings) -> Floats:
    """Each cell's share of the whole distribution (the rest lies outside the grid)."""
    sx, sy = grid.subpoints
    r = np.hypot(sx, sy)
    half = grid.cell / SUB / 2
    density = rings.mean_density(r, half)
    return density.sum(axis=-1) * (grid.cell / SUB) ** 2
