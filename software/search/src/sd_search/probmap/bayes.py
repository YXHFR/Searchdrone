"""The map's state, its starting values and its updates with Bayes' rule.

The map is always worked out as the prior plus an ordered log of passes, so the same
inputs give the same map, and a changed profile or operator area simply recomputes it.

Prior: the ring mass of each cell times its terrain, heading and operator-area weights,
scaled so that the cells and the rest of the world (``row``) sum to 1, with
``row = max(ROW_PRIOR, mass outside the grid)``.

A pass without a find, with ``D = sum_k p(k) POD(k)``::

    p'(c) = p(c) (1 - POD(c)) / (1 - D)
    row'  = row / (1 - D)

so searched cells go down, every other cell and ``row`` go up a little, and the sum stays
1. A confirmed find freezes the map: nothing changes it after that.
"""

from __future__ import annotations

import math
from collections.abc import Sequence
from dataclasses import dataclass, replace
from typing import Final, Protocol

import numpy as np
import shapely
from shapely.geometry import Polygon

from sd_schemas.probmap import PoaHint
from sd_schemas.search import SubjectCategory
from sd_search.probmap.grid import Floats, Grid
from sd_search.probmap.rings import Rings, ring_mass, rings_for
from sd_search.probmap.terrain import TerrainSource, terrain_weight

__all__ = [
    "HINT_FACTOR",
    "ROW_PRIOR",
    "HeadingPrior",
    "MapState",
    "OperatorArea",
    "ProbabilityMap",
    "RingTerrainMap",
    "heading_weight",
]

ROW_PRIOR: Final = 0.10
HINT_FACTOR: Final[dict[str, float]] = {"high": 3.0, "low": 0.3, "excluded": 0.0}
HEADING_KAPPA: Final = 1.0


@dataclass(frozen=True)
class OperatorArea:
    hint: PoaHint
    corners: tuple[tuple[float, float], ...]  # in the grid's plane


@dataclass(frozen=True)
class HeadingPrior:
    category: SubjectCategory
    heading_deg: float | None = None  # the direction walked, if a witness saw it


@dataclass(frozen=True)
class MapState:
    grid: Grid
    p: Floats  # (ny, nx)
    row: float
    pod_cum: Floats  # (ny, nx), the cumulative POD
    version: int = 0
    frozen: bool = False

    def total(self) -> float:
        return float(self.p.sum()) + self.row


class ProbabilityMap(Protocol):
    def prior(
        self,
        subject: HeadingPrior,
        terrain: TerrainSource,
        operator_areas: Sequence[OperatorArea],
        grid: Grid,
    ) -> MapState: ...
    def update_negative(self, state: MapState, pod: Floats) -> MapState: ...
    def freeze(self, state: MapState, reason: str) -> MapState: ...


def heading_weight(grid: Grid, heading_deg: float | None) -> Floats:
    """``0.5 + 0.5 exp(kappa (cos(theta - heading) - 1))``: 1 straight ahead, a little
    over 0.5 behind; 1 everywhere when no direction is known."""
    if heading_deg is None:
        return np.ones(grid.shape)
    cx, cy = grid.centers
    bearing = np.arctan2(cx, cy)  # clockwise from north
    delta = bearing - math.radians(heading_deg)
    w = 0.5 + 0.5 * np.exp(HEADING_KAPPA * (np.cos(delta) - 1))
    return np.where(np.hypot(cx, cy) < grid.cell / 2, 1.0, w)


def area_weight(grid: Grid, areas: Sequence[OperatorArea]) -> Floats:
    w = np.ones(grid.shape)
    cx, cy = grid.centers
    for a in areas:
        poly = Polygon(a.corners)
        if not poly.is_valid or poly.area <= 0:
            continue
        w = np.where(shapely.contains_xy(poly, cx, cy), w * HINT_FACTOR[a.hint], w)
    return w


class RingTerrainMap:
    """Distance rings plus terrain weights (the map of the first version)."""

    def prior(
        self,
        subject: HeadingPrior,
        terrain: TerrainSource,
        operator_areas: Sequence[OperatorArea],
        grid: Grid,
        rings: Rings | None = None,
    ) -> MapState:
        rings = rings_for(subject.category) if rings is None else rings
        mass = ring_mass(grid, rings)
        outside = max(0.0, 1.0 - float(mass.sum()))
        weighted = (
            mass
            * terrain_weight(terrain, grid, rings.weights)
            * heading_weight(grid, subject.heading_deg)
            * area_weight(grid, operator_areas)
        )
        total = float(weighted.sum())
        zeros = np.zeros(grid.shape)
        if total <= 0:  # every cell excluded: the person is somewhere else
            return MapState(grid=grid, p=zeros, row=1.0, pod_cum=zeros.copy())
        row = min(1.0, max(ROW_PRIOR, outside))
        return MapState(grid=grid, p=weighted / total * (1 - row), row=row, pod_cum=zeros)

    def update_negative(self, state: MapState, pod: Floats) -> MapState:
        """One pass without a find, with its POD per cell."""
        if state.frozen:
            return state
        pod = np.clip(pod, 0.0, 1.0)
        found = float((state.p * pod).sum())
        left = 1.0 - found
        if left <= 1e-12:  # cannot happen while row > 0; never divide by 0
            return state
        return replace(
            state,
            p=state.p * (1 - pod) / left,
            row=state.row / left,
            pod_cum=1 - (1 - state.pod_cum) * (1 - pod),
        )

    def freeze(self, state: MapState, reason: str) -> MapState:
        del reason  # recorded with the stored version
        return replace(state, frozen=True)
