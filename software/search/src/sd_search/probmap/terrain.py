"""Terrain for the probability map: distances to tracks, water and buildings, and canopy.

``SimTerrainSource`` reads a GeoJSON layer of the kind the simulator writes for its worlds
and sites (``properties.kind``: ``path``, ``road``, ``track``, ``lake``, ``water``,
``building``, ``forest``), so the map and the simulator see the same terrain.
``NoTerrain`` is the fallback when there is no terrain data: every distance is infinite and
the canopy 0, so the map is the distance rings alone and the panel says so.
"""

from __future__ import annotations

import json
import math
from pathlib import Path
from typing import Any, Final, Literal, Protocol

import numpy as np
import shapely
from shapely.geometry import shape
from shapely.geometry.base import BaseGeometry

from sd_search.probmap.grid import Floats, Grid
from sd_search.probmap.rings import TerrainWeights

__all__ = [
    "FOREST_CANOPY",
    "NoTerrain",
    "SimTerrainSource",
    "TerrainKind",
    "TerrainSource",
    "terrain_weight",
]

TerrainKind = Literal["track", "road", "water", "building"]
FOREST_CANOPY: Final = 0.6  # the share of canopy over a forest cell, a starting value
TRACK_DECAY_M: Final = 50.0
WATER_NEAR_M: Final = 30.0
BUILDING_NEAR_M: Final = 20.0

_KINDS: Final[dict[str, str]] = {
    "path": "track",
    "track": "track",
    "footway": "track",
    "road": "road",
    "lake": "water",
    "water": "water",
    "river": "water",
    "building": "building",
    "forest": "forest",
    "wood": "forest",
}


class TerrainSource(Protocol):
    @property
    def available(self) -> bool: ...
    def distance_to(self, kind: TerrainKind, grid: Grid) -> Floats: ...
    def canopy(self, grid: Grid) -> Floats: ...


class NoTerrain:
    available = False

    def distance_to(self, kind: TerrainKind, grid: Grid) -> Floats:
        return np.full(grid.shape, math.inf)

    def canopy(self, grid: Grid) -> Floats:
        return np.zeros(grid.shape)


class SimTerrainSource:
    """Terrain from a GeoJSON FeatureCollection (WGS84 lon/lat)."""

    available = True

    def __init__(self, features: list[dict[str, Any]]) -> None:
        self._lonlat: dict[str, list[dict[str, Any]]] = {}
        for f in features:
            kind = _KINDS.get(str((f.get("properties") or {}).get("kind", "")))
            geom = f.get("geometry")
            if kind is None or not isinstance(geom, dict):
                continue
            self._lonlat.setdefault(kind, []).append(geom)

    @classmethod
    def from_file(cls, path: Path) -> SimTerrainSource:
        body = json.loads(path.read_text(encoding="utf-8"))
        if body.get("type") != "FeatureCollection":
            raise ValueError("terrängfilen ska vara en GeoJSON FeatureCollection")
        return cls(list(body.get("features") or []))

    def _plane(self, kind: str, grid: Grid) -> BaseGeometry | None:
        geoms = []
        for g in self._lonlat.get(kind, []):
            try:
                ll = shape(g)
            except (ValueError, TypeError, AttributeError, KeyError):
                continue
            geoms.append(
                shapely.transform(
                    ll,
                    lambda c: np.array([grid.xy(float(lat), float(lon)) for lon, lat in c]),
                )
            )
        if not geoms:
            return None
        return shapely.union_all(geoms)

    def distance_to(self, kind: TerrainKind, grid: Grid) -> Floats:
        geom = self._plane(kind, grid)
        if geom is None:
            return np.full(grid.shape, math.inf)
        cx, cy = grid.centers
        return np.asarray(shapely.distance(shapely.points(cx, cy), geom), dtype=np.float64)

    def canopy(self, grid: Grid) -> Floats:
        geom = self._plane("forest", grid)
        if geom is None:
            return np.zeros(grid.shape)
        cx, cy = grid.centers
        inside = shapely.contains_xy(geom, cx, cy)
        return np.where(inside, FOREST_CANOPY, 0.0)


def terrain_weight(terrain: TerrainSource, grid: Grid, w: TerrainWeights) -> Floats:
    """``(1 + a_track exp(-d_track / 50 m)) (1 + a_water [d_water < 30 m])
    (1 + a_build [d_building < 20 m])``; roads count as tracks."""
    d_track = np.minimum(terrain.distance_to("track", grid), terrain.distance_to("road", grid))
    d_water = terrain.distance_to("water", grid)
    d_build = terrain.distance_to("building", grid)
    return (
        (1 + w.track * np.exp(-d_track / TRACK_DECAY_M))
        * (1 + w.water * (d_water < WATER_NEAR_M))
        * (1 + w.building * (d_build < BUILDING_NEAR_M))
    )
