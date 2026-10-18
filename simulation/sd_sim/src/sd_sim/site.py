"""Simulator sites: a real place with height data and map layers, turned into a world.

A site is a folder with ``site.toml`` (home, size, height source), ``dem.asc`` (heights on
a square grid around home) and ``osm.geojson`` (map layers). Public sites live in
``simulation/gazebo/sites/<name>/`` and are committed; a local site (the owner's own
area, from ``just world-fetch``) lives in ``build/sites/<name>/`` with a ``site.toml``
written from ``config/site.local.toml`` and is never committed. A local site with the
same name wins over a public one.

``Site.heights`` gives the ground's height above sea level anywhere in the site:
bicubic interpolation of the grid, minus the forest canopy when the source is a surface
model, lakes set below their water level, and a flat pad around home. The world origin
is home on the ground: world z = height - ``Site.home_alt``.
"""

from __future__ import annotations

import json
import math
import tomllib
from dataclasses import dataclass, field
from functools import cached_property
from pathlib import Path
from typing import Any

import numpy as np
import shapely
from numpy.typing import NDArray
from shapely.geometry import MultiPolygon, Polygon, shape

from sd_safety.targets import Plane
from sd_sim import paths, raster

PUBLIC_DIR = paths.SIMULATION / "gazebo" / "sites"
# The pad around home is flat out to PAD_FLAT_M and blends into the terrain by PAD_BLEND_M.
PAD_FLAT_M = 12.0
PAD_BLEND_M = 35.0
# Forest canopy is removed with a soft edge this wide (the source grid is 30 m).
CANOPY_EDGE_M = 30.0
# The lake bottom under the water surface.
LAKE_DEPTH_M = 1.0
FOREST_KINDS = ("forest",)
# The world's mesh step; home's canopy weight is worked out on the same grid.
HOME_STEP_M = 5.0
WATER_KINDS = ("water",)

Xy = tuple[float, float]
Grid = NDArray[np.float64]


class SiteError(ValueError):
    """A site folder is missing or does not have the expected shape."""


@dataclass(frozen=True)
class Layer:
    """A map feature in the site's plane (metres east/north of home)."""

    kind: str
    osm: str
    geom: Any  # shapely geometry
    props: dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class DemGrid:
    """Heights (m above sea level) on nodes every ``cell_m``; row 0 is the northernmost."""

    cell_m: float
    z: Grid

    @property
    def half_m(self) -> float:
        return (self.z.shape[0] - 1) / 2 * self.cell_m

    def sample(self, east: NDArray[np.float64], north: NDArray[np.float64]) -> Grid:
        """Bicubic (Catmull-Rom) interpolation; points outside clamp to the edge."""
        n = self.z.shape[0]
        col = (np.asarray(east, dtype=np.float64) + self.half_m) / self.cell_m
        row = (self.half_m - np.asarray(north, dtype=np.float64)) / self.cell_m
        col = np.clip(col, 0.0, n - 1.0)
        row = np.clip(row, 0.0, n - 1.0)
        c0 = np.floor(col).astype(np.int64)
        r0 = np.floor(row).astype(np.int64)
        tc, tr = col - c0, row - r0

        def weights(t: Grid) -> list[Grid]:
            t2, t3 = t * t, t * t * t
            return [
                (-t3 + 2 * t2 - t) / 2,
                (3 * t3 - 5 * t2 + 2) / 2,
                (-3 * t3 + 4 * t2 + t) / 2,
                (t3 - t2) / 2,
            ]

        wc, wr = weights(tc), weights(tr)
        out = np.zeros(np.broadcast(col, row).shape, dtype=np.float64)
        for i in range(4):
            rr = np.clip(r0 + i - 1, 0, n - 1)
            for j in range(4):
                cc = np.clip(c0 + j - 1, 0, n - 1)
                out += wr[i] * wc[j] * self.z[rr, cc]
        return out


@dataclass(frozen=True)
class Site:
    name: str
    world: str
    title: str
    folder: Path
    public: bool
    lat: float
    lon: float
    heading_deg: float
    half_size_m: float
    dem: DemGrid
    surface: str  # "dsm" (canopy in the heights) | "dtm" (bare ground)
    canopy_m: float
    layers: tuple[Layer, ...]
    raw_toml: str  # site.toml as read, part of the generator's input hash

    @property
    def plane(self) -> Plane:
        return Plane(self.lat, self.lon)

    @property
    def ortho(self) -> Path | None:
        path = self.folder / "ortho.npy"
        return path if path.is_file() else None

    def of_kind(self, *kinds: str) -> list[Layer]:
        return [layer for layer in self.layers if layer.kind in kinds]

    def union(self, *kinds: str) -> Any:
        geoms = [layer.geom for layer in self.of_kind(*kinds)]
        return shapely.union_all(geoms) if geoms else Polygon()

    @cached_property
    def water_levels(self) -> list[tuple[Any, float]]:
        """Each water body with its level: the median raw height inside it."""
        out = []
        for layer in self.of_kind(*WATER_KINDS):
            pts = _sample_points(layer.geom, 5.0)
            if len(pts) == 0:
                continue
            z = self.dem.sample(pts[:, 0], pts[:, 1])
            out.append((layer.geom, float(np.median(z))))
        return out

    @cached_property
    def raw_home_alt(self) -> float:
        return float(self.dem.sample(np.array([0.0]), np.array([0.0]))[0])

    def heights(self, east: Grid, north: Grid) -> Grid:
        """Ground height above sea level on a regular grid of points (2-D arrays)."""
        z = self.dem.sample(east, north) - self.canopy_m * self.canopy_weight(east, north)
        for geom, level in self.water_levels:
            inside = raster.coverage(geom, east, north) > 0.5
            z = np.where(inside, np.minimum(z, level - LAKE_DEPTH_M), z)
        r = np.hypot(east, north)
        blend = np.clip((r - PAD_FLAT_M) / (PAD_BLEND_M - PAD_FLAT_M), 0.0, 1.0)
        blend = blend * blend * (3 - 2 * blend)  # smoothstep
        return self.home_alt + (z - self.home_alt) * blend

    def canopy_weight(self, east: Grid, north: Grid) -> Grid:
        """How much of the canopy to take off (0-1), soft over CANOPY_EDGE_M at the forest's
        edge; zero for bare-ground sources."""
        if self.surface != "dsm" or self.canopy_m <= 0:
            return np.zeros_like(east)
        forest = raster.coverage(self.union(*FOREST_KINDS), east, north)
        return raster.soften(forest, CANOPY_EDGE_M / _step(east, north))

    @cached_property
    def home_alt(self) -> float:
        """The pad's height above sea level: the height at home with the canopy taken off
        exactly as the terrain around it has it (same grid step as the world's mesh)."""
        xs = np.arange(-60.0, 60.0 + HOME_STEP_M / 2, HOME_STEP_M)
        east, north = np.meshgrid(xs, xs[::-1])
        weight = self.canopy_weight(east, north)
        centre = len(xs) // 2
        return round(self.raw_home_alt - self.canopy_m * float(weight[centre, centre]), 2)

    @property
    def home_arg(self) -> str:
        """``lat,lon,alt,heading`` for SITL's ``--home``."""
        return f"{self.lat:.6f},{self.lon:.6f},{self.home_alt:.1f},{self.heading_deg:g}"

    def input_digest(self) -> str:
        import hashlib

        h = hashlib.sha256()
        h.update(self.raw_toml.encode())
        h.update(self.dem.z.tobytes())
        h.update((self.folder / "osm.geojson").read_bytes())
        if self.ortho is not None:
            h.update(self.ortho.read_bytes())
        return h.hexdigest()


def _step(east: Grid, north: Grid) -> float:
    del north
    return float(abs(east[0, 1] - east[0, 0])) if east.ndim == 2 and east.shape[1] > 1 else 1.0


def _sample_points(geom: Any, step: float) -> NDArray[np.float64]:
    minx, miny, maxx, maxy = geom.bounds
    xs = np.arange(minx + step / 2, maxx, step)
    ys = np.arange(miny + step / 2, maxy, step)
    if len(xs) == 0 or len(ys) == 0:
        return np.zeros((0, 2))
    gx, gy = np.meshgrid(xs, ys)
    inside = shapely.contains_xy(geom, gx, gy)
    return np.stack([gx[inside], gy[inside]], axis=1)


def read_dem(path: Path) -> DemGrid:
    lines = [
        line for line in path.read_text(encoding="ascii").splitlines() if not line.startswith("#")
    ]
    header: dict[str, float] = {}
    for line in lines[:2]:
        key, value = line.split()
        header[key] = float(value)
    if set(header) != {"cell_m", "nodes"}:
        raise SiteError(f"{path}: väntade raderna cell_m och nodes")
    n = int(header["nodes"])
    z = np.array([[float(v) for v in line.split()] for line in lines[2:]], dtype=np.float64)
    if z.shape != (n, n):
        raise SiteError(f"{path}: rutnätet är {z.shape}, inte {n} x {n}")
    return DemGrid(header["cell_m"], z)


def _to_plane(geom: Any, plane: Plane) -> Any:
    def xy(coords: NDArray[np.float64]) -> NDArray[np.float64]:
        return np.array([plane.xy(lat, lon) for lon, lat in coords], dtype=np.float64).reshape(
            -1, 2
        )

    return shapely.transform(geom, xy)


def read_layers(path: Path, plane: Plane) -> tuple[Layer, ...]:
    data = json.loads(path.read_text(encoding="utf-8"))
    out = []
    for feature in data.get("features", []):
        props = dict(feature.get("properties", {}))
        kind = str(props.pop("kind"))
        osm = str(props.pop("osm", ""))
        geom = _to_plane(shape(feature["geometry"]), plane)
        if isinstance(geom, Polygon | MultiPolygon):
            geom = shapely.make_valid(geom)
        out.append(Layer(kind, osm, geom, props))
    return tuple(out)


def _num(table: dict[str, Any], key: str, where: str) -> float:
    value = table.get(key)
    if isinstance(value, bool) or not isinstance(value, int | float) or not math.isfinite(value):
        raise SiteError(f"{where}: {key} ska vara ett tal")
    return float(value)


def folder_for(name: str, repo: Path = paths.REPO) -> tuple[Path, bool]:
    local = repo / "build" / "sites" / name
    if (local / "site.toml").is_file():
        return local, False
    public = repo / "simulation" / "gazebo" / "sites" / name
    if (public / "site.toml").is_file():
        return public, True
    raise SiteError(f"platsen {name} finns inte (kör just world-fetch site={name})")


def load(name: str, repo: Path = paths.REPO) -> Site:
    folder, public = folder_for(name, repo)
    raw = (folder / "site.toml").read_text(encoding="utf-8")
    data = tomllib.loads(raw)
    where = str(folder / "site.toml")
    home, area, dem = data.get("home", {}), data.get("area", {}), data.get("dem", {})
    surface = str(dem.get("surface", "dsm"))
    if surface not in ("dsm", "dtm"):
        raise SiteError(f"{where}: dem.surface ska vara dsm eller dtm")
    lat, lon = _num(home, "lat", where), _num(home, "lon", where)
    plane = Plane(lat, lon)
    for needed in ("dem.asc", "osm.geojson"):
        if not (folder / needed).is_file():
            raise SiteError(f"{folder / needed} saknas (kör just world-fetch site={name})")
    return Site(
        name=str(data.get("name", name)),
        world=str(data.get("world", f"sd_{name}")),
        title=str(data.get("title", name)),
        folder=folder,
        public=public,
        lat=lat,
        lon=lon,
        heading_deg=_num(home, "heading_deg", where),
        half_size_m=_num(area, "half_size_m", where),
        dem=read_dem(folder / "dem.asc"),
        surface=surface,
        canopy_m=float(dem.get("canopy_m", 0.0)),
        layers=read_layers(folder / "osm.geojson", plane),
        raw_toml=raw,
    )


def world_sites(repo: Path = paths.REPO) -> dict[str, str]:
    """World name -> site name, for every public and local site."""
    out: dict[str, str] = {}
    for root in (repo / "simulation" / "gazebo" / "sites", repo / "build" / "sites"):
        if not root.is_dir():
            continue
        for toml in sorted(root.glob("*/site.toml")):
            data = tomllib.loads(toml.read_text(encoding="utf-8"))
            name = str(data.get("name", toml.parent.name))
            out[str(data.get("world", f"sd_{name}"))] = name
    return out
