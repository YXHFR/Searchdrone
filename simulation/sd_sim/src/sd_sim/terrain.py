"""Build a site's Gazebo model: textured terrain, forest, buildings, cars and lakes.

``build(site)`` writes the model ``build/sim/models/sd_site_<name>/`` (``model.sdf``,
``model.config``, ``meshes/``; that folder is on Gazebo's resource path) and
``build/sim/sites/<name>/`` with ``ground.npy`` (the terrain heights the world uses to
put objects on the ground) and ``manifest.json``. Everything is generated here from the site's
height grid and map layers and, when they have been fetched into ``build/``, its trees and
the tree pictures; nothing is downloaded here:

- Terrain: a mesh every ``MESH_STEP_M`` in ``TILES`` x ``TILES`` tiles, each with its own
  texture painted from the map layers (forest floor, meadow, lake, asphalt, gravel, road
  markings) with deterministic noise, about 0.15 m per pixel. With a fetched orthophoto
  (``just ortho-fetch``; ``Site.ortho``) the open ground shows the photo instead, and the
  painted forest floor stays under every tree's crown (``photo_cover``): the photo shows
  the trees from above, and the world's own trees stand there.
- Forest: the site's real trees where they have been fetched (``just trees-fetch``: every
  tree where Skogsstyrelsen's tree height raster has it, as tall as it says, its species
  from Naturvårdsverket's land cover map), as textured models when the tree pictures have
  been fetched too (``just tree-pictures-fetch``; ``sd_sim.forest``). Without them, spruce,
  pine and birch in real sizes on a jittered grid in the map's forest, low-poly in flat
  colours. Merged into one mesh per tile. Trees have no collision (see the README); the
  terrain, buildings and pad do.
- Buildings from the map's footprints, extruded to their tagged height (or a typical one
  for their type) with a gabled roof on rectangular houses.
- Cars (4.5 x 1.8 x 1.5 m) parked in rows in the map's parking areas.
- Lakes: a flat water surface at the lake's level.

The same site always gives the same files, byte for byte (``manifest.json`` lists their
SHA-256). A finished build is reused while the site's inputs and ``GENERATOR_VERSION``
are unchanged.
"""

from __future__ import annotations

import hashlib
import json
import math
import shutil
from collections.abc import Callable, Iterable, Sequence
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, cast

import numpy as np
import shapely
from numpy.typing import NDArray
from shapely.geometry import LineString, MultiLineString, Polygon, box
from shapely.ops import substring

from sd_sim import forest as forest_mod
from sd_sim import paths, png, raster
from sd_sim.site import Site

GENERATOR_VERSION = 6
# Gazebo needs a square heightmap of 2^n + 1 pixels: 257 gives 4.7 m over 1.2 km.
HEIGHTMAP_PX = 257
HEIGHTMAP = "ground_collision.png"
MESH_STEP_M = 5.0
TILES = 4
TEXTURE_PX = 2048
# Trees per square metre of forest (200 per hectare: a managed Swedish forest has
# 500-1500, but the canopy painted on the ground under them closes the gaps from above).
TREE_DENSITY = 0.03
TREE_KEEP_OUT_M = 2.5  # from roads, buildings and water
# Real trees are left out only where a tree cannot stand: on a road or a path (a crown
# over it), on a roof (the raster sees buildings too), in water or a car park, on the pad.
REAL_TREE_ROAD_M = 0.5
REAL_TREE_ROOF_M = 1.0
PARKING_STALL_M = (2.6, 5.2)
PARKING_OCCUPANCY = 0.5
CAR_SIZE_M = (4.5, 1.8, 1.5)
# Written into every ground texture draped with the photo, so tools/check_no_orthophotos.py
# finds the texture even when it is copied without the world's manifest.
PHOTO_PNG_TEXT = {"Source": "Searchdrone ground_texture=orthophoto"}
# Under a tree the ground keeps its painted floor out to this much more than the crown
# (the photo's crowns lean away from where it was taken), blended over PHOTO_EDGE_M.
PHOTO_CROWN_SCALE = 1.15
PHOTO_CROWN_MARGIN_M = 0.5
PHOTO_EDGE_M = 0.6

Grid = NDArray[np.floating[Any]]
Rgb = tuple[int, int, int]


# --- deterministic noise ------------------------------------------------------------


def _hash01(ix: NDArray[np.int64], it: NDArray[np.int64], seed: int) -> Grid:
    """A value in [0, 1) for every integer lattice point, the same everywhere for a seed."""
    h = (ix.astype(np.uint64) * np.uint64(0x9E3779B1)) ^ (
        it.astype(np.uint64) * np.uint64(0x85EBCA77)
    )
    h ^= np.uint64(seed * 0xC2B2AE3D & 0xFFFFFFFF)
    h ^= h >> np.uint64(15)
    h *= np.uint64(0x2C1B3C6D)
    h ^= h >> np.uint64(12)
    h *= np.uint64(0x297A2D39)
    h ^= h >> np.uint64(15)
    return (h & np.uint64(0xFFFFFF)).astype(np.float64) / float(0x1000000)


def value_noise(x: Grid, y: Grid, scale_m: float, seed: int) -> Grid:
    """Smooth noise in [0, 1) with features about ``scale_m`` across, in world metres."""
    gx, gy = x / scale_m, y / scale_m
    ix, it = np.floor(gx).astype(np.int64), np.floor(gy).astype(np.int64)
    fx, fy = gx - ix, gy - it
    fx, fy = fx * fx * (3 - 2 * fx), fy * fy * (3 - 2 * fy)
    a = _hash01(ix, it, seed)
    b = _hash01(ix + 1, it, seed)
    c = _hash01(ix, it + 1, seed)
    d = _hash01(ix + 1, it + 1, seed)
    return (a * (1 - fx) + b * fx) * (1 - fy) + (c * (1 - fx) + d * fx) * fy


def fbm(x: Grid, y: Grid, scale_m: float, seed: int, octaves: int = 4) -> Grid:
    total, amp, norm = np.zeros_like(x), 1.0, 0.0
    for i in range(octaves):
        total += amp * value_noise(x, y, scale_m / 2**i, seed + 101 * i)
        norm += amp
        amp *= 0.5
    return total / norm


def rand01(key: str) -> float:
    """A stable number in [0, 1) for a name (a building, a parking lot)."""
    return int.from_bytes(hashlib.sha256(key.encode()).digest()[:8], "big") / 2**64


# --- meshes -------------------------------------------------------------------------


@dataclass
class Mesh:
    """Triangles grouped by material; written as OBJ with per-face normals or smooth ones."""

    vertices: list[NDArray[np.float64]] = field(default_factory=list)
    groups: dict[str, list[NDArray[np.int64]]] = field(default_factory=dict)
    count: int = 0

    def add(self, material: str, verts: Grid, faces: NDArray[np.int64]) -> None:
        self.vertices.append(np.asarray(verts, dtype=np.float64).reshape(-1, 3))
        self.groups.setdefault(material, []).append(np.asarray(faces, dtype=np.int64) + self.count)
        self.count += len(self.vertices[-1])

    @property
    def empty(self) -> bool:
        return self.count == 0

    def triangles(self) -> int:
        return sum(len(f) for faces in self.groups.values() for f in faces)

    def obj(self, mtl_name: str) -> str:
        verts = np.concatenate(self.vertices) if self.vertices else np.zeros((0, 3))
        lines = [f"mtllib {mtl_name}"]
        lines += [f"v {x:.3f} {y:.3f} {z:.3f}" for x, y, z in verts]
        normal_lines: list[str] = []
        face_lines: list[str] = []
        n_index = 0
        for material in sorted(self.groups):
            faces = np.concatenate(self.groups[material])
            face_lines.append(f"usemtl {material}")
            a, b, c = verts[faces[:, 0]], verts[faces[:, 1]], verts[faces[:, 2]]
            normals = np.cross(b - a, c - a)
            length = np.linalg.norm(normals, axis=1, keepdims=True)
            normals = np.where(
                length > 0, normals / np.where(length > 0, length, 1), [0.0, 0.0, 1.0]
            )
            for (i, j, k), (nx, ny, nz) in zip(faces, normals, strict=True):
                n_index += 1
                normal_lines.append(f"vn {nx:.4f} {ny:.4f} {nz:.4f}")
                face_lines.append(f"f {i + 1}//{n_index} {j + 1}//{n_index} {k + 1}//{n_index}")
        return "\n".join(lines + normal_lines + face_lines) + "\n"


@dataclass(frozen=True)
class Material:
    name: str
    rgb: tuple[float, float, float]
    specular: float = 0.05
    texture: str = ""

    def mtl(self) -> str:
        r, g, b = self.rgb
        # Ambient at 60 % of the diffuse colour keeps shaded sides darker than lit ones.
        ka = (0.6 * r, 0.6 * g, 0.6 * b)
        out = [
            f"newmtl {self.name}",
            "Ka {:.3f} {:.3f} {:.3f}".format(*ka),
            f"Kd {r:.3f} {g:.3f} {b:.3f}",
        ]
        out.append(f"Ks {self.specular:.3f} {self.specular:.3f} {self.specular:.3f}")
        out.append("Ns 10")
        if self.texture:
            out.append(f"map_Kd {self.texture}")
        return "\n".join(out) + "\n"


def write_obj(folder: Path, stem: str, mesh: Mesh, materials: Iterable[Material]) -> None:
    used = set(mesh.groups)
    mtl = "".join(m.mtl() + "\n" for m in sorted(materials, key=lambda m: m.name) if m.name in used)
    (folder / f"{stem}.mtl").write_text(mtl, encoding="utf-8")
    (folder / f"{stem}.obj").write_text(mesh.obj(f"{stem}.mtl"), encoding="utf-8")


def _rgb(c: Rgb) -> tuple[float, float, float]:
    """An sRGB colour as the linear value the renderer expects in a material (textures are
    sRGB pictures and are converted by the renderer itself)."""

    def linear(v: int) -> float:
        x = v / 255
        return x / 12.92 if x <= 0.04045 else ((x + 0.055) / 1.055) ** 2.4

    return (linear(c[0]), linear(c[1]), linear(c[2]))


# --- the terrain -----------------------------------------------------------------------


@dataclass(frozen=True)
class Ground:
    """Terrain heights above home on nodes every ``step`` m, row 0 northernmost."""

    step: float
    half: float
    z: Grid

    def at(self, east: Grid, north: Grid) -> Grid:
        """Bilinear height (the terrain mesh is linear between nodes)."""
        n = self.z.shape[0]
        col = np.clip((np.asarray(east) + self.half) / self.step, 0, n - 1 - 1e-9)
        row = np.clip((self.half - np.asarray(north)) / self.step, 0, n - 1 - 1e-9)
        c0, r0 = np.floor(col).astype(np.int64), np.floor(row).astype(np.int64)
        tc, tr = col - c0, row - r0
        z = self.z
        top = z[r0, c0] * (1 - tc) + z[r0, c0 + 1] * tc
        bottom = z[r0 + 1, c0] * (1 - tc) + z[r0 + 1, c0 + 1] * tc
        return top * (1 - tr) + bottom * tr


def ground_grid(site: Site) -> Ground:
    half = site.half_size_m
    xs = np.arange(-half, half + MESH_STEP_M / 2, MESH_STEP_M)
    east, north = np.meshgrid(xs, xs[::-1])
    z = np.round(site.heights(east, north) - site.home_alt, 3)
    return Ground(MESH_STEP_M, half, z)


def terrain_tiles(ground: Ground) -> list[tuple[int, int, Mesh]]:
    n = ground.z.shape[0]
    per = (n - 1) // TILES
    xs = np.arange(n) * ground.step - ground.half
    out = []
    for ti in range(TILES):  # north to south
        for tj in range(TILES):  # west to east
            r0, c0 = ti * per, tj * per
            rows, cols = np.arange(r0, r0 + per + 1), np.arange(c0, c0 + per + 1)
            ee, nn = np.meshgrid(xs[cols], -xs[rows])
            zz = ground.z[np.ix_(rows, cols)]
            verts = np.stack([ee, nn, zz], axis=-1).reshape(-1, 3)
            m = per + 1
            idx = np.arange(m * m).reshape(m, m)
            a, b = idx[:-1, :-1].ravel(), idx[:-1, 1:].ravel()
            c, d = idx[1:, :-1].ravel(), idx[1:, 1:].ravel()
            # Rows go south as the index grows, so (a, c, b) is counter-clockwise from above.
            faces = np.concatenate([np.stack([a, c, b], 1), np.stack([b, c, d], 1)])
            mesh = Mesh()
            mesh.add(f"terrain_{ti}_{tj}", verts, faces)
            out.append((ti, tj, mesh))
    return out


def terrain_obj(mesh: Mesh, ground: Ground, ti: int, tj: int) -> str:
    """Terrain tiles need texture coordinates and smooth normals, so they are written here."""
    verts = mesh.vertices[0]
    size = 2 * ground.half / TILES
    west = -ground.half + tj * size
    south = ground.half - (ti + 1) * size
    # Smooth normals from the whole grid, so neighbouring tiles shade alike at the seam.
    grad: Any = np.gradient(ground.z, ground.step)
    gz_n, gz_e = grad
    n = ground.z.shape[0]
    col = np.round((verts[:, 0] + ground.half) / ground.step).astype(np.int64)
    row = np.round((ground.half - verts[:, 1]) / ground.step).astype(np.int64)
    dzde, dzdn = gz_e[row, col], -gz_n[row, col]
    normals = np.stack([-dzde, -dzdn, np.ones_like(dzde)], 1)
    normals /= np.linalg.norm(normals, axis=1, keepdims=True)
    del n
    lines = [f"mtllib terrain_{ti}_{tj}.mtl", f"usemtl terrain_{ti}_{tj}"]
    lines += [f"v {x:.3f} {y:.3f} {z:.3f}" for x, y, z in verts]
    # Texture coordinates stay half a pixel inside the tile, so the renderer never
    # samples the opposite edge of the picture at a seam.
    edge = 0.5 / TEXTURE_PX
    u = np.clip((verts[:, 0] - west) / size, edge, 1 - edge)
    v = np.clip((verts[:, 1] - south) / size, edge, 1 - edge)
    lines += [f"vt {a:.6f} {b:.6f}" for a, b in zip(u, v, strict=True)]
    lines += [f"vn {a:.4f} {b:.4f} {c:.4f}" for a, b, c in normals]
    faces = mesh.groups[f"terrain_{ti}_{tj}"][0]
    lines += [
        f"f {i + 1}/{i + 1}/{i + 1} {j + 1}/{j + 1}/{j + 1} {k + 1}/{k + 1}/{k + 1}"
        for i, j, k in faces
    ]
    return "\n".join(lines) + "\n"


# --- textures -------------------------------------------------------------------------

# Surface colours (sRGB) and the noise that gives each its texture.
SURFACE: dict[str, Rgb] = {
    "ground": (96, 112, 62),
    "meadow": (146, 152, 84),
    "farmland": (138, 120, 82),
    "grass": (92, 132, 58),
    "residential": (104, 116, 74),
    "pitch": (72, 128, 64),
    "scrub": (78, 96, 52),
    "forest": (46, 58, 32),
    "wetland": (98, 102, 62),
    "rock": (124, 120, 110),
    "sand": (196, 182, 140),
    "water": (34, 52, 66),
    "parking": (88, 88, 90),
    "asphalt": (72, 72, 76),
    "gravel": (152, 140, 114),
    "building": (90, 90, 90),
}
ASPHALT_ROADS = {
    "motorway",
    "trunk",
    "primary",
    "secondary",
    "tertiary",
    "unclassified",
    "residential",
    "service",
    "living_street",
    "cycleway",
    "pedestrian",
}
MARKED_ROADS = {"motorway", "trunk", "primary", "secondary", "tertiary"}
# Painting order: later kinds cover earlier ones.
PAINT_ORDER = (
    "farmland",
    "meadow",
    "grass",
    "residential",
    "scrub",
    "forest",
    "wetland",
    "rock",
    "sand",
    "pitch",
    "parking",
    "water",
)


@dataclass(frozen=True)
class Paint:
    """The geometry each surface is painted from, in the site's plane."""

    areas: dict[str, Any]
    asphalt: Any
    gravel: Any
    markings: Any
    buildings: Any
    crowns: list[tuple[float, float, float]]  # (east, north, radius) of every tree


def road_geometry(site: Site) -> tuple[Any, Any, Any]:
    asphalt, gravel, marks = [], [], []
    for layer in site.of_kind("road"):
        if layer.props.get("layer") == "tunnel":
            continue
        width = float(layer.props.get("width_m", 3.0))
        geom = layer.geom
        band = geom.buffer(width / 2, cap_style="flat", join_style="round")
        (asphalt if layer.props.get("highway") in ASPHALT_ROADS else gravel).append(band)
        if layer.props.get("highway") in MARKED_ROADS:
            lines = list(geom.geoms) if isinstance(geom, MultiLineString) else [geom]
            for line in lines:
                for side in (-1, 1):  # edge lines 0.3 m in from the edge, 0.1 m wide
                    edge = line.offset_curve(side * (width / 2 - 0.35))
                    if not edge.is_empty:
                        marks.append(edge.buffer(0.05, cap_style="flat"))
                pos = 0.0
                while pos < line.length:  # centre line: 3 m dashes, 9 m gaps
                    dash = substring(line, pos, min(pos + 3.0, line.length))
                    if isinstance(dash, LineString) and dash.length > 0:
                        marks.append(dash.buffer(0.05, cap_style="flat"))
                    pos += 12.0
    return (
        shapely.union_all(asphalt) if asphalt else Polygon(),
        shapely.union_all(gravel) if gravel else Polygon(),
        shapely.union_all(marks) if marks else Polygon(),
    )


def _clip(geom: Any, bounds: tuple[float, float, float, float]) -> Any:
    return geom.intersection(box(*bounds)) if not geom.is_empty else geom


def paint_tile(site: Site, paint: Paint, ti: int, tj: int, px: int, seed: int) -> NDArray[np.uint8]:
    size = 2 * site.half_size_m / TILES
    west = -site.half_size_m + tj * size
    north = site.half_size_m - ti * size
    step = size / px
    xs = west + (np.arange(px) + 0.5) * step
    ys = north - (np.arange(px) + 0.5) * step
    ex, ny = np.meshgrid(xs, ys)
    bounds = (west - 1, north - size - 1, west + size + 1, north + 1)
    fr = (west, north, step, px, px)

    def mask(geom: Any) -> NDArray[np.bool_]:
        return raster.mask(_clip(geom, bounds), *fr)

    kind = np.zeros(ex.shape, dtype=np.int16)  # index into names
    names = ["ground", *PAINT_ORDER, "gravel", "asphalt", "building"]
    for i, name in enumerate(names):
        if name == "ground":
            continue
        geom = {"gravel": paint.gravel, "asphalt": paint.asphalt, "building": paint.buildings}.get(
            name, paint.areas.get(name, Polygon())
        )
        if not geom.is_empty:
            kind[mask(geom)] = i
    base = np.zeros((*ex.shape, 3), dtype=np.float64)
    for i, name in enumerate(names):
        base[kind == i] = SURFACE[name]

    # Texture: large patches, small grain, and kind-specific detail.
    patches = fbm(ex, ny, 40.0, seed, 3) - 0.5
    grain = value_noise(ex, ny, 0.6, seed + 7) - 0.5
    fine = value_noise(ex, ny, 0.15, seed + 9) - 0.5
    shade = 1.0 + 0.28 * patches + 0.18 * grain + 0.10 * fine
    is_kind = {name: kind == i for i, name in enumerate(names)}
    # Meadow and farmland: dry yellow streaks; forest floor: moss and needles.
    dry = value_noise(ex * 0.3, ny, 3.0, seed + 11)
    for name in ("meadow", "farmland", "ground"):
        sel = is_kind[name]
        base[sel] = (
            base[sel] * (1 - 0.35 * dry[sel, None])
            + np.array([182, 168, 96]) * 0.35 * dry[sel, None]
        )
    moss = value_noise(ex, ny, 2.5, seed + 13)
    sel = is_kind["forest"]
    base[sel] = (
        base[sel] * (1 - 0.4 * moss[sel, None]) + np.array([70, 92, 40]) * 0.4 * moss[sel, None]
    )
    pools = fbm(ex, ny, 12.0, seed + 17, 2) > 0.62
    sel = is_kind["wetland"] & pools
    base[sel] = (48, 58, 48)
    sel = is_kind["water"] | is_kind["asphalt"] | is_kind["parking"]
    shade[sel] = 1.0 + 0.04 * patches[sel] + 0.10 * fine[sel]
    rgb = base * shade[..., None]

    # Tree crowns from above: a dark round shadow with a lit rim under every tree.
    crowns = [
        c
        for c in paint.crowns
        if bounds[0] - 6 < c[0] < bounds[2] + 6 and bounds[1] - 6 < c[1] < bounds[3] + 6
    ]
    if crowns:
        darken = np.ones(ex.shape)
        for cx, cy, r in crowns:
            c0 = max(int((cx - r - west) / step), 0)
            c1 = min(int((cx + r - west) / step) + 1, px)
            r0 = max(int((north - cy - r) / step), 0)
            r1 = min(int((north - cy + r) / step) + 1, px)
            if c1 <= c0 or r1 <= r0:
                continue
            dx = ex[r0:r1, c0:c1] - cx
            dy = ny[r0:r1, c0:c1] - cy
            d = np.hypot(dx, dy) / r
            f = np.clip(1.0 - d, 0.0, 1.0)
            darken[r0:r1, c0:c1] = np.minimum(darken[r0:r1, c0:c1], 1.0 - 0.55 * np.sqrt(f))
        rgb *= darken[..., None]

    marks = mask(paint.markings)
    rgb[marks] = (224, 224, 214)
    if site.ortho is not None:
        m = max((photo_reach(c[2]) for c in paint.crowns), default=0.0)
        near = [
            c
            for c in paint.crowns
            if bounds[0] - m < c[0] < bounds[2] + m and bounds[1] - m < c[1] < bounds[3] + m
        ]
        cover = photo_cover(near, ex, ny, west, north, step)[..., None]
        rgb = ortho_tile(site, west, north, size, px) * (1.0 - cover) + rgb * cover
    return np.clip(np.round(rgb), 0, 255).astype(np.uint8)


def photo_reach(crown_radius_m: float) -> float:
    """How far from a crown's centre the ground keeps its painted floor."""
    return crown_radius_m * PHOTO_CROWN_SCALE + PHOTO_CROWN_MARGIN_M


def photo_cover(
    crowns: Sequence[tuple[float, float, float]],
    ex: Grid,
    ny: Grid,
    west: float,
    north: float,
    step: float,
) -> Grid:
    """Where the painted floor stays instead of the photo: 1 under a tree's crown (scaled
    by PHOTO_CROWN_SCALE plus PHOTO_CROWN_MARGIN_M), 0 on open ground, soft in between."""
    cover = np.zeros(ex.shape)
    rows, cols = ex.shape
    for cx, cy, r in crowns:
        reach = photo_reach(r)
        c0 = max(int((cx - reach - west) / step), 0)
        c1 = min(int((cx + reach - west) / step) + 1, cols)
        r0 = max(int((north - cy - reach) / step), 0)
        r1 = min(int((north - cy + reach) / step) + 1, rows)
        if c1 <= c0 or r1 <= r0:
            continue
        d = np.hypot(ex[r0:r1, c0:c1] - cx, ny[r0:r1, c0:c1] - cy)
        f = np.clip((reach - d) / PHOTO_EDGE_M, 0.0, 1.0)
        cover[r0:r1, c0:c1] = np.maximum(cover[r0:r1, c0:c1], f)
    return cover


def ortho_tile(site: Site, west: float, north: float, size: float, px: int) -> Grid:
    """The tile cut from the site's orthophoto (``ortho.npy``: n x n x 3 bytes, row 0
    north, covering the site edge to edge; 0.16 m per pixel from ``just ortho-fetch``)."""
    if site.ortho is None:
        raise ValueError(f"{site.name} har inget ortofoto")
    photo = np.load(site.ortho, mmap_mode="r")
    n = photo.shape[0]
    if photo.dtype != np.uint8 or photo.ndim != 3 or photo.shape[1:] != (n, 3):
        raise ValueError(
            f"{site.ortho}: väntade n x n x 3 byte, fick {photo.shape} {photo.dtype}; "
            "hämta igen med just ortho-fetch"
        )
    cell = 2 * site.half_size_m / n
    xs = west + (np.arange(px) + 0.5) * size / px
    ys = north - (np.arange(px) + 0.5) * size / px
    cols = np.clip(((xs + site.half_size_m) / cell).astype(np.int64), 0, n - 1)
    rows = np.clip(((site.half_size_m - ys) / cell).astype(np.int64), 0, n - 1)
    return photo[np.ix_(rows, cols)].astype(np.float64)


# --- trees -------------------------------------------------------------------------------


@dataclass(frozen=True)
class Tree:
    east: float
    north: float
    z: float
    species: str  # spruce | pine | birch
    height: float
    yaw: float

    @property
    def crown_radius(self) -> float:
        return {"spruce": 0.19, "pine": 0.2, "birch": 0.21}[self.species] * self.height


SPECIES = (("spruce", 0.5, (14.0, 26.0)), ("pine", 0.3, (14.0, 24.0)), ("birch", 0.2, (10.0, 20.0)))


def pick_species(p: float) -> tuple[str, float, float]:
    """Species and height range for a number in [0, 1), by the species' shares."""
    acc = 0.0
    for species, share, (lo, hi) in SPECIES:
        acc += share
        if p < acc:
            return species, lo, hi
    species, _, (lo, hi) = SPECIES[-1]
    return species, lo, hi


def plant_trees(site: Site, ground: Ground, blocked: Any) -> list[Tree]:
    """A jittered grid of trees inside the forest, away from roads, buildings and water."""
    forest = site.union("forest")
    if forest.is_empty:
        return []
    spacing = 1.0 / math.sqrt(TREE_DENSITY)
    half = site.half_size_m - 2.0
    xs = np.arange(-half, half, spacing)
    ex, ny = np.meshgrid(xs, xs[::-1])
    ix = np.round(ex / spacing).astype(np.int64)
    it = np.round(ny / spacing).astype(np.int64)
    seed = int(rand01(site.name) * 1e6)
    ex = ex + (_hash01(ix, it, seed + 1) - 0.5) * spacing * 0.9
    ny = ny + (_hash01(ix, it, seed + 2) - 0.5) * spacing * 0.9
    allowed = forest.difference(blocked) if not blocked.is_empty else forest
    inside = shapely.contains_xy(allowed, ex, ny)
    pick = _hash01(ix, it, seed + 3)
    size = _hash01(ix, it, seed + 4)
    yaw = _hash01(ix, it, seed + 5) * 2 * math.pi
    ez, nz = ex[inside], ny[inside]
    z = ground.at(ez, nz)
    trees = []
    for e, n, zz, p, s, a in zip(ez, nz, z, pick[inside], size[inside], yaw[inside], strict=True):
        species, lo, hi = pick_species(float(p))
        trees.append(
            Tree(
                round(float(e), 2),
                round(float(n), 2),
                round(float(zz), 2),
                species,
                round(lo + (hi - lo) * float(s), 2),
                round(float(a), 4),
            )
        )
    return trees


def _cone(
    base_r: float, z0: float, z1: float, sides: int, yaw: float
) -> tuple[Grid, NDArray[np.int64]]:
    ang = yaw + np.arange(sides) * 2 * math.pi / sides
    ring = np.stack([base_r * np.cos(ang), base_r * np.sin(ang), np.full(sides, z0)], 1)
    verts = np.vstack([ring, [[0.0, 0.0, z1]]])
    faces = np.array([[i, (i + 1) % sides, sides] for i in range(sides)])
    return verts, faces


def _prism(
    r: float, z0: float, z1: float, sides: int, yaw: float
) -> tuple[Grid, NDArray[np.int64]]:
    ang = yaw + np.arange(sides) * 2 * math.pi / sides
    lo = np.stack([r * np.cos(ang), r * np.sin(ang), np.full(sides, z0)], 1)
    hi = lo + np.array([0.0, 0.0, z1 - z0])
    faces = []
    for i in range(sides):
        j = (i + 1) % sides
        faces += [[i, j, sides + j], [i, sides + j, sides + i]]
    return np.vstack([lo, hi]), np.array(faces)


def _blob(rx: float, rz: float, zc: float, yaw: float) -> tuple[Grid, NDArray[np.int64]]:
    """A low-poly crown: an elongated octahedron with an extra waist ring (16 faces)."""
    ang = yaw + np.arange(6) * math.pi / 3
    ring = np.stack([rx * np.cos(ang), rx * np.sin(ang), np.full(6, zc)], 1)
    verts = np.vstack([ring, [[0, 0, zc + rz]], [[0, 0, zc - rz * 0.6]]])
    faces = []
    for i in range(6):
        j = (i + 1) % 6
        faces += [[i, j, 6], [j, i, 7]]
    return verts, np.array(faces)


def tree_meshes(trees: Sequence[Tree], half: float) -> dict[tuple[int, int], Mesh]:
    size = 2 * half / TILES
    out: dict[tuple[int, int], Mesh] = {}
    for t in trees:
        ti = min(int((half - t.north) / size), TILES - 1)
        tj = min(int((t.east + half) / size), TILES - 1)
        mesh = out.setdefault((ti, tj), Mesh())
        h, shade = t.height, int(rand01(f"{t.east},{t.north}") * 3)
        parts: list[tuple[str, tuple[Grid, NDArray[np.int64]]]] = []
        if t.species == "spruce":
            parts.append(("trunk", _prism(0.02 * h, -0.3, 0.25 * h, 4, t.yaw)))
            parts.append((f"spruce_{shade}", _cone(0.19 * h, 0.08 * h, h, 8, t.yaw)))
        elif t.species == "pine":
            parts.append(("pine_trunk", _prism(0.022 * h, -0.3, 0.7 * h, 4, t.yaw)))
            parts.append((f"pine_{shade}", _blob(0.2 * h, 0.2 * h, 0.8 * h, t.yaw)))
        else:
            parts.append(("birch_trunk", _prism(0.02 * h, -0.3, 0.55 * h, 4, t.yaw)))
            parts.append((f"birch_{shade}", _blob(0.21 * h, 0.32 * h, 0.64 * h, t.yaw)))
        for material, (verts, faces) in parts:
            mesh.add(material, verts + np.array([t.east, t.north, t.z]), faces)
    return out


TREE_MATERIALS = [
    Material("trunk", _rgb((58, 44, 32))),
    Material("pine_trunk", _rgb((92, 62, 42))),
    Material("birch_trunk", _rgb((196, 192, 182))),
    *[
        Material(f"spruce_{i}", _rgb(c))
        for i, c in enumerate([(20, 40, 24), (26, 48, 28), (16, 34, 22)])
    ],
    *[
        Material(f"pine_{i}", _rgb(c))
        for i, c in enumerate([(34, 54, 26), (42, 62, 30), (30, 48, 24)])
    ],
    *[
        Material(f"birch_{i}", _rgb(c))
        for i, c in enumerate([(58, 92, 36), (68, 100, 40), (52, 84, 34)])
    ],
]


# --- buildings ---------------------------------------------------------------------------

TYPICAL_HEIGHT_M = {
    "house": 5.0,
    "detached": 5.0,
    "shed": 2.6,
    "garage": 2.8,
    "hut": 2.6,
    "cabin": 3.5,
    "toilets": 3.0,
    "hotel": 7.0,
    "commercial": 6.5,
    "farm_auxiliary": 4.5,
    "service": 3.0,
    "yes": 4.5,
}
WALLS = [
    ("wall_red", (128, 34, 28)),
    ("wall_yellow", (214, 186, 112)),
    ("wall_white", (222, 218, 206)),
    ("wall_grey", (150, 146, 138)),
]
ROOFS = [("roof_black", (46, 46, 50)), ("roof_red", (122, 52, 40)), ("roof_grey", (98, 100, 104))]
BUILDING_MATERIALS = [Material(n, _rgb(c)) for n, c in WALLS + ROOFS]


@dataclass(frozen=True)
class Building:
    osm: str
    footprint: Polygon
    base_z: float
    eave_m: float  # wall height above base
    ridge_m: float  # extra roof height (0 = flat)
    wall: str
    roof: str


def plan_buildings(site: Site, ground: Ground) -> list[Building]:
    out = []
    for layer in site.of_kind("building"):
        geoms = list(layer.geom.geoms) if hasattr(layer.geom, "geoms") else [layer.geom]
        for k, poly in enumerate(g for g in geoms if isinstance(g, Polygon) and g.area > 4):
            poly = shapely.orient_polygons(poly.simplify(0.05))
            poly = cast(Polygon, poly)
            coords = np.asarray(poly.exterior.coords)
            z = ground.at(coords[:, 0], coords[:, 1])
            kind = str(layer.props.get("building", "yes"))
            height = layer.props.get("height_m")
            levels = layer.props.get("levels")
            if height is None:
                height = 3.0 * float(levels) + 1.0 if levels else TYPICAL_HEIGHT_M.get(kind, 4.5)
            key = f"{layer.osm}:{k}"
            rect = poly.minimum_rotated_rectangle
            gabled = (
                layer.props.get("roof", "gabled") in ("gabled", "hipped")
                and len(coords) == 5
                and poly.area > 0.9 * rect.area
                and poly.area < 600
            )
            short = min(_rect_sides(cast(Polygon, rect)))
            ridge = math.tan(math.radians(27)) * short / 2 if gabled else 0.0
            eave = float(height) - ridge if gabled else float(height)
            out.append(
                Building(
                    key,
                    poly,
                    round(float(z.min()), 3),
                    round(max(eave, 2.2), 3),
                    round(ridge, 3),
                    WALLS[int(rand01(key + "w") * len(WALLS))][0],
                    ROOFS[int(rand01(key + "r") * len(ROOFS))][0],
                )
            )
    return out


def _rect_sides(rect: Polygon) -> tuple[float, float]:
    c = np.asarray(rect.exterior.coords)
    return (float(np.hypot(*(c[1] - c[0])[:2])), float(np.hypot(*(c[2] - c[1])[:2])))


def building_mesh(buildings: Sequence[Building]) -> Mesh:
    mesh = Mesh()
    for b in buildings:
        ring = np.asarray(b.footprint.exterior.coords)[:-1]  # counter-clockwise
        n = len(ring)
        top = b.base_z + b.eave_m
        low = np.column_stack([ring, np.full(n, b.base_z - 0.5)])
        high = np.column_stack([ring, np.full(n, top)])
        faces = []
        for i in range(n):
            j = (i + 1) % n
            faces += [[i, j, n + j], [i, n + j, n + i]]
        mesh.add(b.wall, np.vstack([low, high]), np.array(faces))
        if b.ridge_m > 0:
            mesh_roof_gabled(mesh, b, top)
        else:
            tris = shapely.constrained_delaunay_triangles(b.footprint)
            for tri in tris.geoms:
                c = np.asarray(cast(Polygon, shapely.orient_polygons(tri)).exterior.coords)[:3]
                mesh.add(b.roof, np.column_stack([c, np.full(3, top)]), np.array([[0, 1, 2]]))
    return mesh


def mesh_roof_gabled(mesh: Mesh, b: Building, top: float) -> None:
    c = np.asarray(cast(Polygon, b.footprint.minimum_rotated_rectangle).exterior.coords)[:4]
    a, bb = _rect_sides(Polygon(c))
    if a < bb:  # make c[0]-c[1] a long side
        c = np.roll(c, -1, axis=0)
    p0, p1, p2, p3 = c
    m03, m12 = (p0 + p3) / 2, (p1 + p2) / 2  # ridge ends over the short sides
    ridge = top + b.ridge_m
    v = np.array([[*p0, top], [*p1, top], [*p2, top], [*p3, top], [*m03, ridge], [*m12, ridge]])
    faces = np.array([[0, 1, 5], [0, 5, 4], [2, 3, 4], [2, 4, 5]])
    # Orient every face upwards.
    fixed = []
    for f in faces:
        n = np.cross(v[f[1]] - v[f[0]], v[f[2]] - v[f[0]])
        fixed.append(f if n[2] >= 0 else f[::-1])
    mesh.add(b.roof, v, np.array(fixed))
    gables = np.array([[3, 0, 4], [1, 2, 5]])
    mesh.add(b.wall, v, gables)
    mesh.add(b.wall, v, gables[:, ::-1])  # both sides: winding of a gable is not known


# --- cars -----------------------------------------------------------------------------

CAR_COLOURS = [
    ("car_white", (226, 226, 222)),
    ("car_black", (28, 28, 30)),
    ("car_silver", (168, 170, 174)),
    ("car_blue", (36, 64, 128)),
    ("car_red", (150, 28, 26)),
    ("car_grey", (92, 94, 98)),
]
CAR_MATERIALS = [Material(n, _rgb(c), 0.4) for n, c in CAR_COLOURS] + [
    Material("car_glass", _rgb((32, 40, 48)), 0.6)
]


@dataclass(frozen=True)
class Car:
    east: float
    north: float
    z: float
    yaw: float
    colour: str


def park_cars(site: Site, ground: Ground) -> list[Car]:
    cars = []
    sw, sl = PARKING_STALL_M
    for layer in site.of_kind("parking"):
        geoms = list(layer.geom.geoms) if hasattr(layer.geom, "geoms") else [layer.geom]
        for poly in (g for g in geoms if isinstance(g, Polygon)):
            rect = np.asarray(cast(Polygon, poly.minimum_rotated_rectangle).exterior.coords)[:4]
            a, b = rect[1] - rect[0], rect[3] - rect[0]
            if np.linalg.norm(a) < np.linalg.norm(b):
                a, b = b, a
            ua, ub = a / np.linalg.norm(a), b / np.linalg.norm(b)
            yaw = math.atan2(ub[1], ub[0])  # cars stand across the long side
            inner = poly.buffer(-0.5)
            for i in range(int(np.linalg.norm(a) // sw)):
                for j in range(int(np.linalg.norm(b) // (sl + 1.0))):
                    centre = rect[0] + ua * (i + 0.5) * sw + ub * (j * (sl + 1.0) + sl / 2)
                    key = f"{layer.osm}:{i}:{j}"
                    stall = shapely.Point(centre).buffer(1.4)
                    if not inner.contains(stall) or rand01(key) > PARKING_OCCUPANCY:
                        continue
                    colour = CAR_COLOURS[int(rand01(key + "c") * len(CAR_COLOURS))][0]
                    z = float(ground.at(np.array([centre[0]]), np.array([centre[1]]))[0])
                    cars.append(
                        Car(
                            round(float(centre[0]), 2),
                            round(float(centre[1]), 2),
                            round(z, 3),
                            round(yaw, 4),
                            colour,
                        )
                    )
    return cars


def _box(
    cx: float, cy: float, z0: float, lx: float, ly: float, lz: float
) -> tuple[Grid, NDArray[np.int64]]:
    x0, x1, y0, y1, z1 = cx - lx / 2, cx + lx / 2, cy - ly / 2, cy + ly / 2, z0 + lz
    v = np.array(
        [
            [x0, y0, z0],
            [x1, y0, z0],
            [x1, y1, z0],
            [x0, y1, z0],
            [x0, y0, z1],
            [x1, y0, z1],
            [x1, y1, z1],
            [x0, y1, z1],
        ]
    )
    f = np.array(
        [
            [4, 5, 6],
            [4, 6, 7],
            [0, 1, 5],
            [0, 5, 4],
            [1, 2, 6],
            [1, 6, 5],
            [2, 3, 7],
            [2, 7, 6],
            [3, 0, 4],
            [3, 4, 7],
        ]
    )
    return v, f


def car_mesh(cars: Sequence[Car]) -> Mesh:
    mesh = Mesh()
    length, width, height = CAR_SIZE_M
    for car in cars:
        rot = np.array(
            [[math.cos(car.yaw), -math.sin(car.yaw)], [math.sin(car.yaw), math.cos(car.yaw)]]
        )
        for material, (v, f) in (
            (car.colour, _box(0.0, 0.0, 0.18, length, width, 0.72)),
            ("car_glass", _box(-0.2, 0.0, 0.9, 2.4, width - 0.12, height - 0.9)),
        ):
            v = v.copy()
            v[:, :2] = v[:, :2] @ rot.T + [car.east, car.north]
            v[:, 2] += car.z
            mesh.add(material, v, f)
    return mesh


# --- water ----------------------------------------------------------------------------


def water_mesh(site: Site) -> Mesh:
    mesh = Mesh()
    for geom, level in site.water_levels:
        z = round(level - site.home_alt, 3)
        clipped = geom.intersection(
            box(-site.half_size_m, -site.half_size_m, site.half_size_m, site.half_size_m)
        )
        if clipped.is_empty:
            continue
        for tri in shapely.constrained_delaunay_triangles(clipped).geoms:
            c = np.asarray(shapely.orient_polygons(tri).exterior.coords)[:3]
            mesh.add("water", np.column_stack([c, np.full(3, z)]), np.array([[0, 1, 2]]))
    return mesh


WATER_MATERIAL = Material("water", _rgb((40, 62, 80)), 0.6)


# --- the model -----------------------------------------------------------------------------


def model_name(site: Site) -> str:
    return f"sd_site_{site.name}"


def out_dir(site: Site, repo: Path = paths.REPO) -> Path:
    return repo / "build" / "sim" / "sites" / site.name


def _sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def build(
    site: Site,
    repo: Path = paths.REPO,
    *,
    force: bool = False,
    progress: Callable[[str], None] = lambda _: None,
) -> Path:
    """Write the site's model folder (``model://sd_site_<name>``) and return it."""
    folder = out_dir(site, repo)
    model = repo / "build" / "sim" / "models" / model_name(site)
    real = forest_mod.load_trees(site)
    atlas = forest_mod.load_atlas(repo) if real is not None else None
    digest = site.input_digest()
    if atlas is not None:
        h = hashlib.sha256(digest.encode())
        for name in ("atlas.json", *forest_mod.ATLAS_FILES):
            h.update((forest_mod.atlas_folder(repo) / name).read_bytes())
        digest = h.hexdigest()
    manifest_path = folder / "manifest.json"
    if not force and manifest_path.is_file():
        old = json.loads(manifest_path.read_text(encoding="utf-8"))
        if (
            old.get("version") == GENERATOR_VERSION
            and old.get("inputs") == digest
            and all(
                (repo / name).is_file() and _sha(repo / name) == sha
                for name, sha in old["files"].items()
            )
        ):
            return model
    for stale in (folder, model):
        if stale.exists():
            shutil.rmtree(stale)
    folder.mkdir(parents=True)
    meshes = model / "meshes"
    meshes.mkdir(parents=True)
    seed = int(rand01(site.name) * 1e6)

    progress("höjder")
    ground = ground_grid(site)
    np.save(folder / "ground.npy", ground.z)
    progress("byggnader, bilar och vatten")
    buildings = plan_buildings(site, ground)
    cars = park_cars(site, ground)
    asphalt, gravel, markings = road_geometry(site)
    footprints = shapely.union_all([b.footprint for b in buildings]) if buildings else Polygon()
    blocked = shapely.union_all(
        [
            asphalt.buffer(TREE_KEEP_OUT_M),
            gravel.buffer(1.0),
            footprints.buffer(TREE_KEEP_OUT_M),
            site.union("water", "parking", "pitch").buffer(1.0),
            shapely.Point(0, 0).buffer(40.0),
        ]
    )
    progress("skog")
    if real is None:
        trees = plant_trees(site, ground, blocked)
        crowns = [(t.east, t.north, t.crown_radius) for t in trees]
        kept: list[forest_mod.RealTree] = []
    else:
        no_tree = shapely.union_all(
            [
                asphalt.buffer(REAL_TREE_ROAD_M),
                gravel,
                footprints.buffer(REAL_TREE_ROOF_M),
                site.union("water", "parking"),
                shapely.Point(0, 0).buffer(40.0),
            ]
        )
        inside = shapely.contains_xy(
            no_tree, np.array([t.east for t in real]), np.array([t.north for t in real])
        )
        kept = [t for t, out in zip(real, inside, strict=True) if not out]
        crowns = [(t.east, t.north, t.crown_radius) for t in kept]
        trees = [
            Tree(
                t.east,
                t.north,
                round(float(z), 2),
                "birch" if t.species == "deciduous" else t.species,
                t.height,
                round(t.yaw, 4),
            )
            for t, z in zip(
                kept,
                ground.at(np.array([t.east for t in kept]), np.array([t.north for t in kept])),
                strict=True,
            )
        ]
    paint = Paint(
        areas={kind: site.union(kind) for kind in PAINT_ORDER},
        asphalt=asphalt,
        gravel=gravel,
        markings=markings,
        buildings=footprints,
        crowns=crowns,
    )
    tiles = terrain_tiles(ground)
    for ti, tj, mesh in tiles:
        progress(f"terräng {ti * TILES + tj + 1}/{TILES * TILES}")
        stem = f"terrain_{ti}_{tj}"
        (meshes / f"{stem}.obj").write_text(terrain_obj(mesh, ground, ti, tj), encoding="utf-8")
        mat = Material(stem, (1.0, 1.0, 1.0), 0.02, f"{stem}.png")
        (meshes / f"{stem}.mtl").write_text(mat.mtl(), encoding="utf-8")
        png.write(
            meshes / f"{stem}.png",
            paint_tile(site, paint, ti, tj, TEXTURE_PX, seed),
            PHOTO_PNG_TEXT if site.ortho is not None else None,
        )
    tree_stats: dict[str, Any]
    if atlas is not None:
        progress("träd (Poly Haven-bilder)")
        for name in forest_mod.ATLAS_FILES:
            shutil.copyfile(forest_mod.atlas_folder(repo) / name, meshes / name)
        woods = forest_mod.build_forest(
            kept, np.array([t.z for t in trees]), site.half_size_m, atlas
        )
        tree_files = [f"{stem}.glb" for stem in forest_mod.write_tiles(woods, meshes, ".")]
        tree_stats = {
            "tree_triangles": woods.triangles,
            "trees_near": woods.near,
            "tree_model_triangles": woods.model_triangles,
            "tree_models": "poly-haven-cc0",
        }
    else:
        plain = tree_meshes(trees, site.half_size_m)
        tree_files = []
        for (ti, tj), mesh in sorted(plain.items()):
            write_obj(meshes, f"trees_{ti}_{tj}", mesh, TREE_MATERIALS)
            tree_files.append(f"trees_{ti}_{tj}.obj")
        tree_stats = {
            "tree_triangles": sum(m.triangles() for m in plain.values()),
            "tree_models": "plain",
        }
    if real is not None:
        chm = np.load(site.chm_path) / 10.0 if site.chm_path is not None else None
        check = forest_mod.height_check(kept, chm, site.half_size_m)
        tree_stats |= {
            "tree_source": "raster",
            "trees_raster": len(real),
            "trees_left_out": len(real) - len(kept),
            "tree_height_max_m": max((t.height for t in kept), default=0.0),
            "tree_height_vs_raster_mean_abs_m": None if check is None else round(check, 3),
        }
    else:
        tree_stats["tree_source"] = "generated"
    if buildings:
        write_obj(meshes, "buildings", building_mesh(buildings), BUILDING_MATERIALS)
    if cars:
        write_obj(meshes, "cars", car_mesh(cars), CAR_MATERIALS)
    water = water_mesh(site)
    if not water.empty:
        write_obj(meshes, "water", water, [WATER_MATERIAL])
    (model / "model.sdf").write_text(
        model_sdf(
            site,
            [f"terrain_{ti}_{tj}" for ti, tj, _ in tiles],
            tree_files,
            bool(buildings),
            bool(cars),
            not water.empty,
            write_heightmap(ground, meshes / HEIGHTMAP),
        ),
        encoding="utf-8",
    )
    (model / "model.config").write_text(model_config(site), encoding="utf-8")
    stats = {
        "trees": len(trees),
        **tree_stats,
        **({"ground_texture": "orthophoto"} if site.ortho is not None else {}),
        "buildings": len(buildings),
        "cars": len(cars),
        "terrain_triangles": sum(m.triangles() for _, _, m in tiles),
        "ground_min_m": float(ground.z.min()),
        "ground_max_m": float(ground.z.max()),
    }
    files = {
        str(p.relative_to(repo)): _sha(p)
        for p in sorted([*folder.rglob("*"), *model.rglob("*")])
        if p.is_file() and p.name != "manifest.json"
    }
    manifest = {"version": GENERATOR_VERSION, "inputs": digest, "stats": stats, "files": files}
    manifest_path.write_text(
        json.dumps(manifest, indent=1, sort_keys=True) + "\n", encoding="utf-8"
    )
    return model


def write_heightmap(ground: Ground, path: Path) -> tuple[float, float, float]:
    """The collision heightmap (16-bit PNG, HEIGHTMAP_PX square, row 0 north);
    returns (size, lowest z, height range) for the SDF."""
    xs = np.linspace(-ground.half, ground.half, HEIGHTMAP_PX)
    east, north = np.meshgrid(xs, xs[::-1])
    z = ground.at(east, north)
    zmin, zmax = float(z.min()), float(z.max())
    zrange = max(zmax - zmin, 0.01)
    path.write_bytes(png.encode_gray16(np.round((z - zmin) / zrange * 65535).astype(np.uint16)))
    return 2 * ground.half, round(zmin, 3), round(zrange, 3)


def load_ground(site: Site, repo: Path = paths.REPO) -> Ground:
    return Ground(MESH_STEP_M, site.half_size_m, np.load(out_dir(site, repo) / "ground.npy"))


def model_config(site: Site) -> str:
    return f"""<?xml version="1.0"?>
<model>
  <name>{model_name(site)}</name>
  <version>{GENERATOR_VERSION}</version>
  <sdf version="1.9">model.sdf</sdf>
  <description>Generated by sd_sim.terrain from the site {site.name}.</description>
</model>
"""


def _mesh(uri: str) -> str:
    return f"<geometry><mesh><uri>{uri}</uri></mesh></geometry>"


def model_sdf(
    site: Site,
    terrain: Sequence[str],
    trees: Sequence[str],
    buildings: bool,
    cars: bool,
    water: bool,
    collision: tuple[float, float, float],
) -> str:
    base = f"model://{model_name(site)}/meshes"
    links = []
    parts = []
    # The ground collides as a heightmap: a heightfield in ODE (which Gazebo's DART uses
    # for collisions) costs a fraction of triangle meshes at 1000 physics steps per second.
    # It lies 10 cm under the visible ground, so that at home only the pad's box touches
    # the vehicle (two contact surfaces at the same height make the IMU jitter).
    size, zmin, zrange = collision
    parts.append(
        f'      <collision name="ground"><pose>0 0 {zmin - 0.1:.3f} 0 0 0</pose><geometry>'
        f"<heightmap><uri>{base}/{HEIGHTMAP}</uri><size>{size:g} {size:g} {zrange:.3f}</size>"
        "<pos>0 0 0</pos></heightmap></geometry></collision>"
    )
    for stem in terrain:
        parts.append(f'      <visual name="{stem}">{_mesh(f"{base}/{stem}.obj")}</visual>')
    links.append(("terrain", parts))
    links.append(
        (
            "forest",
            [f'      <visual name="{Path(f).stem}">{_mesh(f"{base}/{f}")}</visual>' for f in trees],
        )
    )
    if buildings:
        links.append(
            (
                "buildings",
                [
                    '      <collision name="buildings">'
                    f"{_mesh(f'{base}/buildings.obj')}</collision>",
                    f'      <visual name="buildings">{_mesh(f"{base}/buildings.obj")}</visual>',
                ],
            )
        )
    if cars:
        links.append(("cars", [f'      <visual name="cars">{_mesh(f"{base}/cars.obj")}</visual>']))
    if water:
        links.append(
            ("water", [f'      <visual name="water">{_mesh(f"{base}/water.obj")}</visual>'])
        )
    body = "\n".join(
        f'    <link name="{name}">\n' + "\n".join(p) + "\n    </link>" for name, p in links if p
    )
    return f"""<?xml version="1.0"?>
<!-- Generated by sd_sim.terrain from the site {site.name}; do not edit. -->
<sdf version="1.9">
  <model name="{model_name(site)}">
    <static>true</static>
{body}
  </model>
</sdf>
"""


def main(argv: Sequence[str] | None = None) -> int:
    """``python -m sd_sim.terrain site=<name> [twice=1]``: build a site's model; with
    ``twice=1`` build it a second time from scratch and fail unless every file is the same."""
    import sys

    from sd_sim import _args
    from sd_sim import site as site_mod
    from sd_sim.util import say

    values = _args.parse(
        list(sys.argv[1:] if argv is None else argv), ("site", "twice"), "sd_sim.terrain"
    )
    s = site_mod.load(values.get("site", "hellasgarden"))
    build(s, force=True, progress=lambda m: say(f"    {m}"))
    first = json.loads((out_dir(s) / "manifest.json").read_text(encoding="utf-8"))
    say(json.dumps(first["stats"], indent=1))
    if values.get("twice") != "1":
        return 0
    build(s, force=True)
    second = json.loads((out_dir(s) / "manifest.json").read_text(encoding="utf-8"))
    differ = sorted(
        k
        for k in set(first["files"]) | set(second["files"])
        if first["files"].get(k) != second["files"].get(k)
    )
    say(f"Två byggen: {len(first['files'])} filer, {len(differ)} skiljer sig")
    for name in differ:
        say(f"  skiljer sig: {name}")
    return 1 if differ else 0


if __name__ == "__main__":
    raise SystemExit(main())
