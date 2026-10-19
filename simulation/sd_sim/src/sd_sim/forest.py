"""The site's real trees as textured meshes.

``tools/terrain/fetch_trees.py`` (``just trees-fetch``) finds every tree in Skogsstyrelsen's
tree height raster and writes ``trees.csv`` (east, north, height, species) next to the
site (``build/trees/<name>/`` for a public site); ``tools/terrain/fetch_tree_models.py``
(``just tree-pictures-fetch``) makes one texture atlas of twigs, needles, leaves, whole
crowns and bark from Poly Haven's CC0 trees in ``build/tree_models/``. Neither is
committed. With both, ``sd_sim.terrain`` builds the forest from them; without, it falls
back to its own plain trees.

Every tree stands where the raster has it and is as tall as the raster says: each species
has one model one metre tall, scaled by the tree's height and turned by a yaw of its own.
Trees within ``NEAR_M`` of home, where the vehicle takes off and flies low, are branch
cards (``*_near``, roughly a thousand triangles); the rest are the crown's picture from
the side twice, crossed, and once from above (``*_far``, eight triangles), which from a
search altitude looks the same. One material for all of them (the atlas, glTF PBR with
alpha cut-outs), one mesh per ``FOREST_TILES`` x ``FOREST_TILES`` tile, so the renderer
skips what the camera does not see.
"""

from __future__ import annotations

import csv
import hashlib
import json
import math
from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
from numpy.typing import NDArray

from sd_sim import gltf, paths
from sd_sim.site import Site

SPECIES = ("spruce", "pine", "deciduous")
NEAR_M = 150.0
FOREST_TILES = 8
# Crown radius as a share of the tree's height (Swedish spruce, Scots pine, birch).
CROWN = {"spruce": 0.17, "pine": 0.19, "deciduous": 0.22}
# The trunk starts this far under the ground so that slopes show no gap under it.
SINK_M = 0.3
ATLAS_FILES = ("trees_color.png", "trees_normal.png", "trees_orm.png")
MATERIAL = "trees"

F = NDArray[np.float64]
Faces = NDArray[np.int64]


@dataclass(frozen=True)
class RealTree:
    east: float
    north: float
    height: float
    species: str

    @property
    def crown_radius(self) -> float:
        return CROWN[self.species] * self.height

    @property
    def yaw(self) -> float:
        key = f"{self.east:.1f},{self.north:.1f}".encode()
        return int.from_bytes(hashlib.sha256(key).digest()[:4], "big") / 2**32 * 2 * math.pi


def load_trees(site: Site) -> list[RealTree] | None:
    if site.trees_folder is None or not site.has_real_trees:
        return None
    path = site.trees_folder / "trees.csv"
    with path.open(encoding="ascii", newline="") as f:
        rows = list(csv.DictReader(f))
    out = []
    for r in rows:
        species = r["species"]
        if species not in SPECIES:
            raise ValueError(f"{path}: okänt trädslag {species!r}")
        out.append(RealTree(float(r["east_m"]), float(r["north_m"]), float(r["height_m"]), species))
    return out


def atlas_folder(repo: Path = paths.REPO) -> Path:
    return repo / "build" / "tree_models"


def load_atlas(repo: Path = paths.REPO) -> dict[str, tuple[float, float, float, float]] | None:
    """Each picture's place in the atlas (u0, v0, u1, v1; v down), or None without one."""
    folder = atlas_folder(repo)
    meta = folder / "atlas.json"
    if not meta.is_file() or not all((folder / f).is_file() for f in ATLAS_FILES):
        return None
    regions = json.loads(meta.read_text(encoding="utf-8"))["regions"]
    return {k: (v[0], v[1], v[2], v[3]) for k, v in regions.items()}


# --- one tree, one metre tall -------------------------------------------------------------------


@dataclass
class Parts:
    """A tree's triangles: positions, normals, tangents (xyz + sign), texture coordinates."""

    pos: list[F]
    nor: list[F]
    tan: list[F]
    uv: list[F]
    faces: list[Faces]
    count: int = 0

    @classmethod
    def new(cls) -> Parts:
        return cls([], [], [], [], [])

    def add(self, pos: F, nor: F, tan: F, uv: F, faces: Faces) -> None:
        self.pos.append(pos)
        self.nor.append(nor)
        self.tan.append(tan)
        self.uv.append(uv)
        self.faces.append(faces + self.count)
        self.count += len(pos)

    def arrays(self) -> tuple[F, F, F, F, Faces]:
        return (
            np.concatenate(self.pos),
            np.concatenate(self.nor),
            np.concatenate(self.tan),
            np.concatenate(self.uv),
            np.concatenate(self.faces),
        )


def _unit(v: F) -> F:
    n = np.linalg.norm(v, axis=-1, keepdims=True)
    return v / np.where(n > 0, n, 1.0)


def card(
    parts: Parts,
    base: F,
    along: F,
    across: F,
    region: tuple[float, float, float, float],
    *,
    picture_up: bool = True,
    bend: F | None = None,
) -> None:
    """A picture on a quad: its bottom edge centred on ``base`` and ``across`` wide (the
    vector is half the width), its height ``along``. With ``picture_up`` false the picture's
    left edge is at ``base`` instead (a branch card: trunk left, tip right). ``bend`` turns
    the normals towards a direction (outwards from the crown) so that cards shade like a
    round crown rather than like flat boards."""
    u0, v0, u1, v1 = region
    p = np.array([base - across, base + across, base + across + along, base - across + along])
    if picture_up:
        uv = np.array([[u0, v1], [u1, v1], [u1, v0], [u0, v0]])
        tangent = _unit(across)
    else:  # left edge at the base: u runs along ``along``, v across
        uv = np.array([[u0, v1], [u0, v0], [u1, v0], [u1, v1]])
        tangent = _unit(along)
    n = _unit(np.cross(across, along))
    if bend is not None:
        if float(np.dot(n, bend)) < 0:
            n = -n
        n = _unit(0.35 * n + 0.65 * _unit(bend))
    parts.add(
        p,
        np.tile(n, (4, 1)),
        np.tile(np.append(tangent, 1.0), (4, 1)),
        uv,
        np.array([[0, 1, 2], [0, 2, 3]]),
    )


def trunk(
    parts: Parts,
    z0: float,
    z1: float,
    r0: float,
    r1: float,
    sides: int,
    region: tuple[float, float, float, float],
    lean: F | None = None,
) -> None:
    """A tapered prism from (0, 0, z0) to (0, 0, z1) (or to ``lean`` at the top), bark on
    its sides."""
    u0, v0, u1, v1 = region
    top = np.array([0.0, 0.0, z1]) if lean is None else lean
    bottom = np.array([0.0, 0.0, z0])
    ang = np.arange(sides + 1) * 2 * math.pi / sides
    ring = np.stack([np.cos(ang), np.sin(ang), np.zeros_like(ang)], 1)
    lo = bottom + r0 * ring
    hi = top + r1 * ring
    nor = ring.copy()
    tan = np.stack([-np.sin(ang), np.cos(ang), np.zeros_like(ang), np.ones_like(ang)], 1)
    us = u0 + (u1 - u0) * np.arange(sides + 1) / sides
    uv = np.concatenate(
        [np.stack([us, np.full_like(us, v1)], 1), np.stack([us, np.full_like(us, v0)], 1)]
    )
    faces = []
    n = sides + 1
    for i in range(sides):
        faces += [[i, i + 1, n + i + 1], [i, n + i + 1, n + i]]
    parts.add(
        np.vstack([lo, hi]), np.vstack([nor, nor]), np.vstack([tan, tan]), uv, np.array(faces)
    )


def _rng(species: str) -> np.random.Generator:
    return np.random.default_rng(int.from_bytes(species.encode(), "big") % 2**32)


def _tufts(
    parts: Parts,
    rng: np.random.Generator,
    region: tuple[float, float, float, float],
    count: int,
    centre: F,
    radii: F,
    size: float,
) -> None:
    """Needle tufts or leaf clusters in an ellipsoidal crown, mostly near its surface: two
    crossed upright cards and one facing up and out each."""
    for _ in range(count):
        d = _unit(rng.normal(size=3))
        r = rng.uniform(0.55, 1.0) ** 0.5
        at = centre + d * radii * r
        out = _unit((at - centre) / radii)
        yaw = rng.uniform(0, math.pi)
        s = size * rng.uniform(0.8, 1.2)
        for k in range(2):
            a = yaw + k * math.pi / 2
            across = np.array([math.cos(a), math.sin(a), 0.0]) * s / 2
            card(
                parts, at - np.array([0, 0, s * 0.3]), np.array([0, 0, s]), across, region, bend=out
            )
        # One lying card, tipped outwards, seen from above.
        flat = _unit(np.array([out[0], out[1], 0.0]) + np.array([1e-6, 0, 0]))
        side = np.cross(np.array([0.0, 0.0, 1.0]), flat)
        along = _unit(flat * s + np.array([0, 0, s * 0.35])) * s
        card(parts, at - along / 2, along, side * s / 2, region, bend=out + np.array([0, 0, 0.5]))


def near_model(species: str, atlas: dict[str, tuple[float, float, float, float]]) -> Parts:
    """The detailed model, 1 m tall: trunk and branch cards."""
    rng = _rng(species)
    parts = Parts.new()
    crown = CROWN[species]
    bark, cardr = atlas[f"{species}_bark"], atlas[f"{species}_card"]
    if species == "spruce":
        trunk(parts, -0.0, 0.98, 0.014, 0.002, 6, bark)
        levels = 26
        for i in range(levels):
            t = i / (levels - 1)
            z = 0.1 + 0.86 * t
            length = crown * (1 - t) * 1.05 + 0.02
            n = round(7 - 3 * t)
            for k in range(n):
                a = (k / n) * 2 * math.pi + i * 2.4 + rng.uniform(-0.2, 0.2)
                d = np.array([math.cos(a), math.sin(a), 0.0])
                along = d * length + np.array([0, 0, -0.22 * length])
                base = np.array([0.0, 0.0, z])
                out = _unit(d + np.array([0, 0, 0.4]))
                width = 0.5 * length
                side = np.cross(np.array([0.0, 0.0, 1.0]), d) * width / 2
                card(parts, base, along, side, cardr, picture_up=False, bend=out)
                up = _unit(np.cross(along, side)) * width / 2 * 0.8
                card(parts, base, along, up, cardr, picture_up=False, bend=out)
    elif species == "pine":
        trunk(parts, 0.0, 0.86, 0.016, 0.004, 6, bark)
        for k in range(5):  # limbs into the crown
            a = k * 2 * math.pi / 5 + 0.3
            tip = np.array([math.cos(a) * crown * 0.6, math.sin(a) * crown * 0.6, 0.8])
            sub = Parts.new()
            trunk(sub, 0.0, 0.0, 0.006, 0.002, 4, bark, lean=tip - np.array([0, 0, 0.66]))
            p, nn, tt, uu, ff = sub.arrays()
            parts.add(p + np.array([0, 0, 0.66]), nn, tt, uu, ff)
        _tufts(
            parts, rng, cardr, 140, np.array([0.0, 0.0, 0.8]), np.array([crown, crown, 0.13]), 0.09
        )
    else:
        trunk(parts, 0.0, 0.55, 0.014, 0.006, 6, bark)
        for k in range(4):
            a = k * math.pi / 2 + 0.6
            tip = np.array([math.cos(a) * crown * 0.5, math.sin(a) * crown * 0.5, 0.78])
            sub = Parts.new()
            trunk(sub, 0.0, 0.0, 0.006, 0.002, 4, bark, lean=tip - np.array([0, 0, 0.45]))
            p, nn, tt, uu, ff = sub.arrays()
            parts.add(p + np.array([0, 0, 0.45]), nn, tt, uu, ff)
        _tufts(
            parts, rng, cardr, 150, np.array([0.0, 0.0, 0.64]), np.array([crown, crown, 0.32]), 0.1
        )
    return _one_metre(parts)


def _one_metre(parts: Parts) -> Parts:
    """The model stretched or squeezed upright so that its top is exactly 1 m: a tree in
    the world is then exactly as tall as the raster says."""
    top = max(float(p[:, 2].max()) for p in parts.pos)
    parts.pos = [p * np.array([1.0, 1.0, 1.0 / top]) for p in parts.pos]
    return parts


def far_model(species: str, atlas: dict[str, tuple[float, float, float, float]]) -> Parts:
    """The cheap model, 1 m tall: the side picture on two crossed upright quads and the
    picture from above on a level one."""
    parts = Parts.new()
    crown = CROWN[species]
    side, top = atlas[f"{species}_side"], atlas[f"{species}_top"]
    for k in range(2):
        a = k * math.pi / 2
        across = np.array([math.cos(a), math.sin(a), 0.0]) * crown
        card(
            parts,
            np.zeros(3),
            np.array([0.0, 0.0, 1.0]),
            across,
            side,
            bend=np.cross(np.array([0.0, 0.0, 1.0]), across) + np.array([0, 0, 0.3]),
        )
    level = {"spruce": 0.4, "pine": 0.82, "deciduous": 0.66}[species]
    # Level: ``along`` points north, ``across`` east, the normal up.
    card(
        parts,
        np.array([0.0, -crown, level]),
        np.array([0.0, 2 * crown, 0.0]),
        np.array([crown, 0.0, 0.0]),
        top,
        bend=np.array([0.0, 0.0, 1.0]),
    )
    return parts


# --- the forest ------------------------------------------------------------------------------


def place(model: Parts, tree: RealTree, z: float) -> tuple[F, F, F, F, Faces]:
    pos, nor, tan, uv, faces = model.arrays()
    c, s = math.cos(tree.yaw), math.sin(tree.yaw)
    rot = np.array([[c, -s, 0.0], [s, c, 0.0], [0.0, 0.0, 1.0]])
    h = tree.height
    p = pos @ rot.T * h + np.array([tree.east, tree.north, z])
    # The trunk's foot goes SINK_M into the ground; the top stays at the tree's height.
    p[:, 2] -= SINK_M * (pos[:, 2] <= 1e-9)
    t = np.concatenate([tan[:, :3] @ rot.T, tan[:, 3:]], axis=1)
    return p, nor @ rot.T, t, uv, faces


@dataclass
class Forest:
    tiles: dict[tuple[int, int], gltf.Primitive]
    trees: int
    near: int
    triangles: int
    model_triangles: dict[str, int]


def build_forest(
    trees: Sequence[RealTree],
    ground_z: NDArray[np.float64],
    half: float,
    atlas: dict[str, tuple[float, float, float, float]],
) -> Forest:
    """Every tree as a mesh, merged per tile; ``ground_z`` is the ground under each tree
    (world z)."""
    near = {sp: near_model(sp, atlas) for sp in SPECIES}
    far = {sp: far_model(sp, atlas) for sp in SPECIES}
    size = 2 * half / FOREST_TILES
    tiles: dict[tuple[int, int], gltf.Primitive] = {}
    n_near = 0
    for tree, z in zip(trees, ground_z, strict=True):
        is_near = math.hypot(tree.east, tree.north) < NEAR_M
        n_near += is_near
        model = (near if is_near else far)[tree.species]
        ti = min(int((half - tree.north) / size), FOREST_TILES - 1)
        tj = min(int((tree.east + half) / size), FOREST_TILES - 1)
        tiles.setdefault((ti, tj), gltf.Primitive()).add(*place(model, tree, float(z)))
    counts = {f"{sp}_near": sum(len(f) for f in near[sp].faces) for sp in SPECIES}
    counts |= {f"{sp}_far": sum(len(f) for f in far[sp].faces) for sp in SPECIES}
    return Forest(
        tiles,
        len(trees),
        n_near,
        sum(p.triangles() for p in tiles.values()),
        counts,
    )


def material(textures: str) -> gltf.PbrMaterial:
    return gltf.PbrMaterial(
        MATERIAL,
        f"{textures}/trees_color.png",
        normal=f"{textures}/trees_normal.png",
        orm=f"{textures}/trees_orm.png",
        alpha_cutoff=0.5,
    )


def write_tiles(forest: Forest, folder: Path, textures: str) -> list[str]:
    """One ``trees_<i>_<j>.glb`` per tile in ``folder``; returns their stems."""
    stems = []
    for (ti, tj), prim in sorted(forest.tiles.items()):
        stem = f"trees_{ti}_{tj}"
        (folder / f"{stem}.glb").write_bytes(gltf.glb({MATERIAL: prim}, [material(textures)], stem))
        stems.append(stem)
    return stems


def height_check(trees: Sequence[RealTree], chm: NDArray[Any] | None, half: float) -> float | None:
    """Mean |model height - raster| over the trees: the raster's highest value (m) within
    the radius ``tools/terrain/fetch_trees.py`` looked for the tree's top in (0.8 m +
    9 % of its height, 1.5-4 m), on the 1 m grid ``chm`` (row 0 north)."""
    if chm is None or not trees:
        return None
    n = chm.shape[0]
    errors = []
    for t in trees:
        r = int((half - t.north) // 1.0)
        c = int((t.east + half) // 1.0)
        k = math.ceil(min(max(0.8 + 0.09 * t.height, 1.5), 4.0))
        win = chm[max(r - k, 0) : min(r + k + 1, n), max(c - k, 0) : min(c + k + 1, n)]
        if win.size:
            errors.append(abs(t.height - float(win.max())))
    return float(np.mean(errors)) if errors else None
