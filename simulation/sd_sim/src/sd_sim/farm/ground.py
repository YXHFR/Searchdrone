"""Detailed ground under the photo farm's pictures.

The site's terrain is painted at about 15 cm per pixel (``sd_sim.terrain``), which a
camera at 1-5 cm per pixel shows as smooth colour. Under every picture of a scene the
photo farm lays a **patch**: a copy of the terrain's own mesh over a square, a few
centimetres above it, whose texture is the painted ground's colour times the fine
detail (``DETAIL_M``) of a photographed ground texture of the right kind (grass, forest
floor, gravel; ``just ground-pictures-fetch``, Poly Haven CC0) at its real size on the
ground. So the colours
still follow the map and the trees, and the pixels show grass, needles and stones.

``plan(shots, ...)`` groups the shots' ground footprints into patches (``MAX_SIDE_M``);
``Patch.build`` writes the patch's model (mesh, texture) to
``build/sim/models/sd_ground_patch_i<N>_<k>/``; the patch is a ``WorldObject`` placed
like any other. Patches that overlap are stacked a centimetre apart, and a person on a
patch is raised by its lift (``lift_at``) so the feet stay on the visible ground.

Without the ground pictures the farm runs without patches and says so.
"""

from __future__ import annotations

import json
import math
import random
import shutil
from collections.abc import Callable, Sequence
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np
import shapely
from numpy.typing import NDArray

from sd_sim import forest, objects, paths, raster, site, terrain

MAX_SIDE_M = {"heridal": 110.0, "kios": 240.0}
MAX_PX = 4096
MIN_CELL_M = 0.012  # finer than this the photographs have no more to show
KIND_CELL_M = 0.25  # the grid the kinds of ground are drawn on
LIFT_M = 0.04
LIFT_STEP_M = 0.01
MARGIN = 1.15  # a footprint is grown by this much before it is covered
KINDS = ("grass", "forest", "path", "rock")
CROWN_FLOOR = 1.2  # forest floor out to this share of a crown's radius
# The photos give only detail finer than this (their own larger patches are dropped)
# and change the brightness by at most this share either way.
DETAIL_M = 0.5
DETAIL_RANGE = 0.45
TEXTURES = Path("build") / "ground_textures"


@dataclass(frozen=True)
class Patch:
    """A square of detailed ground, a ``WorldObject`` (its origin on the ground at the
    centre; the mesh is in the model)."""

    index: int
    instance: int
    east_m: float
    north_m: float
    side_m: float
    px: int
    lift_m: float
    seed: int
    up_m: float = 0.0
    # the mesh is built in east/north; a compass heading of 90 is the world yaw 0
    yaw_deg: float = 90.0
    label: int = 0
    footprint_m: float = 0.1
    truth: dict[str, Any] = field(default_factory=lambda: {"role": "ground"})

    @property
    def name(self) -> str:
        return f"sd_ground_patch_i{self.instance}_{self.index}"

    @property
    def cell_m(self) -> float:
        return self.side_m / self.px

    def bounds(self) -> tuple[float, float, float, float]:
        h = self.side_m / 2
        return (self.east_m - h, self.north_m - h, self.east_m + h, self.north_m + h)

    def covers(self, e: float, n: float) -> bool:
        x0, y0, x1, y1 = self.bounds()
        return x0 <= e <= x1 and y0 <= n <= y1

    def sdf(self, pose: objects.Pose) -> str:
        return f"""<model name="{self.name}">
    <static>true</static>
    <pose>{pose.sdf()}</pose>
    <link name="link">
      <visual name="visual">
        <cast_shadows>false</cast_shadows>
        <geometry><mesh><uri>model://{self.name}/{self.name}.obj</uri></mesh></geometry>
      </visual>
    </link>
  </model>"""

    def build(
        self,
        s: site.Site,
        ground: terrain.Ground,
        land: Land,
        repo: Path = paths.REPO,
    ) -> None:
        folder = repo / "build" / "sim" / "models" / self.name
        shutil.rmtree(folder, ignore_errors=True)
        folder.mkdir(parents=True)
        z0 = float(ground.at(np.array([self.east_m]), np.array([self.north_m]))[0])
        # Every file and the material are named after the patch: the renderer keeps
        # textures and materials by name, so patches sharing "patch.jpg" would all show
        # the first one's picture.
        name = self.name
        (folder / f"{name}.obj").write_text(self._obj(ground, z0), encoding="utf-8")
        (folder / f"{name}.mtl").write_text(
            f"newmtl {name}\nKa 0.6 0.6 0.6\nKd 1 1 1\nKs 0.02 0.02 0.02\nmap_Kd {name}.jpg\n",
            encoding="utf-8",
        )
        texture = land.texture(self)
        from PIL import Image  # the dev group's Pillow

        Image.fromarray(texture).save(folder / f"{name}.jpg", quality=92)
        (folder / "model.config").write_text(
            f'<?xml version="1.0"?>\n<model><name>{self.name}</name><sdf version="1.9">'
            "patch.sdf</sdf></model>\n",
            encoding="utf-8",
        )

    def _obj(self, ground: terrain.Ground, z0: float) -> str:
        """The terrain's own triangles inside the square (the same nodes and diagonals,
        so the patch never cuts into the ground), lifted by ``lift_m``, relative to the
        model's origin."""
        step, half = ground.step, ground.half
        x0, y0, x1, y1 = self.bounds()
        c0 = max(0, math.floor((x0 + half) / step))
        c1 = min(ground.z.shape[1] - 1, math.ceil((x1 + half) / step))
        r0 = max(0, math.floor((half - y1) / step))
        r1 = min(ground.z.shape[0] - 1, math.ceil((half - y0) / step))
        rows, cols = np.arange(r0, r1 + 1), np.arange(c0, c1 + 1)
        ee, nn = np.meshgrid(cols * step - half, half - rows * step)
        zz = ground.z[np.ix_(rows, cols)]
        u = (ee - x0) / self.side_m
        v = (nn - y0) / self.side_m
        lines = [f"mtllib {self.name}.mtl", f"usemtl {self.name}"]
        for e, n, z in zip(ee.ravel(), nn.ravel(), zz.ravel(), strict=True):
            lines.append(
                f"v {e - self.east_m:.3f} {n - self.north_m:.3f} {z - z0 + self.lift_m:.3f}"
            )
        lines += [f"vt {a:.6f} {b:.6f}" for a, b in zip(u.ravel(), v.ravel(), strict=True)]
        lines.append("vn 0 0 1")
        m = len(cols)
        idx = np.arange(len(rows) * m).reshape(len(rows), m) + 1
        a, b = idx[:-1, :-1].ravel(), idx[:-1, 1:].ravel()
        c, d = idx[1:, :-1].ravel(), idx[1:, 1:].ravel()
        for tri in (*zip(a, c, b, strict=True), *zip(b, c, d, strict=True)):
            lines.append("f " + " ".join(f"{k}/{k}/1" for k in tri))
        return "\n".join(lines) + "\n"


def footprint(
    e: float, n: float, height_m: float, hfov_deg: float, aspect: float, yaw_deg: float
) -> Any:
    """The ground a straight-down picture covers (a rotated rectangle), grown by
    ``MARGIN`` for tilt and slopes."""
    half_w = height_m * math.tan(math.radians(hfov_deg) / 2) * MARGIN
    half_h = half_w * aspect
    rect = shapely.box(-half_w, -half_h, half_w, half_h)
    from shapely import affinity

    return affinity.translate(affinity.rotate(rect, -yaw_deg, origin=(0, 0)), e, n)


def plan(
    footprints: Sequence[tuple[Any, float]],
    profile: str,
    instance: int,
    seed: int,
) -> list[Patch]:
    """Patches that cover every footprint ``(polygon, ground cm per pixel)``: each joins
    the first patch it still fits in with the patch at most ``MAX_SIDE_M`` across."""
    limit = MAX_SIDE_M[profile]
    groups: list[tuple[list[float], float]] = []  # bounds, finest gsd (m)
    for poly, gsd_cm in footprints:
        bx0, by0, bx1, by1 = poly.bounds
        for k, (b, finest) in enumerate(groups):
            nb = [min(b[0], bx0), min(b[1], by0), max(b[2], bx1), max(b[3], by1)]
            if max(nb[2] - nb[0], nb[3] - nb[1]) <= limit:
                groups[k] = (nb, min(finest, gsd_cm / 100))
                break
        else:
            groups.append(([bx0, by0, bx1, by1], gsd_cm / 100))
    out = []
    for k, (b, finest) in enumerate(groups):
        side = max(b[2] - b[0], b[3] - b[1]) + 2.0
        ce, cn = (b[0] + b[2]) / 2, (b[1] + b[3]) / 2
        if math.hypot(ce, cn) < objects.HOME_KEEP_OUT_M + 1.0:
            # No object may stand at home: move the centre out and grow the square.
            ce += objects.HOME_KEEP_OUT_M + 2.0
            side += 2 * (objects.HOME_KEEP_OUT_M + 2.0)
        cell = max(MIN_CELL_M, finest * 0.8)
        px = int(min(MAX_PX, 2 ** math.ceil(math.log2(max(256.0, side / cell)))))
        out.append(
            Patch(
                index=k,
                instance=instance,
                east_m=round(ce, 2),
                north_m=round(cn, 2),
                side_m=round(side, 2),
                px=px,
                lift_m=round(LIFT_M + LIFT_STEP_M * k, 3),
                seed=seed * 100 + k,
            )
        )
    return out


def lift_at(patches: Sequence[Patch], e: float, n: float) -> float:
    """How high the visible ground is over the terrain at a point: the top patch's lift."""
    return max((p.lift_m for p in patches if p.covers(e, n)), default=0.0)


# --- the texture ------------------------------------------------------------------------


def textures_ready(repo: Path = paths.REPO) -> bool:
    return (repo / TEXTURES / "ground.json").is_file()


@dataclass
class Photo:
    pixels: NDArray[np.float32]  # h x w x 3, 0-1
    size_m: tuple[float, float]
    mean: NDArray[np.float32]  # its mean colour


def fine_detail(im: Any, size_m: float) -> NDArray[np.float32]:
    """A ground photo's brightness detail finer than ``DETAIL_M`` (around 1, grey): the
    painted map gives the colour and the large patches, the photo only the grass blades,
    needles and stones. Its own large patches (moss on rock, sand streaks) would
    otherwise show from the air as blotches that are not on the map."""
    from PIL import ImageFilter  # the dev group's Pillow

    grey = im.convert("L")
    width = grey.size[0]
    sigma = min(max(DETAIL_M / (size_m / width), 2.0), width / 8)
    lum = np.asarray(grey, np.float32) + 1.0
    soft = np.asarray(grey.filter(ImageFilter.GaussianBlur(sigma)), np.float32) + 1.0
    ratio = np.clip(lum / soft, 1 - DETAIL_RANGE, 1 + DETAIL_RANGE)
    return np.repeat(ratio[..., None], 3, axis=2).astype(np.float32)


class Land:
    """The painted ground and the kinds of ground of a site, for the patches' textures."""

    def __init__(self, s: site.Site, ground: terrain.Ground, repo: Path = paths.REPO) -> None:
        from PIL import Image  # the dev group's Pillow

        self.site = s
        self.repo = repo
        self.half = s.half_size_m
        self.tile = 2 * self.half / terrain.TILES
        meshes = repo / "build" / "sim" / "models" / terrain.model_name(s) / "meshes"
        self.tiles: dict[tuple[int, int], NDArray[np.uint8]] = {}
        for ti in range(terrain.TILES):
            for tj in range(terrain.TILES):
                with Image.open(meshes / f"terrain_{ti}_{tj}.png") as im:
                    self.tiles[(ti, tj)] = np.asarray(im.convert("RGB"))
        self.tile_px = next(iter(self.tiles.values())).shape[0]
        info = json.loads((repo / TEXTURES / "ground.json").read_text(encoding="utf-8"))
        self.photos: dict[str, list[Photo]] = {k: [] for k in KINDS}
        for asset, row in sorted(info["textures"].items()):
            if row["kind"] not in self.photos:
                continue
            size = (float(row["size_m"][0]), float(row["size_m"][1]))
            with Image.open(repo / TEXTURES / row["file"]) as im:
                pix = fine_detail(im, size[0])
            self.photos[row["kind"]].append(Photo(pix, size, np.ones(3, np.float32)))
            del asset
        asphalt, gravel, _ = terrain.road_geometry(s)
        self.asphalt = asphalt
        self.paths = shapely.union_all([gravel, s.union("parking")])
        self.rock = s.union("rock", "sand")
        self.forest = s.union("forest", "wetland", "scrub")
        self.water = s.union("water")
        trees = forest.load_trees(s) or []
        self.trees = np.array([(t.east, t.north, t.crown_radius) for t in trees]).reshape(-1, 3)

    def base(self, e: NDArray[np.float64], n: NDArray[np.float64]) -> NDArray[np.float32]:
        """The painted ground's colour (0-1) at points, bilinear between its pixels."""
        out = np.zeros((*e.shape, 3), np.float32)
        ti = np.clip(((self.half - n) // self.tile).astype(np.int64), 0, terrain.TILES - 1)
        tj = np.clip(((e + self.half) // self.tile).astype(np.int64), 0, terrain.TILES - 1)
        px = self.tile_px
        for key, img in self.tiles.items():
            sel = (ti == key[0]) & (tj == key[1])
            if not sel.any():
                continue
            west = -self.half + key[1] * self.tile
            north = self.half - key[0] * self.tile
            col = np.clip((e[sel] - west) / self.tile * px - 0.5, 0, px - 1.001)
            row = np.clip((north - n[sel]) / self.tile * px - 0.5, 0, px - 1.001)
            c0, r0 = col.astype(np.int64), row.astype(np.int64)
            tc, tr = (col - c0)[:, None], (row - r0)[:, None]

            def at(r: NDArray[np.int64], c: NDArray[np.int64], img=img) -> NDArray[np.float32]:
                return img[r, c].astype(np.float32) / 255.0

            top = at(r0, c0) * (1 - tc) + at(r0, c0 + 1) * tc
            bot = at(r0 + 1, c0) * (1 - tc) + at(r0 + 1, c0 + 1) * tc
            out[sel] = top * (1 - tr) + bot * tr
        return out

    def kinds(self, patch: Patch) -> NDArray[np.float32]:
        """Weights of each kind of ground (``KINDS`` order, plus none for asphalt and
        water) on a ``KIND_CELL_M`` grid over the patch, softened at the borders."""
        x0, y0, x1, y1 = patch.bounds()
        n = math.ceil(patch.side_m / KIND_CELL_M)
        fr = (x0, y1, KIND_CELL_M, n, n)
        clip = shapely.box(x0 - 2, y0 - 2, x1 + 2, y1 + 2)

        def mask(geom: Any) -> NDArray[np.bool_]:
            g = geom.intersection(clip) if not geom.is_empty else geom
            return raster.mask(g, *fr) if not g.is_empty else np.zeros((n, n), bool)

        forest_m = mask(self.forest)
        xs = x0 + (np.arange(n) + 0.5) * KIND_CELL_M
        ys = y1 - (np.arange(n) + 0.5) * KIND_CELL_M
        near = self.trees[
            (self.trees[:, 0] > x0 - 15)
            & (self.trees[:, 0] < x1 + 15)
            & (self.trees[:, 1] > y0 - 15)
            & (self.trees[:, 1] < y1 + 15)
        ]
        for te, tn, r in near:
            rr = r * CROWN_FLOOR
            ci = np.flatnonzero(np.abs(xs - te) < rr)
            ri = np.flatnonzero(np.abs(ys - tn) < rr)
            if len(ci) == 0 or len(ri) == 0:
                continue
            d = np.hypot(xs[ci][None, :] - te, ys[ri][:, None] - tn)
            forest_m[np.ix_(ri, ci)] |= d < rr
        path_m = mask(self.paths)
        rock_m = mask(self.rock)
        none_m = mask(self.asphalt) | mask(self.water)
        w = np.zeros((len(KINDS) + 1, n, n), np.float32)
        w[4] = none_m
        w[2] = path_m & ~none_m
        w[3] = rock_m & ~none_m & ~path_m
        w[1] = forest_m & ~none_m & ~path_m & ~rock_m
        w[0] = 1.0 - w[1:].sum(0)
        w = np.stack([np.asarray(raster.soften(c.astype(np.float64), 2.0), np.float32) for c in w])
        return (w / np.maximum(w.sum(0), 1e-6)).astype(np.float32)

    def texture(self, patch: Patch) -> NDArray[np.uint8]:
        """The patch's texture: the painted colour times each kind's photograph, in rows
        of ``STRIP`` pixels so the memory stays small."""
        rng = random.Random(patch.seed)  # noqa: S311 - scenery
        kinds = self.kinds(patch)
        nk = kinds.shape[1]
        # For each kind one photograph, turned and moved at random.
        chosen = []
        for kind in KINDS:
            photos = self.photos[kind]
            if not photos:
                chosen.append(None)
                continue
            ph = rng.choice(photos)
            ang = rng.uniform(0, 2 * math.pi)
            chosen.append(
                (ph, math.cos(ang), math.sin(ang), rng.uniform(0, 100), rng.uniform(0, 100))
            )
        px = patch.px
        x0, _, _, y1 = patch.bounds()
        out = np.zeros((px, px, 3), np.uint8)
        strip = 256
        cols = x0 + (np.arange(px) + 0.5) * patch.cell_m
        for r0 in range(0, px, strip):
            rows = y1 - (np.arange(r0, min(px, r0 + strip)) + 0.5) * patch.cell_m
            ee, nn = (g.astype(np.float64) for g in np.meshgrid(cols, rows))
            base = self.base(ee, nn)
            ki = np.clip(((y1 - nn) / KIND_CELL_M).astype(np.int64), 0, nk - 1)
            kj = np.clip(((ee - x0) / KIND_CELL_M).astype(np.int64), 0, nk - 1)
            detail = np.zeros_like(base)
            total = np.zeros(base.shape[:2], np.float32)
            for k, c in enumerate(chosen):
                wk = kinds[k][ki, kj]
                if c is None or not wk.any():
                    continue
                ph, ca, sa, ox, oy = c
                u = (ca * ee - sa * nn + ox) / ph.size_m[0]
                v = (sa * ee + ca * nn + oy) / ph.size_m[1]
                h, w = ph.pixels.shape[:2]
                pc = ((u % 1.0) * w).astype(np.int64) % w
                pr = ((1 - v % 1.0) * h).astype(np.int64) % h
                detail += (ph.pixels[pr, pc] / np.maximum(ph.mean, 1e-3)) * wk[..., None]
                total += wk
            none = np.clip(1.0 - total, 0.0, 1.0)[..., None]
            colour = base * (detail + none)
            out[r0 : r0 + len(rows)] = np.clip(colour * 255 + 0.5, 0, 255).astype(np.uint8)
        return out


def make(
    patches: Sequence[Patch],
    s: site.Site,
    ground: terrain.Ground,
    repo: Path = paths.REPO,
    say: Callable[[str], None] = print,
) -> None:
    land = Land(s, ground, repo)
    for p in patches:
        say(f"  mark {p.index + 1}/{len(patches)}: {p.side_m:.0f} m, {p.cell_m * 100:.1f} cm/px")
        p.build(s, ground, land, repo)
