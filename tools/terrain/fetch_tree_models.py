"""Fetch the simulator's tree pictures: ``just tree-pictures-fetch``.

The trees in a site's world (``sd_sim.terrain``) are low-poly meshes: branch cards for
the trees near home and a few crossed pictures of the whole crown (impostors) for the
rest. Their pictures are made here from Poly Haven's CC0 tree assets (the spruce
``fir_tree_01``, the pine ``pine_tree_01``, the deciduous ``island_tree_02`` and the bark
textures ``pine_bark`` and ``bark_brown_02``; https://polyhaven.com/license): twigs,
needles and leaves are cut out of the assets' own texture sheets and stamped into branch
cards, and the cards into side and top views of a whole crown.

Everything lands in ``build/tree_models/``, never in the repo: one texture atlas of
``ATLAS_PX`` square (``trees_color.png`` with alpha, ``trees_normal.png`` OpenGL normals,
``trees_orm.png`` occlusion-roughness-metallic) and ``atlas.json`` with each picture's
place in the atlas and the downloaded files' MD5 sums. Without it the world falls back
to the plain low-poly trees with flat colours.

Needs internet and Pillow (a dev dependency) only while it runs.
Exit codes: 0 done, 1 a download or a check failed.
"""

from __future__ import annotations

import hashlib
import io
import json
import math
import sys
import time
import urllib.error
import urllib.request
from collections.abc import Sequence
from pathlib import Path
from typing import Any

import numpy as np
from numpy.typing import NDArray

sys.path.insert(0, str(Path(__file__).resolve().parent))
from fetch_site import ROOT, USER_AGENT, FetchError

API = "https://api.polyhaven.com/files/{asset}"
# (asset, map) pairs, all at 1k.
MAPS = {
    "spruce_twig": ("fir_tree_01", "twig_diff", "twig_alpha"),
    "pine_twig": ("pine_tree_01", "twig_diff", "twig_alpha"),
    "leaves": ("island_tree_02", "leaves_diff", "leaves_alpha"),
}
BARK = {
    "spruce": ("fir_tree_01", "bark_diff", "bark_nor_gl", "bark_arm"),
    "pine": ("pine_bark", "Diffuse", "nor_gl", "arm"),
    "deciduous": ("bark_brown_02", "Diffuse", "nor_gl", "arm"),
}
SPECIES = ("spruce", "pine", "deciduous")
ATLAS_PX = 2048
CELL = ATLAS_PX // 4
# Each picture keeps this far (px) from its cell's edge, so the renderer's smaller mip
# levels do not blend neighbouring pictures.
PAD = 8
INNER = CELL - 2 * PAD
# Atlas rows (each CELL high), one column per species: branch card, side view, top view,
# bark. The fourth column is unused.
ROWS = ("card", "side", "top", "bark")
SEED = 20231
TRIES = 4

Rgba = NDArray[np.float32]  # h x w x 4, 0-1, colour not premultiplied


# --- download -----------------------------------------------------------------------------


def _get(url: str) -> bytes:
    req = urllib.request.Request(url, headers={"User-Agent": USER_AGENT})  # noqa: S310
    last: Exception | None = None
    for attempt in range(TRIES):
        try:
            with urllib.request.urlopen(req, timeout=60) as resp:  # noqa: S310
                return resp.read()
        except (urllib.error.URLError, TimeoutError, ConnectionError) as exc:
            last = exc
            time.sleep(2**attempt)
    raise FetchError(f"kunde inte hämta {url}: {last}")


class Source:
    """Poly Haven downloads, checked against the MD5 sums the API lists."""

    def __init__(self) -> None:
        self.files: dict[str, dict[str, str]] = {}
        self._index: dict[str, Any] = {}

    def picture(self, asset: str, name: str, mode: str) -> NDArray[np.float32]:
        from PIL import Image  # MIT-CMU (dev dependency)

        if asset not in self._index:
            self._index[asset] = json.loads(_get(API.format(asset=asset)))
        try:
            entry = self._index[asset][name]["1k"]["png"]
        except KeyError as exc:
            raise FetchError(f"Poly Haven: {asset} saknar {name} (1k png)") from exc
        data = _get(entry["url"])
        md5 = hashlib.md5(data, usedforsecurity=False).hexdigest()
        if md5 != entry["md5"]:
            raise FetchError(f"Poly Haven: {entry['url']} har fel MD5")
        self.files[f"{asset}/{name}"] = {"url": entry["url"], "md5": md5}
        img = Image.open(io.BytesIO(data)).convert(mode)
        return np.asarray(img, dtype=np.float32) / 255.0


# --- cutting sprites out of a texture sheet ------------------------------------------------


def components(mask: NDArray[np.bool_]) -> list[tuple[int, int, int, int, int]]:
    """Bounding boxes (row0, col0, row1, col1, cells) of the 4-connected parts of a mask."""
    h, w = mask.shape
    seen = np.zeros_like(mask)
    out = []
    for r0 in range(h):
        for c0 in range(w):
            if not mask[r0, c0] or seen[r0, c0]:
                continue
            stack = [(r0, c0)]
            seen[r0, c0] = True
            rows, cols = [r0], [c0]
            while stack:
                r, c = stack.pop()
                for rr, cc in ((r + 1, c), (r - 1, c), (r, c + 1), (r, c - 1)):
                    if 0 <= rr < h and 0 <= cc < w and mask[rr, cc] and not seen[rr, cc]:
                        seen[rr, cc] = True
                        stack.append((rr, cc))
                        rows.append(rr)
                        cols.append(cc)
            out.append((min(rows), min(cols), max(rows) + 1, max(cols) + 1, len(rows)))
    return out


def sprites(diff: NDArray[np.float32], alpha: NDArray[np.float32], min_share: float) -> list[Rgba]:
    """The green parts of a sheet (twigs, needles, leaves) as separate RGBA pictures,
    largest first. Bark, cones and the sheet's filler are not green and are left out."""
    r, g, b = diff[..., 0], diff[..., 1], diff[..., 2]
    green = (alpha > 0.5) & (g > r * 1.02) & (g > b * 1.15)
    k = 8  # find the parts on a coarse grid, then cut at full resolution
    h, w = green.shape
    coarse = green[: h // k * k, : w // k * k].reshape(h // k, k, w // k, k).mean(axis=(1, 3))
    boxes = [bx for bx in components(coarse > 0.15) if bx[4] >= min_share * coarse.size]
    out = []
    for r0, c0, r1, c1, _ in sorted(boxes, key=lambda bx: -bx[4]):
        sl = (slice(r0 * k, r1 * k), slice(c0 * k, c1 * k))
        a = alpha[sl] * _dilate(green[sl], 3)
        out.append(np.dstack([diff[sl], a]).astype(np.float32))
    if not out:
        raise FetchError("hittade inga kvistar eller blad i texturen")
    return out


def _dilate(mask: NDArray[np.bool_], n: int) -> NDArray[np.float32]:
    m = mask.copy()
    for _ in range(n):
        p = np.pad(m, 1)
        m = m | p[:-2, 1:-1] | p[2:, 1:-1] | p[1:-1, :-2] | p[1:-1, 2:]
    return m.astype(np.float32)


# --- stamping -------------------------------------------------------------------------------


def _pil(rgba: Rgba) -> Any:
    from PIL import Image

    return Image.fromarray(np.round(np.clip(rgba, 0, 1) * 255).astype(np.uint8), "RGBA")


def _arr(img: Any) -> Rgba:
    return np.asarray(img, dtype=np.float32) / 255.0


def stamp(
    canvas: Rgba,
    sprite: Rgba,
    x: float,
    y: float,
    length: float,
    angle_deg: float,
    shade: float = 1.0,
    occl: NDArray[np.float32] | None = None,
) -> None:
    """Draw ``sprite`` over ``canvas`` (premultiplied over): its base (bottom centre) at
    pixel (x, y), its long axis ``length`` px long pointing ``angle_deg`` (0 up,
    clockwise), its colour times ``shade``; ``occl`` gets ``shade`` where it is drawn."""
    from PIL import Image

    sh, sw = sprite.shape[:2]
    scale = length / max(sh, 1)
    w, h = max(1, round(sw * scale)), max(1, round(sh * scale))
    img = _pil(sprite).resize((w, h), Image.Resampling.LANCZOS)
    # Pad to a square centred on the base so a rotation turns about the base.
    side = 2 * max(w, h)
    sq = Image.new("RGBA", (side, side), (0, 0, 0, 0))
    sq.paste(img, (side // 2 - w // 2, side // 2 - h))
    rot = _arr(sq.rotate(-angle_deg, resample=Image.Resampling.BICUBIC))
    x0, y0 = round(x) - side // 2, round(y) - side // 2
    ch, cw = canvas.shape[:2]
    a0, b0 = max(0, -y0), max(0, -x0)
    a1, b1 = min(side, ch - y0), min(side, cw - x0)
    if a1 <= a0 or b1 <= b0:
        return
    src = rot[a0:a1, b0:b1]
    dst = canvas[y0 + a0 : y0 + a1, x0 + b0 : x0 + b1]
    a = src[..., 3:4]
    dst[..., :3] = src[..., :3] * shade * a + dst[..., :3] * (1 - a)
    dst[..., 3:4] = a + dst[..., 3:4] * (1 - a)
    if occl is not None:
        o = occl[y0 + a0 : y0 + a1, x0 + b0 : x0 + b1]
        o[:] = shade * a[..., 0] + o * (1 - a[..., 0])


def _line(
    canvas: Rgba,
    p0: tuple[float, float],
    p1: tuple[float, float],
    width: float,
    rgb: tuple[float, float, float],
) -> None:
    from PIL import Image, ImageDraw

    layer = Image.new("RGBA", (canvas.shape[1], canvas.shape[0]), (0, 0, 0, 0))
    ImageDraw.Draw(layer).line(
        [p0, p1], fill=(*(round(c * 255) for c in rgb), 255), width=max(1, round(width))
    )
    src = _arr(layer)
    a = src[..., 3:4]
    canvas[..., :3] = src[..., :3] * a + canvas[..., :3] * (1 - a)
    canvas[..., 3:4] = a + canvas[..., 3:4] * (1 - a)


def unpremultiply(rgba: Rgba) -> Rgba:
    """A canvas (colour times alpha) as a sprite (plain colour)."""
    a = rgba[..., 3:4]
    return np.dstack([np.where(a > 1e-3, rgba[..., :3] / np.maximum(a, 1e-3), 0), a]).astype(
        np.float32
    )


def upright(card: Rgba) -> Rgba:
    """A branch card that grows left to right turned to grow upwards, as ``stamp`` wants."""
    return np.ascontiguousarray(np.rot90(card))


def _blank(px: int = CELL) -> tuple[Rgba, NDArray[np.float32]]:
    return np.zeros((px, px, 4), np.float32), np.ones((px, px), np.float32)


TWIG_RGB = (0.22, 0.16, 0.11)
BIG = 2 * CELL  # crowns are drawn on a larger canvas and fitted into a cell


def fit(
    canvas: Rgba, occl: NDArray[np.float32], cx: float, base: float | None
) -> tuple[Rgba, NDArray[np.float32]]:
    """The drawn part of a canvas scaled to fill a cell: symmetric about column ``cx``,
    from the topmost drawn row down to row ``base`` (the trunk's foot) or, without a base,
    symmetric about the canvas centre in both directions (a view from above)."""
    from PIL import Image

    drawn = canvas[..., 3] > 0.02
    rows = np.nonzero(drawn.any(axis=1))[0]
    cols = np.nonzero(drawn.any(axis=0))[0]
    half_w = max(cx - cols[0], cols[-1] + 1 - cx)
    if base is None:
        cy = canvas.shape[0] / 2
        half = max(half_w, cy - rows[0], rows[-1] + 1 - cy)
        box = (cx - half, cy - half, cx + half, cy + half)
    else:
        box = (cx - half_w, float(rows[0]), cx + half_w, base)
    px = (INNER, INNER)
    out = [
        np.asarray(
            Image.fromarray(np.round(canvas[..., i] * 255).astype(np.uint8), "L").resize(
                px, Image.Resampling.LANCZOS, box=box
            ),
            dtype=np.float32,
        )
        / 255.0
        for i in range(4)
    ]
    o = (
        np.asarray(
            Image.fromarray(np.round(occl * 255).astype(np.uint8), "L").resize(
                px, Image.Resampling.BILINEAR, box=box
            ),
            dtype=np.float32,
        )
        / 255.0
    )
    return np.dstack(out), o


def spruce_card(
    twigs: Sequence[Rgba], rng: np.random.Generator
) -> tuple[Rgba, NDArray[np.float32]]:
    """A spruce branch seen from above, from the trunk (left edge) to the tip (right edge),
    the cell's height as wide as the branch."""
    c, o = _blank(BIG)
    mid = BIG / 2
    _line(c, (0, mid), (BIG * 0.97, mid), BIG * 0.01, TWIG_RGB)
    for i in range(80):
        t = rng.uniform(0.0, 0.93)
        side = 1 if i % 2 else -1
        reach = ((1 - t) * 0.26 + 0.07) * BIG  # long side shoots near the trunk
        ang = 90 - side * rng.uniform(30, 70)  # out to the side and towards the tip
        stamp(
            c,
            twigs[int(rng.integers(len(twigs)))],
            t * BIG,
            mid,
            reach,
            ang,
            rng.uniform(0.75, 1.05),
            o,
        )
    for _ in range(5):  # the leading shoot
        stamp(
            c,
            twigs[0],
            BIG * rng.uniform(0.82, 0.9),
            mid,
            BIG * 0.14,
            90 + rng.uniform(-12, 12),
            1.05,
            o,
        )
    # Keep the branch's full length: from the trunk (x = 0) to the tip.
    drawn = c[..., 3] > 0.02
    rows = np.nonzero(drawn.any(axis=1))[0]
    half = max(mid - rows[0], rows[-1] + 1 - mid)
    cols = np.nonzero(drawn.any(axis=0))[0]
    tip = float(cols[-1] + 1)
    sq = max(tip, 2 * half)
    pad = np.zeros((int(sq), int(sq), 4), np.float32)
    po = np.ones((int(sq), int(sq)), np.float32)
    r0 = int(mid - sq / 2)
    lo, hi = max(r0, 0), min(int(r0 + sq), BIG)
    w = min(int(sq), BIG)
    pad[lo - r0 : hi - r0, :w] = c[lo:hi, :w]
    po[lo - r0 : hi - r0, :w] = o[lo:hi, :w]
    return fit(pad, po, sq / 2, None)


def tuft_card(
    twigs: Sequence[Rgba], rng: np.random.Generator, n: int
) -> tuple[Rgba, NDArray[np.float32]]:
    """A pine needle tuft: shoots fanning up and out from the bottom centre."""
    c, o = _blank(BIG)
    for i in range(n):
        ang = rng.uniform(-70, 70)
        stamp(
            c,
            twigs[int(rng.integers(len(twigs)))],
            BIG / 2 + rng.uniform(-20, 20),
            BIG * 0.9,
            rng.uniform(0.25, 0.45) * BIG,
            ang,
            0.7 + 0.35 * (1 - abs(ang) / 90) * (i / n),
            o,
        )
    return fit(c, o, BIG / 2, BIG * 0.9)


def leaf_card(leaves: Sequence[Rgba], rng: np.random.Generator) -> tuple[Rgba, NDArray[np.float32]]:
    """A cluster of leaves on thin twigs, roughly round, the twigs meeting at the bottom."""
    c, o = _blank(BIG)
    centre = (BIG / 2, BIG / 2)
    for _ in range(9):
        ang = rng.uniform(0, 2 * math.pi)
        end = (centre[0] + math.cos(ang) * BIG * 0.25, centre[1] + math.sin(ang) * BIG * 0.25)
        _line(c, (BIG / 2, BIG * 0.8), end, 3, TWIG_RGB)
    for i in range(110):
        rad = BIG * 0.27 * math.sqrt(rng.uniform(0, 1))
        ang = rng.uniform(0, 2 * math.pi)
        x, y = centre[0] + rad * math.cos(ang), centre[1] + rad * math.sin(ang)
        stamp(
            c,
            leaves[int(rng.integers(len(leaves)))],
            x,
            y,
            rng.uniform(0.09, 0.13) * BIG,
            rng.uniform(0, 360),
            0.65 + 0.4 * i / 110,
            o,
        )
    return fit(c, o, BIG / 2, BIG * 0.8)


# --- whole crowns (impostors) ----------------------------------------------------------------


def side_view(
    species: str, card: Rgba, rng: np.random.Generator
) -> tuple[Rgba, NDArray[np.float32]]:
    """The whole tree from the side, fitted to a cell: the trunk's foot at the bottom
    centre, the top at the top edge, the crown's widest part at the side edges."""
    c, o = _blank(BIG)
    mid, foot = BIG / 2, BIG * 0.95
    tall = BIG * 0.85  # the tree's height on the canvas
    if species == "spruce":
        _line(c, (mid, foot), (mid, foot - tall), BIG * 0.012, TWIG_RGB)
        levels = 40
        for i in range(levels):
            t = i / (levels - 1)  # 0 at the crown's foot, 1 at the top
            y = foot - tall * (0.05 + 0.93 * t)
            length = (1 - t) * tall * 0.2 + 8
            for side in (-1, 1):
                stamp(
                    c,
                    card,
                    mid,
                    y,
                    length,
                    side * rng.uniform(95, 115),
                    0.55 + 0.5 * t + rng.uniform(-0.06, 0.06),
                    o,
                )
    elif species == "pine":
        _line(c, (mid, foot), (mid, foot - tall * 0.8), BIG * 0.014, (0.45, 0.28, 0.17))
        for i in range(160):
            ang = rng.uniform(0, 2 * math.pi)
            rad = math.sqrt(rng.uniform(0, 1))
            x = mid + math.cos(ang) * rad * tall * 0.2
            y = foot - tall * 0.8 + math.sin(ang) * rad * tall * 0.13
            stamp(
                c,
                card,
                x,
                y,
                tall * rng.uniform(0.07, 0.11),
                rng.uniform(-60, 60),
                0.6 + 0.45 * i / 160,
                o,
            )
    else:
        _line(c, (mid, foot), (mid, foot - tall * 0.7), BIG * 0.014, (0.78, 0.76, 0.7))
        for i in range(160):
            ang = rng.uniform(0, 2 * math.pi)
            rad = math.sqrt(rng.uniform(0, 1))
            x = mid + math.cos(ang) * rad * tall * 0.2
            y = foot - tall * 0.62 + math.sin(ang) * rad * tall * 0.27
            stamp(
                c,
                card,
                x,
                y,
                tall * rng.uniform(0.08, 0.12),
                rng.uniform(-40, 40),
                0.6 + 0.45 * i / 160,
                o,
            )
    return fit(c, o, mid, foot)


def top_view(
    species: str, card: Rgba, rng: np.random.Generator
) -> tuple[Rgba, NDArray[np.float32]]:
    """The crown from straight above, fitted to a cell; the brightest part is the top."""
    c, o = _blank(BIG)
    mid = BIG / 2
    if species == "spruce":
        whorls = 14
        for i in range(whorls):  # the lowest whorl first, it is the widest and darkest
            t = i / (whorls - 1)
            length = (1 - t) * BIG * 0.4 + 8
            for k in range(7):
                ang = (k / 7) * 360 + i * 137.5 + rng.uniform(-10, 10)
                stamp(c, card, mid, mid, length, ang, 0.5 + 0.55 * t, o)
    else:
        n = 90 if species == "pine" else 140
        for i in range(n):
            rad = math.sqrt(rng.uniform(0, 1)) * BIG * 0.3 * (1 - 0.4 * i / n)
            ang = rng.uniform(0, 2 * math.pi)
            x, y = mid + rad * math.cos(ang), mid + rad * math.sin(ang)
            stamp(
                c,
                card,
                x,
                y,
                BIG * rng.uniform(0.1, 0.15),
                rng.uniform(0, 360),
                0.5 + 0.55 * i / n,
                o,
            )
    return fit(c, o, mid, None)


# --- maps -------------------------------------------------------------------------------------


def normal_from_colour(rgba: Rgba, strength: float = 2.0) -> NDArray[np.float32]:
    """An OpenGL-convention normal map from the picture's brightness (bright = raised)."""
    lum = (rgba[..., :3] @ np.array([0.3, 0.59, 0.11], np.float32)) * rgba[..., 3]
    gy, gx = np.gradient(lum)
    n = np.dstack([-gx * strength, gy * strength, np.ones_like(lum)])
    n /= np.linalg.norm(n, axis=2, keepdims=True)
    return (n * 0.5 + 0.5).astype(np.float32)


def birch(diff: NDArray[np.float32]) -> NDArray[np.float32]:
    """A dark bark texture lightened to a birch's white with dark streaks."""
    lum = diff @ np.array([0.3, 0.59, 0.11], np.float32)
    lum = (lum - lum.min()) / max(float(np.ptp(lum)), 1e-6)
    white = np.array([0.86, 0.84, 0.79], np.float32)
    dark = np.array([0.18, 0.16, 0.14], np.float32)
    t = np.clip((lum - 0.25) * 2.2, 0, 1)[..., None]
    return (dark * (1 - t) + white * t).astype(np.float32)


def resize(img: NDArray[np.float32], px: int) -> NDArray[np.float32]:
    from PIL import Image

    chans = [
        np.asarray(
            Image.fromarray(np.round(img[..., i] * 255).astype(np.uint8), "L").resize(
                (px, px), Image.Resampling.LANCZOS
            ),
            dtype=np.float32,
        )
        / 255.0
        for i in range(img.shape[2])
    ]
    return np.dstack(chans)


def build(src: Source) -> tuple[dict[str, NDArray[np.float32]], dict[str, Any]]:
    rng = np.random.default_rng(SEED)
    color = np.zeros((ATLAS_PX, ATLAS_PX, 4), np.float32)
    normal = np.tile(np.array([0.5, 0.5, 1.0], np.float32), (ATLAS_PX, ATLAS_PX, 1))
    # Occlusion 1, roughness 0.9, metallic 0 by default.
    orm = np.tile(np.array([1.0, 0.9, 0.0], np.float32), (ATLAS_PX, ATLAS_PX, 1))
    regions: dict[str, list[float]] = {}

    def put(
        name: str,
        row: str,
        col: int,
        rgba: Rgba,
        occl: NDArray[np.float32] | None,
        nor: NDArray[np.float32] | None = None,
        arm: NDArray[np.float32] | None = None,
    ) -> None:
        r0, c0 = ROWS.index(row) * CELL + PAD, col * CELL + PAD
        sl = (slice(r0, r0 + INNER), slice(c0, c0 + INNER))
        a = rgba[..., 3:4]
        # Colour where the picture is transparent is the nearby colour, so the edges of a
        # cut-out do not fade to black when the renderer blends mip levels.
        rgb = np.where(a > 0.01, rgba[..., :3] / np.maximum(a, 1e-3), 0)
        fill = np.where(a > 0.01, rgba[..., :3] / np.maximum(a, 1e-3), np.nan)
        mean = np.nanmean(fill.reshape(-1, 3), axis=0) if np.any(a > 0.01) else np.zeros(3)
        rgb = np.where(a > 0.01, rgb, mean)
        color[sl] = np.dstack([np.clip(rgb, 0, 1), a[..., 0]])
        normal[sl] = nor if nor is not None else normal_from_colour(rgba)
        if arm is not None:
            orm[sl] = arm
        elif occl is not None:
            orm[sl] = np.dstack([occl, np.full_like(occl, 0.85), np.zeros_like(occl)])
        # u0, v0 (top left), u1, v1 (bottom right) in glTF texture coordinates.
        regions[name] = [
            c0 / ATLAS_PX,
            r0 / ATLAS_PX,
            (c0 + INNER) / ATLAS_PX,
            (r0 + INNER) / ATLAS_PX,
        ]

    sheets = {
        key: sprites(
            src.picture(asset, d, "RGB"),
            src.picture(asset, a, "L"),
            0.002 if key == "leaves" else 0.004,
        )
        for key, (asset, d, a) in MAPS.items()
    }
    # The pine sheet's two needle shoots; the rest are loose needles. The second shoot
    # grows to the right and is turned upright.
    sheets["pine_twig"] = [sheets["pine_twig"][0], upright(sheets["pine_twig"][1])]
    cards = {
        "spruce": spruce_card(sheets["spruce_twig"], rng),
        "pine": tuft_card(sheets["pine_twig"], rng, 30),
        "deciduous": leaf_card(sheets["leaves"], rng),
    }
    for col, sp in enumerate(SPECIES):
        card, occl = cards[sp]
        put(f"{sp}_card", "card", col, card, occl)
        sprite = unpremultiply(upright(card) if sp == "spruce" else card)
        put(f"{sp}_side", "side", col, *side_view(sp, sprite, rng))
        put(f"{sp}_top", "top", col, *top_view(sp, sprite, rng))
        asset, d, n, arm = BARK[sp]
        diff = src.picture(asset, d, "RGB")
        if sp == "deciduous":
            diff = birch(diff)
        put(
            f"{sp}_bark",
            "bark",
            col,
            np.dstack([resize(diff, INNER), np.ones((INNER, INNER, 1), np.float32)]),
            None,
            resize(src.picture(asset, n, "RGB"), INNER),
            resize(src.picture(asset, arm, "RGB"), INNER),
        )
    maps = {"trees_color.png": color, "trees_normal.png": normal, "trees_orm.png": orm}
    meta = {
        "atlas_px": ATLAS_PX,
        "regions": regions,
        "sources": src.files,
        "licence": "CC0-1.0 (Poly Haven, https://polyhaven.com/license)",
    }
    return maps, meta


def save(maps: dict[str, NDArray[np.float32]], meta: dict[str, Any], dest: Path) -> None:
    from PIL import Image

    dest.mkdir(parents=True, exist_ok=True)
    for name, arr in maps.items():
        mode = "RGBA" if arr.shape[2] == 4 else "RGB"
        Image.fromarray(np.round(np.clip(arr, 0, 1) * 255).astype(np.uint8), mode).save(
            dest / name, optimize=False
        )
    (dest / "atlas.json").write_text(
        json.dumps(meta, indent=1, sort_keys=True) + "\n", encoding="utf-8"
    )


def main(argv: Sequence[str] | None = None) -> int:
    args = list(sys.argv[1:] if argv is None else argv)
    if args:
        print("tree-pictures-fetch tar inga inställningar", file=sys.stderr)
        return 2
    dest = ROOT / "build" / "tree_models"
    try:
        print("==> trädbilder (Poly Haven, CC0)")
        maps, meta = build(Source())
    except FetchError as exc:
        print(str(exc), file=sys.stderr)
        return exc.code
    save(maps, meta, dest)
    print(f"Klart: {dest} ({len(meta['regions'])} bilder i en atlas på {ATLAS_PX} px)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
