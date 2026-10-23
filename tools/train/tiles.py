"""The training tiles: the bank's pictures cut into the tiles the drone looks at.

The drone looks at a 12 MP still in 640 px tiles (``sd_detect.tiling``), so the model is
trained on tiles of the same size and overlap (the grid is laid out a little
differently: a fixed step and a last tile at the edge). Each picture of the split's
``train`` (or ``val``) part is cut into ``tile_px`` tiles, and each tile is saved once
as a JPEG with its person boxes. A person is kept in a tile when at least half of their
box is in it. Every tile with a person is kept; of the empty ones (grass, rock, forest)
a share ``empty_share`` is kept so the model also learns what is not a person.

The tiles go to ``<out>/tiles/<part>/`` with ``tiles.json``; a finished cache is reused
(its ``key`` says what it was cut from). Nothing here is ever in git: the tiles are
cut from the downloaded datasets on the owner's computer.
"""

from __future__ import annotations

import hashlib
import json
import random
from collections.abc import Iterable, Sequence
from dataclasses import dataclass
from pathlib import Path

import numpy as np
from bench.common import inside, safe_relpath
from bench.datasets import Image, image_root, load_register, read_index
from bench.splits import load_split
from PIL import Image as PILImage

TILE_PX = 640
OVERLAP = 0.15
MIN_VISIBLE = 0.5
EMPTY_SHARE = 0.25
CACHE_SCHEMA = "train.tiles/v1"


@dataclass(frozen=True)
class Tile:
    file: str  # relative to the cache folder
    boxes: tuple[tuple[float, float, float, float], ...]  # corners, tile pixels
    source: str  # picture id


def starts(side: int, tile: int, overlap: float) -> list[int]:
    if side <= tile:
        return [0]
    step = max(1, round(tile * (1 - overlap)))
    out = list(range(0, side - tile, step))
    out.append(side - tile)
    return sorted(set(out))


def cut(
    width: int,
    height: int,
    boxes: Sequence[tuple[float, float, float, float]],
    tile: int = TILE_PX,
    overlap: float = OVERLAP,
) -> list[tuple[tuple[int, int, int, int], list[tuple[float, float, float, float]]]]:
    """Tiles ``(x0, y0, x1, y1)`` of a picture and the boxes kept in each."""
    out = []
    for y in starts(height, tile, overlap):
        for x in starts(width, tile, overlap):
            x1, y1 = min(x + tile, width), min(y + tile, height)
            kept = []
            for b in boxes:
                cx0, cy0 = max(b[0], x), max(b[1], y)
                cx1, cy1 = min(b[2], x1), min(b[3], y1)
                if cx1 <= cx0 or cy1 <= cy0:
                    continue
                area = (b[2] - b[0]) * (b[3] - b[1])
                if area > 0 and (cx1 - cx0) * (cy1 - cy0) / area >= MIN_VISIBLE:
                    kept.append((cx0 - x, cy0 - y, cx1 - x, cy1 - y))
            out.append(((x, y, x1, y1), kept))
    return out


def pictures(
    data: Path, part: str, sets: Iterable[str], limit: int = 0
) -> list[tuple[Image, Path]]:
    """The pictures of ``part`` (train, val; ``pretrain``: the synthetic sets) of every
    set in ``sets`` that is fetched, with the folder their files are in."""
    register = load_register()
    split = load_split()
    out: list[tuple[Image, Path]] = []
    for name in sets:
        ds = register[name]
        if part == "pretrain":
            if "pretrain" not in ds.use:
                continue
            rows = read_index(data, name)
        else:
            if "train" not in ds.use:
                continue  # measure-only sets are never trained on
            rows = read_index(data, name)
            split.check(name, [(im.local_id, im.seq) for im in rows])
            rows = [im for im in rows if split.part_of(name, im.seq) == part]
        root = image_root(data, name)
        out.extend((im, root) for im in sorted(rows, key=lambda r: r.image_id))
    if limit:
        out = out[:limit]
    return out


def cache_key(items: Sequence[tuple[Image, Path]], tile: int, empty_share: float) -> str:
    digest = hashlib.sha256(f"{CACHE_SCHEMA}|{tile}|{empty_share}".encode())
    for im, _ in items:
        digest.update(f"|{im.image_id}:{len(im.boxes)}".encode())
    return digest.hexdigest()[:16]


def build_cache(
    items: Sequence[tuple[Image, Path]],
    folder: Path,
    tile: int = TILE_PX,
    empty_share: float = EMPTY_SHARE,
    seed: int = 0,
    progress: bool = True,
) -> list[Tile]:
    """Cut every picture once; a cache with the same key is read back instead."""
    key = cache_key(items, tile, empty_share)
    listing = folder / "tiles.json"
    if listing.is_file():
        body = json.loads(listing.read_text(encoding="utf-8"))
        if body.get("schema") == CACHE_SCHEMA and body.get("key") == key:
            return [
                Tile(t["file"], tuple(map(tuple, t["boxes"])), t["source"]) for t in body["tiles"]
            ]
    folder.mkdir(parents=True, exist_ok=True)
    rng = random.Random(seed)  # noqa: S311 - which empty tiles to keep, not a secret
    tiles: list[Tile] = []
    for n, (im, root) in enumerate(items, 1):
        with PILImage.open(inside(root, safe_relpath(im.file))) as pic:
            rgb = pic.convert("RGB")
        for (x0, y0, x1, y1), kept in cut(im.width, im.height, im.boxes, tile):
            if not kept and rng.random() >= empty_share:
                continue
            name = f"{len(tiles):07d}.jpg"
            rgb.crop((x0, y0, x1, y1)).save(folder / name, quality=92)
            tiles.append(Tile(name, tuple(kept), im.image_id))
        if progress and (n % 50 == 0 or n == len(items)):
            print(f"  skär bilder i rutor: {n}/{len(items)}", flush=True)
    body = {
        "schema": CACHE_SCHEMA,
        "key": key,
        "tiles": [{"file": t.file, "boxes": t.boxes, "source": t.source} for t in tiles],
    }
    tmp = listing.with_name(".tiles.json.tmp")
    tmp.write_text(json.dumps(body) + "\n", encoding="utf-8")
    tmp.replace(listing)
    return tiles


def load_tile(
    folder: Path, t: Tile, tile: int, pad: int, rng: random.Random | None = None
) -> tuple[np.ndarray, np.ndarray]:
    """A tile as the family's input (BGR, 0-255, ``[3, tile, tile]``, padded right and
    bottom) and its boxes; with ``rng``, flipped and turned at random (seen from above,
    every way up is a way a person lies)."""
    with PILImage.open(folder / t.file) as pic:
        rgb = np.asarray(pic.convert("RGB"))
    boxes = np.asarray(t.boxes, dtype=np.float32).reshape(-1, 4)
    h, w = rgb.shape[:2]
    if rng is not None:
        if rng.random() < 0.5:
            rgb = rgb[:, ::-1]
            boxes[:, [0, 2]] = w - boxes[:, [2, 0]]
        if rng.random() < 0.5:
            rgb = rgb[::-1]
            boxes[:, [1, 3]] = h - boxes[:, [3, 1]]
        if rng.random() < 0.5 and h == w:
            rgb = rgb.transpose(1, 0, 2)
            boxes = boxes[:, [1, 0, 3, 2]]
        gain = rng.uniform(0.75, 1.25)
        rgb = np.clip(rgb.astype(np.float32) * gain, 0, 255)
    canvas = np.full((tile, tile, 3), pad, dtype=np.float32)
    canvas[:h, :w] = rgb[:, :, ::-1]
    return np.ascontiguousarray(canvas.transpose(2, 0, 1)), boxes
