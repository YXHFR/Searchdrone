"""The training tiles: the bank's pictures cut into the tiles the drone looks at.

The drone looks at a 12 MP still in 640 px tiles (``sd_detect.tiling``), so the model is
trained on tiles of the same size and overlap (the grid is laid out a little
differently: a fixed step and a last tile at the edge). Each picture of the split's
``train`` (or ``val``) part is cut into ``tile_px`` tiles, and each tile is saved once
as a JPEG with its person boxes. A person is kept in a tile when at least half of their
box is in it. Every tile with a person is kept; of the empty ones (grass, rock, forest)
a share ``empty_share`` is kept so the model also learns what is not a person.

With ``upscale`` above 1 (``just train-detector upscale=2``) each tile covers
``tile_px / upscale`` picture pixels and is enlarged ``upscale`` times, bilinear, to
``tile_px``, with its boxes: a 15 px person is trained on as 30 px. The drone does the
same before the model sees a tile (``sd_detect.tiling.enlarge``, read from the model's
manifest); ``enlarge`` here is the same resize, written out because the training
environment has no ``sd_detect`` (``tools/tests/test_detector_upscale.py`` keeps the
two equal).

With ``scales`` (``just train-detector scales=native,wide,tele``) each training picture
is first resampled so its people are as many pixels as our cameras will see them, then
cut as above. A picture's own scale, in pixels per metre on the ground, is the
register's ``px_per_m`` for its set when the set has one; otherwise it is read off the
person boxes (``person_px_per_m``: the median longest side over ``PERSON_M``), from the
picture's own boxes when it has at least ``MIN_OWN_BOXES``, else from its sequence's,
else from its set's. ``wide`` and ``tele`` are the cameras in ``CAMERAS``: each picture
gets one target drawn (fixed by its id) between the camera's pixels per metre at the top
and the bottom of ``SEARCH_HEIGHTS_M``, so the set as a whole covers the search
heights. A number is a fixed target in pixels per metre; ``native`` keeps the picture as
it is. A picture is never enlarged more than ``max_up`` times (an enlarged person is
blurred, not sharper) and two scales within ``MERGE`` of each other are cut once. Only
the training part is resampled; the val part stays as the locked test part is, so the
best epoch is picked on pictures like the ones the model is measured on.

A set's ``ignore`` boxes (things that hold people who are not boxed as persons, such as
UIT-Drone's motorcycles with their riders) are never shown to the model: a tile that
overlaps one is left out, so the model is never taught that a rider is not a person.

The tiles go to ``<out>/tiles/<part>/`` with ``tiles.json``; a finished cache is reused
(its ``key`` says what it was cut from). Nothing here is ever in git: the tiles are
cut from the downloaded datasets on the owner's computer.
"""

from __future__ import annotations

import hashlib
import json
import math
import random
import statistics
from collections import defaultdict
from collections.abc import Iterable, Mapping, Sequence
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
MAX_UPSCALE = 4  # as sd_detect.tiling

# --- scale-matched training ------------------------------------------------------------
# A person's box seen from above, metres along its longest side: the owner's figures of
# 2026-11-03 (about 33 px at the wide camera's 71 px/m at 50 m, about 100 px at the
# tele's 230 px/m) give 0.47 m.
PERSON_M = 0.47
SEARCH_HEIGHTS_M = (30.0, 60.0)  # the search heights the camera targets are drawn between
# our cameras: picture width in pixels and horizontal field of view in degrees
# (Camera Module 3 at full resolution; Pi HQ with a lens of about 20 degrees)
CAMERAS: dict[str, tuple[int, float]] = {"wide": (4608, 66.0), "tele": (4056, 20.0)}
NATIVE = "native"
MAX_UP = 2.0  # by default a picture is enlarged at most this much to reach a target
MIN_FACTOR = 0.05
MERGE = 0.15  # two factors within 15 % of each other are cut once
MIN_OWN_BOXES = 3  # fewer boxes than this: the picture's sequence gives its scale
PX_PER_M_RANGE = (1.0, 5000.0)


@dataclass(frozen=True)
class Tile:
    file: str  # relative to the cache folder
    boxes: tuple[tuple[float, float, float, float], ...]  # corners, tile pixels
    source: str  # picture id
    scale: float = 1.0  # how many times the picture was resampled before it was cut


Box = tuple[float, float, float, float]


def camera_px_per_m(camera: str, height_m: float) -> float:
    """Pixels per metre on the ground straight below ``camera`` at ``height_m``."""
    width, fov = CAMERAS[camera]
    return width / (2 * height_m * math.tan(math.radians(fov) / 2))


def parse_scales(text: str) -> tuple[str, ...]:
    """``native,wide,tele,150`` checked (``ValueError`` names the bad part); empty is
    ``native`` alone. A number is pixels per metre."""
    out: list[str] = []
    for part in (p.strip() for p in text.split(",")):
        if not part:
            continue
        if part not in (NATIVE, *CAMERAS):
            try:
                value = float(part)
            except ValueError:
                raise ValueError(
                    f"{part!r}: a scale is {NATIVE}, {', '.join(CAMERAS)} or pixels per metre"
                ) from None
            if not (math.isfinite(value) and PX_PER_M_RANGE[0] <= value <= PX_PER_M_RANGE[1]):
                raise ValueError(f"{part!r}: pixels per metre must be within {PX_PER_M_RANGE}")
            part = f"{value:g}"
        if part not in out:
            out.append(part)
    return tuple(out) or (NATIVE,)


def person_px_per_m(boxes: Sequence[Box]) -> float | None:
    """The ground scale the person boxes suggest: their median longest side over
    ``PERSON_M`` (``None``: no boxes)."""
    sides = [max(b[2] - b[0], b[3] - b[1]) for b in boxes]
    return statistics.median(sides) / PERSON_M if sides else None


def target_px_per_m(scale: str, image_id: str) -> float | None:
    """The pixels per metre ``scale`` asks of a picture (``None``: ``native``). A camera's
    target is drawn evenly in log scale between its values at the two search heights,
    fixed by the picture's id, so a cache is cut the same way every time."""
    if scale == NATIVE:
        return None
    if scale not in CAMERAS:
        return float(scale)
    low, high = (camera_px_per_m(scale, h) for h in sorted(SEARCH_HEIGHTS_M, reverse=True))
    u = random.Random(f"{scale}|{image_id}").random()  # noqa: S311 - a fixed draw
    return math.exp(math.log(low) + u * (math.log(high) - math.log(low)))


def factors(
    source: float | None, scales: Sequence[str], image_id: str, max_up: float = MAX_UP
) -> list[tuple[str, float]]:
    """``(scale, factor)`` to resample a picture of ``source`` pixels per metre by, one per
    scale: never more than ``max_up`` times larger, and a factor within ``MERGE`` of an
    earlier one is left out. Without a ``source`` only ``native`` can be cut."""
    out: list[tuple[str, float]] = []
    for scale in scales:
        target = target_px_per_m(scale, image_id)
        if target is None:
            f = 1.0
        elif source is None:
            continue
        else:
            f = max(MIN_FACTOR, min(max_up, target / source))
        if any(abs(f / g - 1) < MERGE for _, g in out):
            continue
        out.append((scale, f))
    return out


def source_scales(
    items: Sequence[tuple[Image, Path]], known: Mapping[str, float] | None = None
) -> dict[str, float | None]:
    """Each picture's ground scale in pixels per metre: its set's ``known`` value, else
    read off its own boxes (at least ``MIN_OWN_BOXES``), its sequence's or its set's."""
    known = known or {}
    by_seq: dict[tuple[str, str], list[Box]] = defaultdict(list)
    by_set: dict[str, list[Box]] = defaultdict(list)
    for im, _ in items:
        by_seq[(im.dataset, im.seq)].extend(im.boxes)
        by_set[im.dataset].extend(im.boxes)
    seq_scale = {k: person_px_per_m(v) for k, v in by_seq.items()}
    set_scale = {k: person_px_per_m(v) for k, v in by_set.items()}
    out: dict[str, float | None] = {}
    for im, _ in items:
        if im.dataset in known:
            out[im.image_id] = known[im.dataset]
        elif len(im.boxes) >= MIN_OWN_BOXES:
            out[im.image_id] = person_px_per_m(im.boxes)
        else:
            out[im.image_id] = seq_scale[(im.dataset, im.seq)] or set_scale[im.dataset]
    return out


def resample(pic: PILImage.Image, factor: float) -> PILImage.Image:
    """``pic`` resampled ``factor`` times: Lanczos when it shrinks (as a coarser sensor
    would see it), bicubic when it grows."""
    if factor == 1.0:
        return pic
    size = (max(1, round(pic.width * factor)), max(1, round(pic.height * factor)))
    how = PILImage.Resampling.LANCZOS if factor < 1 else PILImage.Resampling.BICUBIC
    return pic.resize(size, how)


def _scaled(boxes: Iterable[Box], f: float) -> list[Box]:
    return [(b[0] * f, b[1] * f, b[2] * f, b[3] * f) for b in boxes]


def starts(side: int, tile: int, overlap: float) -> list[int]:
    if side <= tile:
        return [0]
    step = max(1, round(tile * (1 - overlap)))
    out = list(range(0, side - tile, step))
    out.append(side - tile)
    return sorted(set(out))


def check_upscale(tile: int, upscale: int) -> int:
    """The picture pixels one tile side covers, ``tile / upscale`` (``ValueError``: a
    factor that is not a whole number from 1 to ``MAX_UPSCALE`` or does not divide
    ``tile``)."""
    if type(upscale) is not int or not 1 <= upscale <= MAX_UPSCALE:
        raise ValueError(f"upscale must be a whole number from 1 to {MAX_UPSCALE}")
    if tile % upscale:
        raise ValueError("upscale must divide the tile")
    return tile // upscale


def enlarge(rgb: np.ndarray, upscale: int) -> np.ndarray:
    """``rgb`` (``[h, w, 3]``, uint8) enlarged ``upscale`` times, bilinear: the same
    resize as ``sd_detect.tiling.enlarge`` on the drone."""
    if upscale == 1:
        return rgb
    h, w = rgb.shape[:2]
    big = PILImage.fromarray(rgb).resize((w * upscale, h * upscale), PILImage.Resampling.BILINEAR)
    return np.ascontiguousarray(np.asarray(big))


def cut(
    width: int,
    height: int,
    boxes: Sequence[tuple[float, float, float, float]],
    tile: int = TILE_PX,
    overlap: float = OVERLAP,
    upscale: int = 1,
    ignore: Sequence[tuple[float, float, float, float]] = (),
) -> list[tuple[tuple[int, int, int, int], list[tuple[float, float, float, float]]]]:
    """Tiles ``(x0, y0, x1, y1)`` of a picture (picture pixels) and the boxes kept in
    each, in the pixels of the tile enlarged ``upscale`` times. Each tile covers
    ``tile / upscale`` picture pixels, so enlarged it is ``tile`` px (less at an edge).
    A tile that overlaps an ``ignore`` box is left out."""
    span = check_upscale(tile, upscale)
    out = []
    for y in starts(height, span, overlap):
        for x in starts(width, span, overlap):
            x1, y1 = min(x + span, width), min(y + span, height)
            if any(g[0] < x1 and g[2] > x and g[1] < y1 and g[3] > y for g in ignore):
                continue
            kept = []
            for b in boxes:
                cx0, cy0 = max(b[0], x), max(b[1], y)
                cx1, cy1 = min(b[2], x1), min(b[3], y1)
                if cx1 <= cx0 or cy1 <= cy0:
                    continue
                area = (b[2] - b[0]) * (b[3] - b[1])
                if area > 0 and (cx1 - cx0) * (cy1 - cy0) / area >= MIN_VISIBLE:
                    kept.append(
                        (
                            (cx0 - x) * upscale,
                            (cy0 - y) * upscale,
                            (cx1 - x) * upscale,
                            (cy1 - y) * upscale,
                        )
                    )
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


def cache_key(
    items: Sequence[tuple[Image, Path]],
    tile: int,
    empty_share: float,
    upscale: int = 1,
    scales: Sequence[str] = (NATIVE,),
    max_up: float = MAX_UP,
    known: Mapping[str, float] | None = None,
) -> str:
    # upscale=1 and the native scale leave the key as it was, so an older cache is reused
    up = f"|x{upscale}" if upscale != 1 else ""
    if tuple(scales) != (NATIVE,):
        up += f"|s{','.join(scales)}|m{max_up:g}|k{sorted((known or {}).items())}"
    digest = hashlib.sha256(f"{CACHE_SCHEMA}|{tile}|{empty_share}{up}".encode())
    for im, _ in items:
        digest.update(f"|{im.image_id}:{len(im.boxes)}".encode())
        if im.ignore:
            digest.update(f"i{len(im.ignore)}".encode())
    return digest.hexdigest()[:16]


def known_scales(items: Sequence[tuple[Image, Path]]) -> dict[str, float]:
    """The register's ``px_per_m`` for the sets among ``items`` that have one."""
    register = load_register()
    out: dict[str, float] = {}
    for name in sorted({im.dataset for im, _ in items}):
        value = register[name].px_per_m if name in register else None
        if value:
            out[name] = float(value)
    return out


def build_cache(
    items: Sequence[tuple[Image, Path]],
    folder: Path,
    tile: int = TILE_PX,
    empty_share: float = EMPTY_SHARE,
    seed: int = 0,
    progress: bool = True,
    upscale: int = 1,
    scales: Sequence[str] = (NATIVE,),
    max_up: float = MAX_UP,
) -> list[Tile]:
    """Cut every picture once per scale (each tile enlarged ``upscale`` times, as
    ``sd_detect`` enlarges it on the drone); a cache with the same key is read back
    instead."""
    scales = tuple(scales)
    scaled = scales != (NATIVE,)
    known = known_scales(items) if scaled else {}
    key = cache_key(items, tile, empty_share, upscale, scales, max_up, known)
    listing = folder / "tiles.json"
    if listing.is_file():
        body = json.loads(listing.read_text(encoding="utf-8"))
        if body.get("schema") == CACHE_SCHEMA and body.get("key") == key:
            return [
                Tile(
                    t["file"],
                    tuple(map(tuple, t["boxes"])),
                    t["source"],
                    float(t.get("scale", 1.0)),
                )
                for t in body["tiles"]
            ]
    folder.mkdir(parents=True, exist_ok=True)
    rng = random.Random(seed)  # noqa: S311 - which empty tiles to keep, not a secret
    source = source_scales(items, known) if scaled else {}
    tiles: list[Tile] = []
    seen: dict[tuple[str, str], list[float]] = defaultdict(list)  # (set, scale) -> sides
    for n, (im, root) in enumerate(items, 1):
        with PILImage.open(inside(root, safe_relpath(im.file))) as pic:
            rgb = pic.convert("RGB")
        todo = factors(source[im.image_id], scales, im.image_id, max_up) if scaled else []
        for scale, f in todo or [(NATIVE, 1.0)]:
            pic_f = resample(rgb, f)
            boxes, ignore = _scaled(im.boxes, f), _scaled(im.ignore, f)
            seen[(im.dataset, scale)].extend(max(b[2] - b[0], b[3] - b[1]) for b in boxes)
            for (x0, y0, x1, y1), kept in cut(
                pic_f.width, pic_f.height, boxes, tile, upscale=upscale, ignore=ignore
            ):
                if not kept and rng.random() >= empty_share:
                    continue
                name = f"{len(tiles):07d}.jpg"
                crop = np.asarray(pic_f.crop((x0, y0, x1, y1)))
                PILImage.fromarray(enlarge(crop, upscale)).save(folder / name, quality=92)
                tiles.append(Tile(name, tuple(kept), im.image_id, round(f, 4)))
        if progress and (n % 50 == 0 or n == len(items)):
            print(f"  skär bilder i rutor: {n}/{len(items)}", flush=True)
    if progress and scaled:
        for line in scale_report(seen):
            print(line, flush=True)
    body = {
        "schema": CACHE_SCHEMA,
        "key": key,
        "tiles": [
            {"file": t.file, "boxes": t.boxes, "source": t.source}
            | ({"scale": t.scale} if t.scale != 1.0 else {})
            for t in tiles
        ],
    }
    tmp = listing.with_name(".tiles.json.tmp")
    tmp.write_text(json.dumps(body) + "\n", encoding="utf-8")
    tmp.replace(listing)
    return tiles


def scale_report(seen: Mapping[tuple[str, str], Sequence[float]]) -> list[str]:
    """One line per set and scale: how many persons were cut, and their sizes."""
    out = []
    for (name, scale), sides in sorted(seen.items()):
        if not sides:
            out.append(f"  {name} {scale}: inga personer")
            continue
        q = statistics.quantiles(sides, n=4) if len(sides) > 1 else [sides[0]] * 3
        out.append(
            f"  {name} {scale}: {len(sides)} personer, längsta sidan "
            f"{q[0]:.0f}/{q[1]:.0f}/{q[2]:.0f} px (kvartiler)"
        )
    return out


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
