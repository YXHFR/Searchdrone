"""``TiledDetector``: runs a frame detector on overlapping tiles of a large picture.

A person in a 12 MP still is 15-60 px long; a detector that takes 640 x 640 px would see
them as a few pixels if the whole picture were shrunk to fit. So the picture is cut into
tiles of ``tile_px`` with ``overlap`` between neighbours (45 tiles for 4608 x 2592 at
640 px and 15 %), the inner detector runs on each tile at full resolution, and its boxes
are moved back into the picture's pixels. Two steps then remove the doubles the overlap
creates (the idea of SAHI, own code, no extra dependency):

1. seam merge: a box cut off by its tile's inner edge and a box from the neighbouring
   tile that continues it across that edge (they touch or overlap, and they cover the
   same span along the edge) become one box, their union, with the higher score;
2. non-maximum suppression per label (IoU >= ``nms_iou``): a person inside the overlap
   seen whole by both tiles is one box.

With ``upscale`` above 1 each tile covers ``tile_px / upscale`` picture pixels and is
enlarged ``upscale`` times (bilinear) before the inner detector sees it, so a person of
15 px reaches the model as 30 px; the boxes are shrunk back the same way. The model must
have been trained on tiles enlarged the same way (``tools/train/tiles.py``); its manifest
says by how much (``upscale``, 1 when missing). At 2x a 12 MP still takes 170 tiles
instead of 45.

Tiles can be skipped (``skip_mask``), for example those wholly outside the search
frame. ``last_tiles`` and ``last_latency_ms`` say how many tiles the last picture took
and how long it took, for the perception report.

Like every detector it gives hints, never decisions: nothing here reaches the vehicle.
"""

from __future__ import annotations

import math
import time
from collections.abc import Callable
from dataclasses import dataclass
from typing import ClassVar

import numpy as np

from sd_detect.api import Detection, Detector, Frame, SourceKind
from sd_detect.boxes import Boxes, from_bbox, nms, to_bbox

__all__ = [
    "DEFAULT_OVERLAP",
    "DEFAULT_TILE_PX",
    "MAX_UPSCALE",
    "NMS_IOU",
    "SEAM_PX",
    "Tile",
    "TiledDetector",
    "check_upscale",
    "enlarge",
    "tile_grid",
    "tile_to_picture",
]

DEFAULT_TILE_PX = 640
DEFAULT_OVERLAP = 0.15
NMS_IOU = 0.5
MAX_UPSCALE = 4
SEAM_PX = 2.0  # a box edge this close to a tile's inner edge counts as cut off by it
SEAM_SPAN = 0.5  # share of the shorter span along the seam two halves must have in common


@dataclass(frozen=True)
class Tile:
    """One tile in picture pixels: ``x0 <= x < x1``, ``y0 <= y < y1``."""

    x0: int
    y0: int
    x1: int
    y1: int


def _origins(length: int, tile: int, overlap: float) -> list[int]:
    if length <= tile:
        return [0]
    stride = tile * (1.0 - overlap)
    n = math.ceil((length - tile) / stride) + 1
    step = (length - tile) / (n - 1)
    return [round(i * step) for i in range(n)]


def tile_grid(
    width: int, height: int, tile_px: int = DEFAULT_TILE_PX, overlap: float = DEFAULT_OVERLAP
) -> list[Tile]:
    """The tiles covering a ``width`` x ``height`` picture, row by row. Neighbours overlap
    by at least ``overlap`` of a tile; the last tile in a row or column ends at the edge.
    A side shorter than ``tile_px`` is one tile of that side's length."""
    if tile_px < 32:
        raise ValueError("tile_px must be at least 32")
    if not 0.0 <= overlap < 0.9:
        raise ValueError("overlap must be in [0, 0.9)")
    tw, th = min(tile_px, width), min(tile_px, height)
    return [
        Tile(x, y, x + tw, y + th)
        for y in _origins(height, th, overlap)
        for x in _origins(width, tw, overlap)
    ]


def check_upscale(tile_px: int, upscale: int) -> int:
    """The picture pixels one tile side covers, ``tile_px / upscale``. Refuses a factor
    that is not a whole number from 1 to ``MAX_UPSCALE`` or does not divide ``tile_px``."""
    if type(upscale) is not int or not 1 <= upscale <= MAX_UPSCALE:
        raise ValueError(f"upscale must be a whole number from 1 to {MAX_UPSCALE}")
    if tile_px % upscale:
        raise ValueError("upscale must divide tile_px")
    return tile_px // upscale


def tile_to_picture(xyxy: Boxes, tile: Tile, upscale: int = 1) -> Boxes:
    """Boxes ``(x0, y0, x1, y1)`` in the (enlarged) tile's pixels, in the picture's."""
    offset = np.array([tile.x0, tile.y0, tile.x0, tile.y0], dtype=np.float64)
    return np.asarray(xyxy, dtype=np.float64) / upscale + offset


def enlarge(rgb: np.ndarray, upscale: int) -> np.ndarray:
    """``rgb`` (``[h, w, 3]``, uint8) enlarged ``upscale`` times, bilinear, as the
    training tiles are (``tools/train/tiles.py``)."""
    if upscale == 1:
        return rgb
    from PIL import Image

    h, w = rgb.shape[:2]
    big = Image.fromarray(rgb).resize((w * upscale, h * upscale), Image.Resampling.BILINEAR)
    return np.ascontiguousarray(np.asarray(big))


@dataclass
class _Box:
    label: str
    score: float
    xyxy: Boxes  # shape (4,)
    tile: int
    cut: tuple[bool, bool, bool, bool]  # cut off by the tile's left, top, right, bottom edge
    source: SourceKind | None = None


def _cut(xyxy: Boxes, tile: Tile, width: int, height: int) -> tuple[bool, bool, bool, bool]:
    x0, y0, x1, y1 = xyxy
    return (
        tile.x0 > 0 and x0 <= tile.x0 + SEAM_PX,
        tile.y0 > 0 and y0 <= tile.y0 + SEAM_PX,
        tile.x1 < width and x1 >= tile.x1 - SEAM_PX,
        tile.y1 < height and y1 >= tile.y1 - SEAM_PX,
    )


def _span_overlap(a0: float, a1: float, b0: float, b1: float) -> float:
    shorter = min(a1 - a0, b1 - b0)
    return 0.0 if shorter <= 0 else max(0.0, min(a1, b1) - max(a0, b0)) / shorter


def _continues(a: _Box, b: _Box, tiles: list[Tile]) -> bool:
    """Whether ``b`` continues ``a`` across one of the inner edges that cut ``a``."""
    if a.label != b.label or a.tile == b.tile:
        return False
    ax0, ay0, ax1, ay1 = a.xyxy
    bx0, by0, bx1, by1 = b.xyxy
    ta = tiles[a.tile]
    gap = SEAM_PX
    touches_x = bx0 <= ax1 + gap and bx1 >= ax0 - gap
    touches_y = by0 <= ay1 + gap and by1 >= ay0 - gap
    if not (touches_x and touches_y):
        return False
    left, top, right, bottom = a.cut
    # The other box must reach past the edge that cut this one, and the two must cover
    # the same span along that edge.
    across_x = (left and bx0 < ta.x0) or (right and bx1 > ta.x1)
    across_y = (top and by0 < ta.y0) or (bottom and by1 > ta.y1)
    if across_x and _span_overlap(ay0, ay1, by0, by1) >= SEAM_SPAN:
        return True
    return bool(across_y and _span_overlap(ax0, ax1, bx0, bx1) >= SEAM_SPAN)


def _merge_seams(boxes: list[_Box], tiles: list[Tile]) -> list[_Box]:
    parent = list(range(len(boxes)))

    def find(i: int) -> int:
        while parent[i] != i:
            parent[i] = parent[parent[i]]
            i = parent[i]
        return i

    for i, a in enumerate(boxes):
        if not any(a.cut):
            continue
        for j, b in enumerate(boxes):
            if i != j and _continues(a, b, tiles):
                parent[find(i)] = find(j)
    groups: dict[int, list[_Box]] = {}
    for i, box in enumerate(boxes):
        groups.setdefault(find(i), []).append(box)
    out: list[_Box] = []
    for members in groups.values():
        if len(members) == 1:
            out.append(members[0])
            continue
        xyxy = np.array(
            [
                min(m.xyxy[0] for m in members),
                min(m.xyxy[1] for m in members),
                max(m.xyxy[2] for m in members),
                max(m.xyxy[3] for m in members),
            ]
        )
        best = max(members, key=lambda m: m.score)
        uncut = (False, False, False, False)
        out.append(_Box(best.label, best.score, xyxy, best.tile, uncut, best.source))
    return out


def _nms_per_label(boxes: list[_Box], iou: float) -> list[_Box]:
    out: list[_Box] = []
    for label in dict.fromkeys(b.label for b in boxes):
        same = [b for b in boxes if b.label == label]
        xyxy = np.stack([b.xyxy for b in same])
        out.extend(same[i] for i in nms(xyxy, [b.score for b in same], iou))
    return out


class TiledDetector:
    """A ``Detector`` that runs ``inner`` on tiles and joins the results (module doc)."""

    name: ClassVar[str] = "tiled"
    license: ClassVar[str] = "Apache-2.0"  # this wrapper; the model's licence is inner's

    def __init__(
        self,
        inner: Detector,
        tile_px: int = DEFAULT_TILE_PX,
        overlap: float = DEFAULT_OVERLAP,
        nms_iou: float = NMS_IOU,
        skip_mask: Callable[[Tile], bool] | None = None,
        clock: Callable[[], float] = time.perf_counter,
        upscale: int = 1,
    ) -> None:
        span = check_upscale(tile_px, upscale)
        tile_grid(span, span, span, overlap)  # validates the settings
        self.inner = inner
        self.tile_px = tile_px
        self.upscale = upscale
        self.span = span  # picture pixels per tile side
        self.overlap = overlap
        self.nms_iou = nms_iou
        self.skip_mask = skip_mask
        self._clock = clock
        self.last_tiles = 0
        self.last_latency_ms = 0.0

    def tiles(self, width: int, height: int) -> list[Tile]:
        grid = tile_grid(width, height, self.span, self.overlap)
        if self.skip_mask is None:
            return grid
        return [t for t in grid if not self.skip_mask(t)]

    def detect(self, image: Frame) -> list[Detection]:
        start = self._clock()
        pixels = np.frombuffer(image.rgb, dtype=np.uint8).reshape(image.height, image.width, 3)
        tiles = self.tiles(image.width, image.height)
        found: list[_Box] = []
        for index, tile in enumerate(tiles):
            crop = np.ascontiguousarray(pixels[tile.y0 : tile.y1, tile.x0 : tile.x1])
            crop = enlarge(crop, self.upscale)
            frame = Frame(
                width=crop.shape[1],
                height=crop.shape[0],
                rgb=crop.tobytes(),
                ts_mono_ms=image.ts_mono_ms,
            )
            for det in self.inner.detect(frame):
                xyxy = tile_to_picture(from_bbox([det.bbox])[0], tile, self.upscale)
                xyxy = np.clip(xyxy, 0, [image.width, image.height, image.width, image.height])
                if xyxy[2] - xyxy[0] <= 0 or xyxy[3] - xyxy[1] <= 0:
                    continue
                cut = _cut(xyxy, tile, image.width, image.height)
                found.append(_Box(det.label, det.score, xyxy, index, cut, det.source))
        merged = _nms_per_label(_merge_seams(found, tiles), self.nms_iou)
        merged.sort(key=lambda b: -b.score)
        self.last_tiles = len(tiles)
        self.last_latency_ms = (self._clock() - start) * 1000.0
        return [
            Detection(label=b.label, score=b.score, bbox=to_bbox(b.xyxy), source=b.source)
            for b in merged
        ]
