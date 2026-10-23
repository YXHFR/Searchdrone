"""The tiled detector: the grid, moving boxes back, and joining halves across a seam."""

from __future__ import annotations

import itertools
from typing import ClassVar

import numpy as np
import pytest
from hypothesis import given, settings
from hypothesis import strategies as st

from sd_detect import BBox, Detection, Detector, Frame
from sd_detect.boxes import iou_matrix, nms
from sd_detect.tiling import Tile, TiledDetector, tile_grid


class BrightBlobs:
    """Reports every bright column run per tile as one box: a stand-in for a model that
    sees whatever part of a person lies in its tile."""

    name: ClassVar[str] = "bright"
    license: ClassVar[str] = "Apache-2.0"

    def __init__(self) -> None:
        self.calls = 0

    def detect(self, image: Frame) -> list[Detection]:
        self.calls += 1
        px = np.frombuffer(image.rgb, dtype=np.uint8).reshape(image.height, image.width, 3)
        mask = px[:, :, 0] > 128
        out: list[Detection] = []
        cols = np.flatnonzero(mask.any(axis=0))
        if cols.size == 0:
            return out
        # split into runs of touching columns: one box per run
        runs = np.split(cols, np.flatnonzero(np.diff(cols) > 1) + 1)
        for run in runs:
            rows = np.flatnonzero(mask[:, run].any(axis=1))
            x0, x1 = int(run[0]), int(run[-1]) + 1
            y0, y1 = int(rows[0]), int(rows[-1]) + 1
            out.append(
                Detection(label="person", score=0.9, bbox=BBox(x=x0, y=y0, w=x1 - x0, h=y1 - y0))
            )
        return out


def _frame(width: int, height: int, people: list[tuple[int, int, int, int]]) -> Frame:
    px = np.zeros((height, width, 3), dtype=np.uint8)
    for x0, y0, x1, y1 in people:
        px[y0:y1, x0:x1] = 255
    return Frame(width=width, height=height, rgb=px.tobytes(), ts_mono_ms=0)


def _xyxy(d: Detection) -> tuple[float, float, float, float]:
    return (d.bbox.x, d.bbox.y, d.bbox.x + d.bbox.w, d.bbox.y + d.bbox.h)


def test_twelve_megapixels_is_45_tiles() -> None:
    grid = tile_grid(4608, 2592)
    assert len(grid) == 45
    assert grid[0] == Tile(0, 0, 640, 640)
    assert grid[-1] == Tile(4608 - 640, 2592 - 640, 4608, 2592)
    xs = sorted({t.x0 for t in grid})
    assert all(b - a <= 640 * 0.85 for a, b in itertools.pairwise(xs))


def test_small_picture_is_one_tile() -> None:
    assert tile_grid(300, 200) == [Tile(0, 0, 300, 200)]


def test_tiled_detector_is_a_detector() -> None:
    det: Detector = TiledDetector(BrightBlobs())
    assert isinstance(det, Detector)


def test_tiling_roundtrip() -> None:
    """A person crossing a seam, larger than the overlap, so no tile sees them whole,
    comes out as exactly one box: the whole person."""
    inner = BrightBlobs()
    det = TiledDetector(inner)
    # 1184 = 640 + 544: two tiles side by side, [0, 640) and [544, 1184)
    person = (500, 100, 700, 160)
    found = det.detect(_frame(1184, 640, [person]))
    assert inner.calls == 2 and det.last_tiles == 2
    assert len(found) == 1
    assert _xyxy(found[0]) == pytest.approx(person)


def test_person_inside_the_overlap_is_one_box() -> None:
    det = TiledDetector(BrightBlobs())
    person = (580, 300, 600, 340)  # seen whole by both tiles
    found = det.detect(_frame(1184, 640, [person]))
    assert [_xyxy(d) for d in found] == [pytest.approx(person)]


def test_two_people_near_a_seam_stay_two() -> None:
    det = TiledDetector(BrightBlobs())
    people = [(560, 100, 630, 140), (660, 400, 720, 440)]
    found = det.detect(_frame(1184, 640, people))
    assert sorted(_xyxy(d) for d in found) == [pytest.approx(p) for p in people]


def test_corner_of_four_tiles() -> None:
    det = TiledDetector(BrightBlobs())
    person = (500, 500, 700, 700)  # across a vertical and a horizontal seam
    found = det.detect(_frame(1184, 1184, [person]))
    assert [_xyxy(d) for d in found] == [pytest.approx(person)]


@settings(max_examples=300, deadline=None)
@given(
    x=st.integers(0, 1184 - 40),
    y=st.integers(0, 1184 - 40),
    w=st.integers(8, 220),
    h=st.integers(8, 220),
)
def test_one_person_anywhere_is_one_box(x: int, y: int, w: int, h: int) -> None:
    person = (x, y, min(x + w, 1184), min(y + h, 1184))
    found = TiledDetector(BrightBlobs()).detect(_frame(1184, 1184, [person]))
    assert len(found) == 1
    assert _xyxy(found[0]) == pytest.approx(person)


def test_skip_mask_skips_tiles() -> None:
    inner = BrightBlobs()
    det = TiledDetector(inner, skip_mask=lambda t: t.x0 > 0)
    det.detect(_frame(1184, 640, []))
    assert inner.calls == 1 and det.last_tiles == 1


def test_latency_is_measured_with_the_clock() -> None:
    ticks = iter([1.0, 1.25])
    det = TiledDetector(BrightBlobs(), clock=lambda: next(ticks))
    det.detect(_frame(64, 64, []))
    assert det.last_latency_ms == pytest.approx(250.0)


def test_bad_settings_are_refused() -> None:
    with pytest.raises(ValueError):
        TiledDetector(BrightBlobs(), tile_px=8)
    with pytest.raises(ValueError):
        TiledDetector(BrightBlobs(), overlap=0.95)


def test_nms_and_iou() -> None:
    boxes = np.array([[0, 0, 10, 10], [1, 1, 11, 11], [50, 50, 60, 60]], dtype=np.float64)
    assert nms(boxes, [0.5, 0.9, 0.1], 0.5) == [1, 2]
    assert iou_matrix(boxes[:1], boxes[:1])[0, 0] == pytest.approx(1.0)
