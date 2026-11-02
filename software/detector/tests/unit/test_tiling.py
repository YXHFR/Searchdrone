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
from sd_detect.tiling import Tile, TiledDetector, check_upscale, enlarge, tile_grid, tile_to_picture


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


# --- upscale: each tile covers tile_px / upscale picture pixels, enlarged to tile_px ---


class Sizes(BrightBlobs):
    """``BrightBlobs`` that also notes the size of every tile it is given."""

    def __init__(self) -> None:
        super().__init__()
        self.sizes: list[tuple[int, int]] = []

    def detect(self, image: Frame) -> list[Detection]:
        self.sizes.append((image.width, image.height))
        return super().detect(image)


def test_twelve_megapixels_at_2x_is_170_tiles_of_320_picture_pixels() -> None:
    det = TiledDetector(BrightBlobs(), upscale=2)
    grid = det.tiles(4608, 2592)
    assert det.span == 320 and len(grid) == 170  # 17 x 10, against 45 at 1x
    assert grid[0] == Tile(0, 0, 320, 320)
    assert grid[-1] == Tile(4608 - 320, 2592 - 320, 4608, 2592)
    assert len(TiledDetector(BrightBlobs()).tiles(4608, 2592)) == 45


@pytest.mark.parametrize(("width", "height"), [(4608, 2592), (1000, 333), (321, 700), (100, 50)])
def test_2x_tiles_cover_the_picture_and_overlap_up_to_the_edges(width: int, height: int) -> None:
    grid = TiledDetector(BrightBlobs(), upscale=2).tiles(width, height)
    covered = np.zeros((height, width), dtype=bool)
    for t in grid:
        assert 0 <= t.x0 < t.x1 <= width and 0 <= t.y0 < t.y1 <= height
        assert t.x1 - t.x0 == min(320, width) and t.y1 - t.y0 == min(320, height)
        covered[t.y0 : t.y1, t.x0 : t.x1] = True
    assert covered.all()
    for xs in (sorted({t.x0 for t in grid}), sorted({t.y0 for t in grid})):
        # neighbours overlap by at least 15 %, the last pair (at the edge) too
        assert all(b - a <= 320 * 0.85 for a, b in itertools.pairwise(xs))


@settings(max_examples=200, deadline=None)
@given(
    x0=st.integers(0, 2000),
    y0=st.integers(0, 2000),
    box=st.tuples(*[st.floats(0, 640, allow_nan=False)] * 4),
    upscale=st.integers(1, 4),
)
def test_tile_boxes_map_back_to_the_picture_and_round_trip(
    x0: int, y0: int, box: tuple[float, float, float, float], upscale: int
) -> None:
    tile = Tile(x0, y0, x0 + 640 // upscale, y0 + 640 // upscale)
    xyxy = np.array(box)
    picture = tile_to_picture(xyxy, tile, upscale)
    offset = np.array([x0, y0, x0, y0])
    assert picture == pytest.approx(xyxy / upscale + offset)
    assert (picture - offset) * upscale == pytest.approx(xyxy)  # back to the tile


def test_enlarge_is_bilinear_and_keeps_a_block_in_place() -> None:
    rgb = np.zeros((10, 20, 3), np.uint8)
    rgb[2:5, 6:9] = 255
    big = enlarge(rgb, 2)
    assert big.shape == (20, 40, 3) and big.dtype == np.uint8
    mask = big[:, :, 0] > 128
    rows, cols = np.flatnonzero(mask.any(axis=1)), np.flatnonzero(mask.any(axis=0))
    assert (rows[0], rows[-1] + 1, cols[0], cols[-1] + 1) == (4, 10, 12, 18)
    assert enlarge(rgb, 1) is rgb


def test_at_2x_the_engine_sees_enlarged_tiles_and_boxes_come_back_in_picture_pixels() -> None:
    inner = Sizes()
    det = TiledDetector(inner, upscale=2)
    people = [(40, 30, 52, 58), (900, 500, 915, 530)]  # 12 and 15 px wide
    found = det.detect(_frame(1000, 600, people))
    assert det.last_tiles == len(inner.sizes) == 4 * 3
    assert max(inner.sizes) == (640, 640)  # each 320 px tile enlarged to 640
    assert sorted(_xyxy(d) for d in found) == [pytest.approx(p) for p in people]


def test_at_2x_a_person_across_a_seam_is_one_box() -> None:
    det = TiledDetector(BrightBlobs(), upscale=2)
    # 592 = 320 + 272: two tiles side by side, [0, 320) and [272, 592)
    person = (250, 100, 350, 130)
    found = det.detect(_frame(592, 320, [person]))
    assert det.last_tiles == 2
    assert [_xyxy(d) for d in found] == [pytest.approx(person)]


@settings(max_examples=150, deadline=None)
@given(
    x=st.integers(0, 600 - 20),
    y=st.integers(0, 600 - 20),
    w=st.integers(6, 120),
    h=st.integers(6, 120),
)
def test_at_2x_one_person_anywhere_is_one_box(x: int, y: int, w: int, h: int) -> None:
    person = (x, y, min(x + w, 600), min(y + h, 600))
    found = TiledDetector(BrightBlobs(), upscale=2).detect(_frame(600, 600, [person]))
    assert len(found) == 1
    assert _xyxy(found[0]) == pytest.approx(person)


@pytest.mark.parametrize("upscale", [0, 5, 3, 2.0, True])
def test_a_bad_upscale_is_refused(upscale: object) -> None:
    with pytest.raises(ValueError, match="upscale"):
        TiledDetector(BrightBlobs(), upscale=upscale)  # type: ignore[arg-type]
    with pytest.raises(ValueError, match="upscale"):
        check_upscale(640, upscale)  # type: ignore[arg-type]


def test_upscale_may_not_leave_a_tile_under_32_picture_pixels() -> None:
    with pytest.raises(ValueError):
        TiledDetector(BrightBlobs(), tile_px=64, upscale=4)  # 16 px
    assert TiledDetector(BrightBlobs(), tile_px=128, upscale=4).span == 32
