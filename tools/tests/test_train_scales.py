"""Scale-matched training tiles (``tools/train/tiles.py``, ``scales=``): each training
picture resampled so its people are as many pixels as our cameras see them, and the
ignore regions of a set kept out of every tile."""

from __future__ import annotations

import importlib.util
import json
import math
import sys
from pathlib import Path
from types import ModuleType

import numpy as np
import pytest
from bench.datasets import Image
from PIL import Image as PILImage

TRAIN = Path(__file__).resolve().parents[1] / "train"


def _tiles() -> ModuleType:
    spec = importlib.util.spec_from_file_location("train_tiles_scales", TRAIN / "tiles.py")
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module  # its dataclass looks itself up there
    spec.loader.exec_module(module)
    return module


tiles = _tiles()


def test_the_cameras_give_the_owners_pixels_per_metre_at_50_m() -> None:
    """Camera Module 3 at full resolution, about 71 px/m; the tele, about 230 px/m; a
    standing person about 33 and 108 px."""
    wide, tele = tiles.camera_px_per_m("wide", 50), tiles.camera_px_per_m("tele", 50)
    assert wide == pytest.approx(71.0, abs=0.5)
    assert tele == pytest.approx(230.0, abs=1.0)
    assert wide * tiles.PERSON_M == pytest.approx(33, abs=1)
    assert 100 <= tele * tiles.PERSON_M <= 110


@pytest.mark.parametrize(
    ("text", "want"),
    [
        ("", ("native",)),
        ("native", ("native",)),
        ("wide,tele", ("wide", "tele")),
        ("native, wide,wide,150", ("native", "wide", "150")),
        ("71.0", ("71",)),
    ],
)
def test_scales_are_parsed(text: str, want: tuple[str, ...]) -> None:
    assert tiles.parse_scales(text) == want


@pytest.mark.parametrize("bad", ["zoom", "0", "-5", "nan", "1e9"])
def test_a_bad_scale_is_refused(bad: str) -> None:
    with pytest.raises(ValueError):
        tiles.parse_scales(bad)


def test_a_camera_target_lies_between_the_search_heights_and_is_fixed_by_the_id() -> None:
    low = tiles.camera_px_per_m("wide", max(tiles.SEARCH_HEIGHTS_M))
    high = tiles.camera_px_per_m("wide", min(tiles.SEARCH_HEIGHTS_M))
    got = [tiles.target_px_per_m("wide", f"heridal:{n}") for n in range(200)]
    assert all(t is not None and low <= t <= high for t in got)
    assert min(got) < low * 1.1 and max(got) > high * 0.9  # the whole range is used
    assert tiles.target_px_per_m("wide", "heridal:7") == got[7]
    assert tiles.target_px_per_m("native", "x") is None
    assert tiles.target_px_per_m("150", "x") == 150.0


def test_person_boxes_give_the_ground_scale() -> None:
    boxes = [(0.0, 0.0, 10.0, 47.0), (0.0, 0.0, 47.0, 20.0), (0.0, 0.0, 200.0, 10.0)]
    assert tiles.person_px_per_m(boxes) == pytest.approx(100.0)
    assert tiles.person_px_per_m([]) is None


def test_factors_never_enlarge_more_than_max_up_and_merge_near_ones() -> None:
    # 50 px/m: a 150 px/m target is 3 times, capped at 2; native and 52 px/m are one
    assert tiles.factors(50.0, ("native", "150", "52"), "x") == [("native", 1.0), ("150", 2.0)]
    assert tiles.factors(50.0, ("150",), "x", max_up=4.0) == [("150", 3.0)]
    # shrinking has no cap: a 300 px/m picture to 75 px/m is a quarter
    assert tiles.factors(300.0, ("75",), "x") == [("75", 0.25)]
    # no scale known (an empty picture of an empty set): only native
    assert tiles.factors(None, ("native", "wide"), "x") == [("native", 1.0)]
    assert tiles.factors(None, ("wide",), "x") == []


def _im(name: str, seq: str, boxes: list[tuple[float, float, float, float]]) -> Image:
    return Image(f"{name}:{seq}_{len(boxes)}", name, seq, "x.jpg", 100, 100, tuple(boxes))


def test_a_pictures_scale_comes_from_its_boxes_its_sequence_or_its_set() -> None:
    side = [(0.0, 0.0, 9.4, 4.0)]  # 9.4 px: 20 px/m
    own = _im("a", "S", side * 3)
    few = _im("a", "S", [(0.0, 0.0, 47.0, 4.0)])  # one box: the sequence's median
    empty = _im("a", "T", [])  # no box in its sequence: the set's
    root = Path()
    got = tiles.source_scales([(own, root), (few, root), (empty, root)])
    assert got[own.image_id] == pytest.approx(20.0)
    assert got[few.image_id] == pytest.approx(20.0)  # median of the sequence's 4 boxes
    assert got[empty.image_id] == pytest.approx(20.0)
    known = tiles.source_scales([(own, root)], {"a": 66.0})
    assert known[own.image_id] == 66.0


def test_a_tile_that_touches_an_ignore_box_is_left_out() -> None:
    boxes = [(10.0, 10.0, 30.0, 40.0)]
    plain = tiles.cut(300, 200, boxes, 128)
    cut = tiles.cut(300, 200, boxes, 128, ignore=[(200.0, 150.0, 210.0, 160.0)])
    assert len(plain) == 6
    left = {rect for rect, _ in plain} - {rect for rect, _ in cut}
    assert left and all(r[0] < 210 and r[2] > 200 and r[1] < 160 and r[3] > 150 for r in left)


def _bank(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> list[tuple[Image, Path]]:
    """Two 400 x 300 pictures of set ``heridal`` with 20 px people; the register says
    40 px/m."""
    monkeypatch.setattr(tiles, "known_scales", lambda items: {"heridal": 40.0})
    items = []
    for n in range(2):
        rgb = np.full((300, 400, 3), 90, dtype=np.uint8)
        boxes = [(40.0 + 100 * k, 50.0, 60.0, 70.0) for k in range(1)]
        boxes += [(150.0, 150.0, 170.0, 170.0), (300.0, 200.0, 320.0, 220.0)]
        for x0, y0, x1, y1 in boxes:
            rgb[int(y0) : int(y1), int(x0) : int(x1)] = (220, 40, 40)
        PILImage.fromarray(rgb).save(tmp_path / f"p{n}.png")
        items.append(
            (
                Image(f"heridal:p_S_{n}", "heridal", "S", f"p{n}.png", 400, 300, tuple(boxes)),
                tmp_path,
            )
        )
    return items


def test_the_cache_cuts_each_picture_at_every_scale_with_its_boxes(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    items = _bank(tmp_path, monkeypatch)
    native = tiles.build_cache(items, tmp_path / "n", 128, empty_share=0.0, progress=False)
    scaled = tiles.build_cache(
        items, tmp_path / "s", 128, empty_share=0.0, scales=("native", "20", "80")
    )
    assert {t.scale for t in native} == {1.0}
    assert {t.scale for t in scaled} == {1.0, 0.5, 2.0}
    for t in scaled:
        for b in t.boxes:
            side = max(b[2] - b[0], b[3] - b[1])
            assert side == pytest.approx(20 * t.scale, abs=1.5) or side < 20 * t.scale
    # the report says how large the people were cut, per set and scale
    out = capsys.readouterr().out
    assert "heridal 80: 6 personer" in out and "heridal 20: 6 personer" in out
    # a person still lies under its box in a resampled tile
    t = next(t for t in scaled if t.scale == 2.0 and t.boxes)
    x0, y0, x1, y1 = (round(v) for v in t.boxes[0])
    with PILImage.open(tmp_path / "s" / t.file) as pic:
        rgb = np.asarray(pic.convert("RGB"))
    centre = rgb[(y0 + y1) // 2, (x0 + x1) // 2]
    assert centre[0] > 180 and centre[1] < 90
    # the scale is kept in the listing and read back from the cache
    body = json.loads((tmp_path / "s" / "tiles.json").read_text(encoding="utf-8"))
    assert {t.get("scale", 1.0) for t in body["tiles"]} == {1.0, 0.5, 2.0}
    again = tiles.build_cache(
        items, tmp_path / "s", 128, empty_share=0.0, scales=("native", "20", "80")
    )
    assert again == scaled


def test_native_scale_keeps_the_old_cache_key_and_scales_change_it(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    items = _bank(tmp_path, monkeypatch)
    old = tiles.cache_key(items, 640, 0.25)
    assert tiles.cache_key(items, 640, 0.25, scales=("native",)) == old
    wide = tiles.cache_key(items, 640, 0.25, scales=("native", "wide"))
    assert wide != old
    assert tiles.cache_key(items, 640, 0.25, scales=("native", "wide"), max_up=3.0) != wide


def test_the_wide_camera_shrinks_big_people_to_about_33_px(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A picture whose people are 100 px (213 px/m) is shrunk for the wide camera so its
    people come out at what the camera sees between 30 and 60 m (about 28-56 px)."""
    monkeypatch.setattr(tiles, "known_scales", lambda items: {})
    rgb = np.full((600, 800, 3), 90, dtype=np.uint8)
    boxes = ((100.0, 100.0, 200.0, 140.0), (400.0, 300.0, 440.0, 400.0), (600.0, 50.0, 700.0, 90.0))
    PILImage.fromarray(rgb).save(tmp_path / "big.png")
    item = (Image("heridal:big_S_0", "heridal", "S", "big.png", 800, 600, boxes), tmp_path)
    # one 640 px tile holds the whole shrunk picture, so no box is cut at a tile's edge
    got = tiles.build_cache([item], tmp_path / "w", 640, empty_share=0.0, scales=("wide",))
    sides = [max(b[2] - b[0], b[3] - b[1]) for t in got for b in t.boxes]
    lo = tiles.camera_px_per_m("wide", 60) * tiles.PERSON_M
    hi = tiles.camera_px_per_m("wide", 30) * tiles.PERSON_M
    assert sides and all(lo - 1 <= s <= hi + 1 for s in sides)
    assert all(math.isclose(t.scale, got[0].scale) and t.scale < 1 for t in got)
