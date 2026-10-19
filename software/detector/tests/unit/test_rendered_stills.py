"""The colour detector and the oracle source on pictures rendered by Gazebo.

The fixtures (``tests/testdata/stills``) were rendered in the simulator's world with the
SK-RED objects (three red blocks, a red car and a red house as decoys, a blue car and a
road) by ``python -m sd_sim.render_stills``. ``stills.json`` holds each picture's camera
pose and the simulator's labelled boxes for the same moment, which are the answer key.

A block whose labelled box touches the picture's edge does not count: the colour
detector skips patches it cannot see whole, and the overlapping next still sees it.
"""

from __future__ import annotations

import json
import math
from functools import cache
from pathlib import Path
from typing import Any

import pytest
from PIL import Image

from sd_detect import (
    BBox,
    ColorDetector,
    Detection,
    Frame,
    SimBox,
    SimLabelDetector,
    Still,
    build_sources,
    default_registry,
    detect_all,
    keep_plausible,
)
from sd_geo import Attitude, PinholeCamera, ground_offset, px_per_m

FIXTURES = Path(__file__).resolve().parents[1] / "testdata" / "stills"
META: dict[str, Any] = json.loads((FIXTURES / "stills.json").read_text(encoding="utf-8"))
CAM = PinholeCamera(META["camera"]["width"], META["camera"]["height"], META["camera"]["hfov_deg"])
SHOTS: list[dict[str, Any]] = META["shots"]
BLOCKS = [o for o in META["objects"] if o["cls"] == "red_block"]
MAX_GEO_ERROR_M = 0.5  # exact pose: only the block's height and the pixel grid remain


@cache
def frame(name: str) -> Frame:
    with Image.open(FIXTURES / f"{name}.png") as im:
        rgb = im.convert("RGB")
        return Frame(width=rgb.width, height=rgb.height, rgb=rgb.tobytes(), ts_mono_ms=0)


def red_blocks_found(shot: dict[str, Any]) -> list[Detection]:
    reg = default_registry()
    found = ColorDetector.from_registry(reg).detect(frame(shot["name"]))
    return keep_plausible(found, px_per_m(CAM, shot["height_m"]), reg)


def whole_truth_boxes(shot: dict[str, Any], cls: str) -> list[BBox]:
    out = []
    for b in shot["boxes"]:
        if b["cls"] != cls:
            continue
        inside = b["x"] > 0 and b["y"] > 0 and b["x"] + b["w"] < CAM.width_px - 1
        if inside and b["y"] + b["h"] < CAM.height_px - 1:
            out.append(BBox(x=b["x"], y=b["y"], w=b["w"], h=b["h"]))
    return out


def iou(a: BBox, b: BBox) -> float:
    over_x = max(0.0, min(a.x + a.w, b.x + b.w) - max(a.x, b.x))
    over_y = max(0.0, min(a.y + a.h, b.y + b.h) - max(a.y, b.y))
    inter = over_x * over_y
    return inter / (a.w * a.h + b.w * b.h - inter)


def test_fixtures_cover_the_cases() -> None:
    names = {s["name"] for s in SHOTS}
    assert len(BLOCKS) == 3
    assert {"red_car_0_h30", "building_0_h30", "car_1_h30", "block0_tilted"} <= names
    assert sum(1 for s in SHOTS if whole_truth_boxes(s, "red_block")) >= 10
    assert sum(1 for s in SHOTS if not whole_truth_boxes(s, "red_block")) >= 6
    assert {o["role"] for o in META["objects"]} == {"target", "decoy", "context"}


def test_recall_and_precision() -> None:
    """Every whole red block is found (IoU >= 0.5 with its labelled box), and nothing
    else is called a red block: not the red car, not the red house, not the road."""
    tp = fp = fn = 0
    img_tp = img_fp = img_fn = img_tn = 0
    for shot in SHOTS:
        truth = whole_truth_boxes(shot, "red_block")
        found = red_blocks_found(shot)
        assert all(d.label == "red_block" and d.source == "color" for d in found)
        unmatched = list(truth)
        for d in found:
            best = max(unmatched, key=lambda t: iou(d.bbox, t), default=None)
            if best is not None and iou(d.bbox, best) >= 0.5:
                tp += 1
                unmatched.remove(best)
            else:
                fp += 1
        fn += len(unmatched)
        img_tp += bool(truth) and bool(found)
        img_fn += bool(truth) and not found
        img_fp += not truth and bool(found)
        img_tn += not truth and not found
    recall, precision = tp / (tp + fn), tp / (tp + fp)
    img_recall, img_precision = img_tp / (img_tp + img_fn), img_tp / (img_tp + img_fp)
    assert (recall, precision) == (1.0, 1.0), (tp, fp, fn)
    assert (img_recall, img_precision) == (1.0, 1.0), (img_tp, img_fp, img_fn, img_tn)
    assert tp == 14 and img_tn == 8  # every picture was looked at


def test_red_car_is_seen_but_dropped_by_size() -> None:
    shot = next(s for s in SHOTS if s["name"] == "red_car_0_h30")
    raw = ColorDetector.from_registry(default_registry()).detect(frame(shot["name"]))
    assert len(raw) == 1  # red pixels: the colour alone cannot tell a car from a block
    assert red_blocks_found(shot) == []  # 4.5 m long: no red block


@pytest.mark.parametrize(
    "shot", [s for s in SHOTS if whole_truth_boxes(s, "red_block")], ids=lambda s: s["name"]
)
def test_georeferenced_block_lies_on_the_truth(shot: dict[str, Any]) -> None:
    att = Attitude(**shot["attitude"])
    for d in red_blocks_found(shot):
        off = ground_offset(
            d.bbox.x + d.bbox.w / 2, d.bbox.y + d.bbox.h / 2, CAM, att, shot["height_m"]
        )
        assert off is not None
        east, north = shot["east_m"] + off[0], shot["north_m"] + off[1]
        err = min(math.hypot(east - b["east_m"], north - b["north_m"]) for b in BLOCKS)
        assert err <= MAX_GEO_ERROR_M, (shot["name"], err)


def _still(shot: dict[str, Any]) -> Still:
    labels = {spec.id: n for n, spec in default_registry().by_gz_label().items()}
    boxes = tuple(
        SimBox(gz_label=labels[b["cls"]], bbox=BBox(x=b["x"], y=b["y"], w=b["w"], h=b["h"]))
        for b in shot["boxes"]
    )
    return Still(frame=frame(shot["name"]), sim_boxes=boxes)


def test_oracle_reports_cars_buildings_and_roads_as_oracle() -> None:
    oracle = SimLabelDetector.from_registry(default_registry(), {"SD_SIM": "1"})
    seen: set[str] = set()
    for shot in SHOTS:
        for d in oracle.detect_still(_still(shot)):
            assert d.source == "oracle" and d.score == 1.0
            assert d.label in {"car", "building", "road"}  # never the targets
            seen.add(d.label)
    assert seen == {"car", "building", "road"}


def test_all_sources_together_in_the_simulator() -> None:
    reg = default_registry()
    built = build_sources(reg, env={"SD_SIM": "1"})
    shot = next(s for s in SHOTS if s["name"] == "block0_h40")
    found = detect_all(_still(shot), built.sources, reg)
    by_source = {(d.label, d.source) for d in found}
    assert ("red_block", "color") in by_source
    assert ("car", "oracle") in by_source
    assert ("red_block", "oracle") not in by_source


def test_oracle_refuses_outside_the_simulator() -> None:
    with pytest.raises(RuntimeError, match="SD_SIM=1"):
        SimLabelDetector.from_registry(default_registry(), {})
