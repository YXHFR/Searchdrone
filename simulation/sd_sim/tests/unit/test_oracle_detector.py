"""The detector oracle and its synthetic stills: the world drawn from the drone's pose,
recall by pixel size from the profile, only in the simulator."""

from __future__ import annotations

import logging
import math
from typing import Any

import pytest
import yaml

from sd_detect.api import BBox, Detection
from sd_geo import PinholeCamera
from sd_mav.vehicle import FcTelemetry
from sd_sim import oracle_detector
from sd_sim.oracle_detector import (
    ATT_SIGMA_DEG,
    NOT_IN_SIMULATOR,
    POS_SIGMA_M,
    Drawn,
    OracleDetector,
    OraclePerception,
    SyntheticStillSource,
    load_det_profile,
)
from sd_sim.search_world import CARD_ERR_MAX_M, DECOY_CLEAR_M, SCENARIOS, load
from sd_sim.world import STILLS

SCRIPTS = SCENARIOS / "scripts"

SIM = {"SD_SIM": "1"}
CAM = PinholeCamera(1280, 720, 66.0)


def _source(scenario: str, thing: str, alt_m: float = 50.0) -> SyntheticStillSource:
    w = load(scenario, 1)
    t = next(x for x in w.things if x.thing_id == thing)
    lat, lon = w.where(t)
    tel = FcTelemetry(lat=lat, lon=lon, alt_rel_home_m=alt_m, alt_amsl_m=584 + alt_m, yaw_deg=0)
    return SyntheticStillSource(w, lambda: tel, CAM, env=SIM)


def test_refuses_outside_the_simulator() -> None:
    w = load("SK01", 1)
    with pytest.raises(RuntimeError, match="SD_SIM"):
        SyntheticStillSource(w, lambda: FcTelemetry(), CAM, env={})
    assert "SD_SIM=1" in NOT_IN_SIMULATOR


def test_the_perception_task_gets_the_oracle_as_its_only_source() -> None:
    src = _source("SK01", "target")
    det = OracleDetector(src, seed=5, sleep=False, env=SIM)
    clock: Any = object()  # never read: nothing runs
    lp = OraclePerception(det, src.telemetry, lambda _p: None, clock, CAM, env=SIM)
    assert lp.built.sources == {"model": det}  # passed in, never built from the register


def test_the_profile_recall_grows_with_pixels_and_falls_with_occlusion() -> None:
    p = load_det_profile()
    assert p.p_found(10, 0) < p.p_found(24, 0) < p.p_found(40, 0)
    assert p.p_found(40, 70) < p.p_found(40, 10)


def test_a_lying_person_under_the_drone_is_drawn_and_mostly_found() -> None:
    src = _source("SK01", "target")
    det = OracleDetector(src, seed=5, sleep=False, env=SIM)
    hits = 0
    for _ in range(40):
        src._last = None  # pyright: ignore[reportPrivateUsage] - one still per call
        cs = src.poll()
        assert cs is not None and cs.still.frame.width == 1280
        found = det.detect_still(cs.still)
        hits += any(d.score > 0 for d in found)
    assert det.in_view.get("target", 0) == 40
    assert hits >= 28  # recall over 32 px is 0.889


def test_the_same_seed_draws_the_same_detections() -> None:
    def draw() -> list[float]:
        src = _source("SK01", "target")
        det = OracleDetector(src, seed=9, sleep=False, env=SIM)
        out: list[float] = []
        for _ in range(5):
            src._last = None  # pyright: ignore[reportPrivateUsage]
            cs = src.poll()
            assert cs is not None
            out += [round(d.score, 4) for d in det.detect_still(cs.still)]
        return out

    assert draw() == draw()


def test_every_detection_says_what_it_was_drawn_from() -> None:
    """The rig judges a card by its picture's source: each detection the oracle returns
    (the person's and the false positives on plain ground) maps back to what it drew it
    from, by its box and score; a box it never made maps to nothing."""
    src = _source("SK01", "target")
    det = OracleDetector(src, seed=5, sleep=False, env=SIM)
    seen: set[str | None] = set()
    for _ in range(40):
        src._last = None  # pyright: ignore[reportPrivateUsage]
        cs = src.poll()
        assert cs is not None
        for d in det.detect_still(cs.still):
            drawn = det.source_of(d.bbox, d.score)
            assert drawn is not None and drawn.score == d.score
            seen.add(drawn.thing_id)
    assert "target" in seen
    assert det.source_of(BBox(x=1.0, y=2.0, w=3.0, h=4.0), 0.5) is None


def _script_alts(node: Any) -> list[float]:
    if isinstance(node, dict):
        out = [float(v) for k, v in node.items() if k == "alt_m" and isinstance(v, int | float)]
        return out + [a for v in node.values() for a in _script_alts(v)]
    if isinstance(node, list):
        return [a for v in node for a in _script_alts(v)]
    return []


def test_the_card_error_bound_covers_the_oracles_position_error() -> None:
    """CARD_ERR_MAX_M is more than four times the oracle's error per axis at the corner
    of the stills camera's picture (where the attitude moves a point most) and at the
    highest altitude any SK script or world flies, and a decoy drawn in an area lies
    twice that from every person, so a person's card is nearer the person."""
    alts = [
        a
        for p in sorted(SCRIPTS.glob("SK*.yaml"))
        for a in _script_alts(yaml.safe_load(p.read_text()))
    ]
    alts += [load(p, 1).survey.get("alt_m", 0.0) for p in sorted(SCENARIOS.glob("SK*.yaml"))]
    h = max(alts)
    assert h >= 50.0
    cam = PinholeCamera(int(STILLS["width"]), int(STILLS["height"]), float(STILLS["hfov_deg"]))
    tan_off = math.hypot(cam.width_px / 2, cam.height_px / 2) / cam.focal_px
    assert math.degrees(math.atan(tan_off)) == pytest.approx(36.7, abs=0.5)
    att = math.radians(ATT_SIGMA_DEG)
    tilt = h * (1.0 + tan_off**2) * att  # along the ray's direction: h / cos^2
    turn = h * tan_off * att  # across it, from the heading's error
    sigma = math.sqrt(POS_SIGMA_M**2 + tilt**2 + turn**2)
    assert 4.0 * sigma <= CARD_ERR_MAX_M
    assert DECOY_CLEAR_M >= 2.0 * CARD_ERR_MAX_M


def test_forgotten_sources_are_counted_and_logged_once(
    caplog: pytest.LogCaptureFixture, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Past SOURCES_KEPT the oldest sources go (their cards are judged by position); the
    run counts them and says so once."""
    monkeypatch.setattr(oracle_detector, "SOURCES_KEPT", 2)
    det = OracleDetector(_source("SK01", "target"), seed=5, sleep=False, env=SIM)
    drawn = Drawn(None, False, False, 0.5, 10.0)
    with caplog.at_level(logging.WARNING, logger=oracle_detector.__name__):
        for i in range(5):
            box = BBox(x=float(i), y=0.0, w=1.0, h=1.0)
            det._drew(Detection(label="person", score=0.5, bbox=box), drawn)  # pyright: ignore[reportPrivateUsage]
    assert det.sources_dropped == 3
    assert len(caplog.records) == 1
    assert det.source_of(BBox(x=0.0, y=0.0, w=1.0, h=1.0), 0.5) is None
    assert det.source_of(BBox(x=4.0, y=0.0, w=1.0, h=1.0), 0.5) is drawn
