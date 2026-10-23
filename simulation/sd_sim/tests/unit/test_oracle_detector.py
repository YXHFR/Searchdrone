"""The detector oracle and its synthetic stills: the world drawn from the drone's pose,
recall by pixel size from the profile, only in the simulator."""

from __future__ import annotations

import pytest

from sd_geo import PinholeCamera
from sd_mav.vehicle import FcTelemetry
from sd_sim.oracle_detector import (
    NOT_IN_SIMULATOR,
    OracleDetector,
    SyntheticStillSource,
    load_det_profile,
)
from sd_sim.search_world import load

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
