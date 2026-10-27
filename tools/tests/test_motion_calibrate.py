"""The motion calibration tool on a small world: it fits, and writes back in place."""

from __future__ import annotations

import math
from datetime import date
from itertools import pairwise

import motion_calibrate as mc
import numpy as np
import pytest

from sd_motion.params import load_all

SMALL = mc.World(size_km=6.0, cell_m=100.0, hours=3.0, dt_s=60.0)


def test_walk_keeps_the_walkers() -> None:
    w = mc.walk(SMALL, load_all()["hiker"], 1.0)
    assert w.times_h[0] == 0.0
    assert w.shares[0, 0] == pytest.approx(1.0)
    assert np.allclose(w.shares.sum(axis=1), 1.0, atol=1e-4)
    assert w.shares[-1, -1] > 0  # some walked off the map


def test_longer_walking_finds_people_further_out() -> None:
    p = load_all()["despondent"]
    w = mc.walk(SMALL, p, 1.0)
    short = mc.found_quartiles(w, mc.scale_mobility(p, 0.25))
    long = mc.found_quartiles(w, mc.scale_mobility(p, 2.0))
    assert short[1] < long[1]
    assert all(a <= b for a, b in pairwise(short))


def test_people_who_never_move_are_found_at_the_start() -> None:
    p = load_all()["dementia"]  # 25 % never move
    w = mc.walk(SMALL, p, 1.0)
    q25, _, _ = mc.found_quartiles(w, p)
    assert q25 <= SMALL.cell_m / 1000


def test_error_is_zero_on_target_and_huge_off_the_map() -> None:
    assert mc.error((1.0, 2.0, 3.0), (1.0, 2.0, 3.0)) == 0.0
    assert mc.error((1.0, 2.0, math.inf), (1.0, 2.0, 3.0)) > 1e5


def test_write_back_replaces_values_and_keeps_comments() -> None:
    text = mc.TOML.read_text(encoding="utf-8")
    table = load_all(text)
    fit = mc.Fit("hiker", 2.0, 0.5, (0.6, 1.7, 3.4))
    out = mc.write_back(text, table, [fit], date(2026, 10, 26))
    new = load_all(out)
    assert new["hiker"].turn45 == pytest.approx(table["hiker"].turn45 * 2.0)
    assert new["hiker"].mobility[1][0] == pytest.approx(table["hiker"].mobility[1][0] * 0.5)
    assert new["dementia"] == table["dementia"]
    assert "Tobler" in out  # comments kept
    assert 'calibrated = "2026-10-26: found within 0.60/1.70/3.40 km' in out
    again = mc.write_back(out, new, [mc.Fit("hiker", 1.0, 1.0, (0.6, 1.7, 3.4))],
                          date(2026, 10, 27))  # fmt: skip
    assert again.count("calibrated =") == 1
    assert load_all(again)["hiker"] == new["hiker"]


def test_a_finished_walk_is_kept_and_reused(tmp_path) -> None:
    hiker = load_all()["hiker"]
    _, _, first = mc._walk_job((SMALL, "hiker", hiker, 1.0, tmp_path))
    assert len(list(tmp_path.glob("hiker-*.npz"))) == 1
    _, _, again = mc._walk_job((SMALL, "hiker", hiker, 1.0, tmp_path))
    assert np.array_equal(first.shares, again.shares)
    assert again.bin_km == first.bin_km
