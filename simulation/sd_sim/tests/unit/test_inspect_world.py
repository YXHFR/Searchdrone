"""The closer look's scenario files and camera oracle (``sd_sim.inspect_world``), and the
run's report rows (``sd_sim.inspect_metrics.write_report``)."""

from __future__ import annotations

import json
import math
from pathlib import Path

import pytest

from sd_schemas.inspect import CLEAR_PX
from sd_sim import inspect_world as iw
from sd_sim.inspect_metrics import InspectMetrics, write_report
from sd_sim.obstacles import Scene

FILES = sorted(iw.SCENARIOS.glob("TN*.yaml"))


def test_there_is_a_file_for_every_flown_scenario() -> None:
    names = {p.name.split("_")[0] for p in FILES}
    # TN17 is a unit test, TN18 flies Hellasgården in Gazebo (its own site and world).
    want = {f"TN{n:02d}" for n in range(1, 20)} - {"TN17", "TN18"}
    assert names == want


@pytest.mark.parametrize("path", FILES, ids=lambda p: p.name)
def test_every_scenario_loads_for_every_seed(path: Path) -> None:
    name = path.name.split("_")[0]
    first = iw.load(name, 1)
    assert first.seeds == (1, 2, 3)
    for seed in first.seeds:
        w = iw.load(name, seed)
        assert w.scenario == name and w.seed == seed
        assert w.alt_min_agl_m < w.start_alt_m <= w.alt_max_agl_m
        assert w.overflight == (w.mode != "live")
        assert iw.load(name, seed) == w  # the same seed, the same world
        assert [s.trees() for s in w.stands] == [s.trees() for s in iw.load(name, seed).stands]


def test_the_seed_moves_the_person_and_the_estimate_within_their_jitter() -> None:
    worlds = [iw.load("TN01", s) for s in (1, 2, 3)]
    assert len({w.target.enu for w in worlds}) == 3
    for w in worlds:
        assert math.dist(w.target.enu, (40.0, 0.0)) <= 5.0
        assert math.dist(w.estimate_enu, w.target.enu) <= 3.0


def test_a_bad_file_says_what_is_wrong(tmp_path: Path) -> None:
    path = tmp_path / "TN99_x.yaml"
    base = "scenario: TN99\ntarget: { at_enu: [1, 2] }\n"
    path.write_text(base + "wind: 3\n", encoding="utf-8")
    with pytest.raises(iw.WorldError, match="wind"):
        iw.load("TN99", 1, path)
    path.write_text(base + "laser: lidar\n", encoding="utf-8")
    with pytest.raises(iw.WorldError, match="laser"):
        iw.load("TN99", 1, path)
    path.write_text(base + "start_alt_m: 10\n", encoding="utf-8")
    with pytest.raises(iw.WorldError, match="start_alt_m"):
        iw.load("TN99", 1, path)
    path.write_text("scenario: TN98\ntarget: { at_enu: [1, 2] }\n", encoding="utf-8")
    with pytest.raises(iw.WorldError, match="TN98"):
        iw.load("TN99", 1, path)
    path.write_text(base.replace("[1, 2]", "[1]"), encoding="utf-8")
    with pytest.raises(iw.WorldError, match="at_enu"):
        iw.load("TN99", 1, path)
    with pytest.raises(iw.WorldError, match="filer"):
        iw.scenario_path("TN99")


def test_the_cameras_are_the_sensors() -> None:
    main, tele = iw.CAMERAS
    assert (main.name, main.width_px, main.hfov_deg) == ("main", 4608, 66.0)
    assert main.focal_px == pytest.approx(3548, abs=1)
    assert tele.focal_px == pytest.approx(11501, abs=2)
    assert tele.half_view_deg < 8.0 < 19.0 < main.half_view_deg < 21.0


def test_the_oracle_sees_bigger_lower_and_nothing_out_of_view() -> None:
    w = iw.load("TN01", 1)
    oracle = iw.Oracle(w)
    main, tele = oracle.cameras
    over = (w.target.east, w.target.north)
    high = oracle.px(tele, (*over, 50.0), w.target)
    low = oracle.px(tele, (*over, 21.0), w.target)
    assert high is not None and low is not None
    assert high < CLEAR_PX <= low  # TN01: clear only below 30 m
    assert oracle.px(tele, (*over, 30.0), w.target) < CLEAR_PX  # type: ignore[operator]
    assert oracle.px(tele, (over[0] + 20.0, over[1], 30.0), w.target) is None
    # Under the detector's 32 px nothing is reported; the same key, the same noise.
    assert oracle.still(main, (*over, 50.0), (0, 0)) == []
    seen = oracle.still(tele, (*over, 21.0), (3, 1))
    assert [s.person.id for s in seen] == ["target"]
    assert seen == oracle.still(tele, (*over, 21.0), (3, 1))
    assert seen != oracle.still(tele, (*over, 21.0), (3, 0))
    assert math.dist((seen[0].east, seen[0].north), over) < 3.0


def test_the_report_rows(tmp_path: Path) -> None:
    m = InspectMetrics(
        min_clearance_m=19.5,
        min_target_horizontal_m=0.4,
        floor_breach_m=0.0,
        lowest_above_ground_m=20.6,
        inspect_duration_s=88.0,
        target_px_first_step={"tele": 46.0},
        target_px_last_step={"tele": 112.0, "main": 34.0},
        outcome="clear",
    )
    (tmp_path / "search_report.md").write_text("# Sökning\n\n## Annat\n\nrad\n", "utf-8")
    write_report(tmp_path, "TN01", 1, m)
    write_report(tmp_path, "TN01", 2, m)
    text = (tmp_path / "search_report.md").read_text("utf-8")
    assert text.startswith("# Sökning\n\n## Annat\n\nrad\n\n## Titta närmare\n")
    rows = [ln for ln in text.splitlines() if ln.startswith("| TN01")]
    assert rows == [
        f"| TN01 | {s} | clear | 19,5 | 0,0 | 0,4 | 20,6 | 88,0 | tele 46 | main 34, tele 112 |"
        for s in (1, 2)
    ]
    runs = json.loads((tmp_path / "inspect_metrics.json").read_text("utf-8"))
    assert [r["seed"] for r in runs] == [1, 2] and runs[0]["outcome"] == "clear"


def test_a_stand_the_map_has_short() -> None:
    w = iw.load("TN13", 1)
    (stand,) = w.stands
    (tree,) = stand.trees()
    assert tree.height == 20.0 and tree.map_height == 12.0
    scene = Scene((0.0, 0.0), 0.0, 100.0).with_trees(stand.trees())
    assert scene.trees[0].map_height == 12.0
