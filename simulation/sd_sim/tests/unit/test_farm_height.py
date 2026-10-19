"""The farm round's height clears the tallest tree under and beside its flight."""

from __future__ import annotations

import importlib.util
import shutil
import subprocess
from pathlib import Path
from typing import Any

import numpy as np
import pytest
import shapely
import yaml

from sd_sim import fixture, paths, site, site_world, terrain, world
from sd_sim.farm import gazebo as farm_gazebo
from sd_sim.farm import height
from sd_sim.farm import round as farm_round

_spec = importlib.util.spec_from_file_location(
    "site_world_tests", Path(__file__).with_name("test_site_world.py")
)
assert _spec is not None and _spec.loader is not None
_sw: Any = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(_sw)

SEEDS = range(60)


def _tops(trees: list[world.Tree], area: shapely.Geometry) -> list[float]:
    """Every tree top whose crown reaches the flight area (sd_lake_forest is flat)."""
    return [
        t.height
        for t in trees
        if shapely.distance(area, shapely.Point(t.x, t.y)) <= height.LAKE_FOREST_CROWN * t.height
    ]


def test_no_round_is_planned_under_a_tree_top(monkeypatch: pytest.MonkeyPatch) -> None:
    fix = fixture.load()
    trees = [
        *world.trees(),
        world.Tree("on_the_way", 0.0, 100.0, 45.0),  # on the line from home to the box
        world.Tree("beside_the_box", 70.0, 200.0, 38.0),  # 20 m east of the box, crown 5.7 m
    ]
    monkeypatch.setattr(world, "trees", lambda: trees)
    tops = _tops(trees, height.flight_area(fix))
    assert max(tops) == 45.0
    drawn = []
    for seed in SEEDS:
        sc = height.draw(seed, [world.WORLD_NAME], fix)
        drawn.append(sc.extra["height_drawn_m"])
        assert sc.height_m >= max(tops) + height.CLEARANCE_M, seed
        assert sc.height_m <= fix.charters[height.CHARTER].alt_max_agl_m
        assert sc.as_dict()["height_m"] == sc.height_m
    assert min(drawn) < max(tops)  # the draw alone would have flown into the tree


def test_the_area_holds_a_straight_return_home_from_the_box(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    # The survey ends at a line end such as (35, 150); RTL flies straight home from there,
    # beside the way out. A tree in that wedge counts.
    fix = fixture.load()
    wedge = world.Tree("wedge", 25.0, 100.0, 45.0)
    assert shapely.distance(shapely.LineString([(0, 0), (0, 200)]), shapely.Point(25, 100)) > 15
    assert shapely.contains_xy(height.flight_area(fix), 17.5, 75.0)  # on the RTL line
    monkeypatch.setattr(world, "trees", lambda: [wedge])
    assert height.draw(1, [world.WORLD_NAME], fix).height_m >= 45.0 + height.CLEARANCE_M


def test_the_charter_keeps_the_ai_at_the_round_s_height() -> None:
    assert height.charter_floor(75.0) == {"alt_min_agl_m": 75.0, "default_alt_agl_m": 75.0}


def test_no_lake_forest_tree_reaches_the_charter_the_ai_may_fly() -> None:
    # sd-brain may fly anywhere in charter_test, not only the flight area: in the only
    # world the farm flies, no crown reaches the charter or its geofence buffer.
    charter = fixture.load().charters[height.CHARTER]
    reach = shapely.buffer(shapely.Polygon(charter.area_enu), charter.geofence_buffer_m)
    assert _tops(world.trees(), reach) == []


def test_today_s_lake_forest_rounds_keep_their_drawn_height() -> None:
    # sd_lake_forest's own trees stand well west of the demo's flight.
    fix = fixture.load()
    assert _tops(world.trees(), height.flight_area(fix)) == []
    for seed in SEEDS:
        sc = height.draw(seed, [world.WORLD_NAME], fix)
        assert sc.height_m == sc.extra["height_drawn_m"]


def test_flown_height_is_whole_steps_over_the_clearance() -> None:
    assert height.flown_m(25.0, 0.0, 120.0) == 25.0
    assert height.flown_m(60.0, 30.0, 120.0) == 60.0
    assert height.flown_m(25.0, 31.2, 120.0) == 60.0  # 56.2 up to whole 5 m
    assert height.flown_m(25.0, 90.0, 120.0) == 115.0
    with pytest.raises(farm_round.SkippedError):
        height.flown_m(25.0, 95.0, 120.0)  # the charter's lowest height would be its ceiling


def test_a_round_that_cannot_clear_the_trees_is_skipped(monkeypatch: pytest.MonkeyPatch) -> None:
    fix = fixture.load()
    monkeypatch.setattr(world, "trees", lambda: [world.Tree("mast", 0.0, 220.0, 97.3)])
    with pytest.raises(farm_round.SkippedError) as err:
        height.draw(1, [world.WORLD_NAME], fix)
    assert "97" not in str(err.value)  # the reason names no tree height
    assert "120" in str(err.value)


def test_a_skipped_round_writes_its_reason_and_exits_3(tmp_path: Path) -> None:
    class TooHigh:
        def run(self, spec: farm_round.RoundSpec, work: Path) -> farm_round.RawRound:
            raise farm_round.SkippedError("varvet hoppas över: för högt")

    path = tmp_path / "round" / "spec.json"
    path.parent.mkdir()
    farm_round.RoundSpec.new(3, "script", "0123456789").write(path)
    assert farm_round.execute(path, TooHigh()) == farm_round.EXIT_ERROR
    result = farm_round.read_result(path.parent)
    assert result is not None and result["status"] == "skipped"
    assert "för högt" in result["error"]


def test_a_site_round_clears_the_real_trees_under_its_own_fixture(tmp_path: Path) -> None:
    _sw.make_site(tmp_path)
    shutil.copytree(paths.REPO / "simulation" / "scenarios", tmp_path / "simulation" / "scenarios")
    trees = tmp_path / "build" / "trees" / "testsite"
    trees.mkdir(parents=True)
    # A 40 m tree on the site fixture's line from home to its search box, one beside it.
    (trees / "trees.csv").write_text(
        "east_m,north_m,height_m,species\n0.0,60.0,40.0,deciduous\n-70.0,-15.0,30.0,deciduous\n",
        encoding="ascii",
    )
    np.save(trees / "chm_dm.npy", np.zeros((200, 200), np.uint16))
    s = site.load("testsite", tmp_path)
    ground = terrain.ground_grid(s)
    own = fixture.parse(yaml.safe_load(site_world.fixture_yaml(s, ground, tmp_path)))
    tree_top = float(ground.at(np.array([0.0]), np.array([60.0]))[0]) + 40.0
    assert shapely.contains_xy(height.flight_area(own), 0.0, 60.0)
    top, ceiling = height.clear_of("sd_testsite", fixture.load(), tmp_path)
    assert top == pytest.approx(tree_top) and ceiling == 120.0
    for seed in range(10):
        sc = height.draw(seed, ["sd_testsite"], repo=tmp_path)
        assert sc.height_m >= tree_top + height.CLEARANCE_M, seed


def test_an_unknown_world_is_an_error(tmp_path: Path) -> None:
    with pytest.raises(farm_round.RoundError, match="okänd värld"):
        height.clear_of("sd_nowhere", fixture.load(), tmp_path)


def test_the_runner_starts_nothing_for_a_skipped_round(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    monkeypatch.setattr(world, "trees", lambda: [world.Tree("mast", 0.0, 220.0, 97.3)])
    monkeypatch.setattr(farm_gazebo.GazeboRunner, "check", lambda self: None)
    calls: list[Any] = []
    path = tmp_path / "round" / "spec.json"
    path.parent.mkdir()
    farm_round.RoundSpec.new(3, "script", "0123456789").write(path)

    def run(argv: list[str], **kw: Any) -> subprocess.CompletedProcess[str]:
        calls.append(argv)
        return subprocess.CompletedProcess(argv, 0, "", "")

    runner = farm_gazebo.GazeboRunner(run=run, repo=tmp_path)
    assert farm_round.execute(path, runner) == farm_round.EXIT_ERROR
    result = farm_round.read_result(path.parent)
    assert result is not None and result["status"] == "skipped" and calls == []
