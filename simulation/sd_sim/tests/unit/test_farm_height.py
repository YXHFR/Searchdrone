"""The farm round's height clears the tallest tree under and beside its flight."""

from __future__ import annotations

import importlib.util
import shutil
from pathlib import Path
from typing import Any

import numpy as np
import pytest
import shapely
import yaml

from sd_sim import fixture, paths, site, site_world, terrain, world
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
    assert height.flown_m(25.0, 95.0, 120.0) == 120.0  # exactly the ceiling


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


def test_a_site_round_clears_the_real_trees(tmp_path: Path) -> None:
    _sw.make_site(tmp_path)
    shutil.copytree(paths.REPO / "simulation" / "scenarios", tmp_path / "simulation" / "scenarios")
    trees = tmp_path / "build" / "trees" / "testsite"
    trees.mkdir(parents=True)
    (trees / "trees.csv").write_text(
        "east_m,north_m,height_m,species\n-50.0,60.0,21.5,spruce\n-70.0,-15.0,30.0,deciduous\n",
        encoding="ascii",
    )
    np.save(trees / "chm_dm.npy", np.zeros((200, 200), np.uint16))
    # The demo's fixture with its search box moved over the test site's forest.
    data = yaml.safe_load((paths.SIMULATION / "scenarios" / "demo_fixture.yaml").read_text())
    for pin in data["pins"]:
        if pin["id"] == "P1":
            pin["east_m"], pin["north_m"] = -50.0, 30.0
    data["survey_box"]["size_m"] = [60.0, 60.0]
    fix = fixture.parse(data)
    s = site.load("testsite", tmp_path)
    ground = terrain.ground_grid(s)
    tree_top = float(ground.at(np.array([-50.0]), np.array([60.0]))[0]) + 21.5
    area = height.flight_area(fix)
    assert height.obstacle_top("sd_testsite", area, tmp_path) == pytest.approx(
        site_world.tallest(s, ground, area)
    )
    for seed in range(10):
        sc = height.draw(seed, ["sd_testsite"], fix, tmp_path)
        assert sc.height_m >= tree_top + height.CLEARANCE_M, seed


def test_an_unknown_world_is_an_error(tmp_path: Path) -> None:
    with pytest.raises(farm_round.RoundError, match="okänd värld"):
        height.obstacle_top("sd_nowhere", shapely.box(0, 0, 1, 1), tmp_path)
