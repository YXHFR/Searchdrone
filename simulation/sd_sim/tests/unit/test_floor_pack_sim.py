"""Map packs from the simulated world: the drone's own loader and floor read them, the
map is the world unless a test puts an error in, and the errors land where asked."""

from __future__ import annotations

import json
import math
from datetime import date
from pathlib import Path

import numpy as np
import pytest

from sd_safety.floor import floor_at
from sd_safety.floor_pack import CHM_UNKNOWN_DM, FloorPack, load_floor_pack
from sd_safety.targets import Plane
from sd_sim import floor_pack
from sd_sim.floor_pack import MapErrors
from sd_sim.obstacles import Ground, LineSpec, PowerLine, Scene, SimTree, TreeStand, lake_forest

HOME = (-35.363261, 149.165230)
HOME_ALT = 584.0
TODAY = date(2026, 11, 3)
YEAR = TODAY.year
TREE = SimTree(40.3, 60.7, 24.0)


def _scene(*trees: SimTree, lines: tuple[LineSpec, ...] = ()) -> Scene:
    return Scene(HOME, HOME_ALT, 100.0, Ground(), trees, lines, name="sd_test")


def _floor(pack: FloorPack, east: float, north: float, start: float = HOME_ALT + 80.0):
    lat, lon = Plane(*HOME).latlon(east, north)
    return floor_at(pack, lat, lon, flight_date=TODAY, laser=None, start_alt_amsl_m=start)


def _cell(pack: FloorPack, east: float, north: float) -> tuple[int, int]:
    rows, cols = pack.chm_dm.shape
    return math.floor(rows / 2 - north), math.floor(east + cols / 2)


def _build(tmp_path: Path, scene: Scene, errors: MapErrors | None = None) -> FloorPack:
    errors = errors or MapErrors(scan_year=YEAR)
    return floor_pack.build(scene, tmp_path / "pack", name="sim-test", errors=errors)


def test_the_drone_loads_the_pack_and_it_is_the_world(tmp_path: Path) -> None:
    pack = _build(tmp_path, _scene(TREE))
    assert load_floor_pack(tmp_path / "pack").sha256 == pack.sha256
    assert pack.dtm_cm.shape == (200, 200) and pack.origin.lat == HOME[0]
    assert pack.dtm_cm.min() == pack.dtm_cm.max() == round(HOME_ALT * 100)
    assert int(pack.chm_dm[_cell(pack, TREE.east, TREE.north)]) == 240  # the trunk's top
    # The lower crown (radius 3.6 m, top 0.65 h) reaches the cell 3 m east, the upper
    # crown (1.92 m, 0.975 h) the one 1.5 m east.
    assert int(pack.chm_dm[_cell(pack, TREE.east + 3.0, TREE.north)]) == math.ceil(0.65 * 240)
    assert int(pack.chm_dm[_cell(pack, TREE.east + 1.5, TREE.north)]) == math.ceil(0.975 * 240)
    assert int(pack.chm_dm[_cell(pack, TREE.east + 5.0, TREE.north)]) == 0
    assert pack.scan_year.tolist() == [[YEAR]]


def test_the_floor_over_a_tree_and_over_the_meadow(tmp_path: Path) -> None:
    """With a fresh scan the floor is the world's tree + 2 m + 15 m, the meadow's 30 m."""
    pack = _build(tmp_path, _scene(TREE))
    over = _floor(pack, TREE.east, TREE.north)
    assert over.reason == "map" and over.tree_m == pytest.approx(26.0)
    assert over.floor_auto_amsl_m == pytest.approx(HOME_ALT + 24.0 + 2.0 + 15.0)
    meadow = _floor(pack, -50.0, -50.0)
    assert meadow.floor_auto_amsl_m == pytest.approx(HOME_ALT + 30.0)
    assert meadow.floor_lowest_amsl_m == pytest.approx(HOME_ALT + 2.0 + 15.0)


def test_map_errors(tmp_path: Path) -> None:
    trees = [SimTree(-60.0 + 6 * i, 0.0, 20.0) for i in range(20)]
    scene = _scene(*trees)
    low = _build(tmp_path, scene, MapErrors(tree_bias_m=8.0, scan_year=YEAR))
    assert int(low.chm_dm[_cell(low, trees[0].east, 0.0)]) == 120
    some = _build(tmp_path, scene, MapErrors(missing_trees=0.5, seed=1, scan_year=YEAR))
    left = sum(int(some.chm_dm[_cell(some, t.east, 0.0)]) > 0 for t in trees)
    assert 0 < left < len(trees)
    again = _build(tmp_path, scene, MapErrors(missing_trees=0.5, seed=1, scan_year=YEAR))
    assert np.array_equal(again.chm_dm, some.chm_dm)  # the same seed, the same map
    holes = _build(tmp_path, scene, MapErrors(holes=((30.0, 30.0, 4.0),), scan_year=YEAR))
    assert int(holes.chm_dm[_cell(holes, 30.0, 30.0)]) == CHM_UNKNOWN_DM
    assert int(holes.chm_dm[_cell(holes, 30.0, 36.0)]) == 0
    result = _floor(holes, 30.0, 30.0)
    assert result.reason == "unknown" and result.floor_auto_amsl_m == pytest.approx(
        HOME_ALT + 30 + 2 + 5 + 15  # unknown: 30 m forest, the growth margin in full
    )
    old = _build(tmp_path, scene, MapErrors(scan_year=2012))
    assert _floor(old, 30.0, 30.0).reason == "unknown"  # older than SCAN_MAX_AGE_Y
    unknown = _build(tmp_path, scene, MapErrors(scan_year=0))
    assert unknown.scan_year.tolist() == [[0]]


def test_one_tree_the_map_has_wrong(tmp_path: Path) -> None:
    """A tree 8 m taller than the map says, and one the map does not have at all."""
    taller = SimTree(20.0, 20.0, 24.0, map_height=16.0)
    missing = SimTree(-20.0, 20.0, 24.0, map_height=0.0)
    pack = _build(tmp_path, _scene(taller, missing))
    assert int(pack.chm_dm[_cell(pack, 20.0, 20.0)]) == 160
    assert int(pack.chm_dm[_cell(pack, -20.0, 20.0)]) == 0
    scene = _scene(taller, missing)
    assert scene.top(20.0, 20.0) == pytest.approx(24.0)  # the world keeps the real tree


def test_power_lines_go_to_the_pack_and_hold_the_floor(tmp_path: Path) -> None:
    line = PowerLine("line_a", (50.0, -40.0), (50.0, 40.0), 20.0)
    scene = lake_forest(HOME, HOME_ALT, [line])
    pack = floor_pack.build(
        scene, tmp_path / "p", name="sim-line", half_m=100.0, errors=MapErrors(scan_year=YEAR)
    )
    data = json.loads((tmp_path / "p" / "lines.geojson").read_text(encoding="utf-8"))
    assert [f["properties"]["name"] for f in data["features"]] == ["line_a"]
    assert len(pack.lines) == 1
    near = _floor(pack, 30.0, 0.0, start=HOME_ALT + 60.0)
    assert near.reason == "powerline" and near.floor_auto_amsl_m == pytest.approx(HOME_ALT + 60)
    assert _floor(pack, -10.0, 0.0).reason == "map"


def test_sloping_ground_and_a_stand(tmp_path: Path) -> None:
    xs = np.arange(-100.0, 101.0, 5.0)
    east, _ = np.meshgrid(xs, xs[::-1])
    scene = Scene(HOME, HOME_ALT, 100.0, Ground(5.0, 100.0, 0.1 * east), name="sd_slope")
    scene = scene.with_trees(TreeStand("s", 40.0, 0.0, 12.0, 24.0).trees())
    pack = _build(tmp_path, scene)
    assert int(pack.dtm_cm[_cell(pack, 60.5, -50.5)]) == round((HOME_ALT + 6.05) * 100)
    t = scene.trees[0]
    assert t.base == pytest.approx(0.1 * t.east)
    assert int(pack.chm_dm[_cell(pack, t.east, t.north)]) == pytest.approx(240, abs=2)


def test_refusals(tmp_path: Path) -> None:
    with pytest.raises(ValueError):
        floor_pack.build(_scene(), tmp_path / "p", name="Not A Name")
    for bad in (
        {"tree_bias_m": -1.0},
        {"missing_trees": 2.0},
        {"holes": ((0, 0, 0),)},
        {"scan_year": 12},
    ):
        with pytest.raises(ValueError):
            MapErrors(**bad)  # type: ignore[arg-type]


def test_command_line_writes_the_flat_worlds_pack(tmp_path: Path) -> None:
    code = floor_pack.main(
        [
            "--out",
            str(tmp_path / "lake"),
            "--half-m",
            "50",
            "--hole",
            "10,10,3",
            "--scan-year",
            "2020",
        ]
    )
    assert code == 0
    pack = load_floor_pack(tmp_path / "lake")
    assert pack.name == "sim-lake-forest" and pack.chm_dm.shape == (100, 100)
    assert int(pack.chm_dm[_cell(pack, 10.0, 10.0)]) == CHM_UNKNOWN_DM
    assert floor_pack.main(["--out", str(tmp_path / "x"), "--world", "sd_nowhere"]) == 2
