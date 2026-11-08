"""The simulated world's solids: the truth the laser, the map pack and the clearance use."""

from __future__ import annotations

import math
import shutil
import xml.etree.ElementTree as ET
from pathlib import Path

import numpy as np
import pytest
from shapely.geometry import Polygon

from sd_sim import objects, obstacles, paths, site, terrain, world
from sd_sim.obstacles import Ground, LineSpec, PowerLine, Scene, SimTree, TreeStand

HOME = (-35.363261, 149.165230)


def _scene(
    *trees: SimTree, lines: tuple[LineSpec, ...] = (), ground: Ground | None = None
) -> Scene:
    return Scene(HOME, 584.0, 300.0, ground or Ground(), trees, lines)


def test_lake_forest_trees_are_the_world_files_cylinders() -> None:
    """Every crown and trunk the laser can hit is where and as big as Gazebo draws it."""
    sdf = world.world_sdf(world.WORLD_NAME, 0)
    root = ET.fromstring(sdf)  # noqa: S314 - our own template
    link = next(el for el in root.iter("link") if el.get("name") == "tree_000")
    drawn = []
    for el in link:
        if el.tag in ("visual", "collision"):
            zc = float((el.findtext("pose") or "").split()[2])
            cyl = el.find("geometry/cylinder")
            assert cyl is not None
            r, ln = float(cyl.findtext("radius") or 0), float(cyl.findtext("length") or 0)
            drawn.append((round(zc - ln / 2, 2), round(r, 2), round(zc + ln / 2, 2)))
    tree = world.trees()[0]
    ours = {
        (round(z0, 2), round(r0, 2), round(z1, 2))
        for z0, r0, z1, _ in SimTree(0, 0, tree.height).parts()
    }
    assert ours == set(drawn)


def test_a_beam_straight_down_meets_the_crown_top_the_ground_and_the_wire() -> None:
    tree = SimTree(50.0, 50.0, 24.0)
    line = LineSpec((100.0, 0.0), (100.0, 100.0), 20.0)
    scene = _scene(tree, lines=(line,))
    hit = scene.ray((50.0, 50.0, 60.0), (0, 0, -1), 250)
    assert hit is not None and hit.kind == "tree" and hit.distance == pytest.approx(36.0, abs=0.01)
    # Beside the trunk but under the lower crown (radius 3.6 m).
    hit = scene.ray((53.0, 50.0, 60.0), (0, 0, -1), 250)
    assert hit is not None and hit.distance == pytest.approx(60.0 - 0.65 * 24, abs=0.01)
    hit = scene.ray((60.0, 50.0, 60.0), (0, 0, -1), 250)  # a gap: the ground
    assert (
        hit is not None and hit.kind == "ground" and hit.distance == pytest.approx(60.0, abs=0.01)
    )
    hit = scene.ray((100.0, 50.0, 60.0), (0, 0, -1), 250)
    assert hit is not None and hit.kind == "line"
    assert hit.distance == pytest.approx(60.0 - 20.0 - obstacles.WIRE_RADIUS_M, abs=0.01)
    assert scene.ray((60.0, 50.0, 60.0), (0, 0, -1), 50) is None  # out of reach


def test_a_tilted_beam_measures_along_the_slant_and_can_see_through_the_crowns() -> None:
    scene = _scene(SimTree(50.0, 50.0, 24.0))
    tilt = math.radians(20)
    hit = scene.ray((-50.0, 0.0, 40.0), (math.sin(tilt), 0, -math.cos(tilt)), 250)
    assert hit is not None and hit.distance == pytest.approx(40 / math.cos(tilt), abs=0.01)
    hit = scene.ray((50.0, 50.0, 60.0), (0, 0, -1), 250, crowns=False)
    assert hit is not None and hit.kind == "ground"


def test_the_beam_follows_sloping_ground() -> None:
    xs = np.arange(-100.0, 101.0, 5.0)
    east, _ = np.meshgrid(xs, xs[::-1])
    ground = Ground(5.0, 100.0, 0.2 * east)  # rises 0.2 m per metre to the east
    scene = _scene(ground=ground)
    hit = scene.ray((50.0, 0.0, 40.0), (0, 0, -1), 250)
    assert hit is not None and hit.distance == pytest.approx(30.0, abs=0.01)
    assert scene.clearance(50.0, 0.0, 40.0) == pytest.approx(30.0 / math.hypot(1, 0.2), abs=0.01)


def test_clearance_and_top() -> None:
    tree = SimTree(0.0, 50.0, 20.0, "spruce")
    scene = _scene(tree, lines=(LineSpec((40.0, 0.0), (40.0, 100.0), 15.0),))
    assert scene.clearance(0.0, 50.0, 26.0) == pytest.approx(6.0, abs=0.01)
    assert scene.clearance(0.0, 50.0, 10.0) < 0  # inside the crown
    assert scene.clearance(40.0, 50.0, 18.0) == pytest.approx(
        3.0 - obstacles.WIRE_RADIUS_M, abs=0.01
    )
    assert scene.top(0.0, 50.0) == pytest.approx(20.0)
    assert scene.top(10.0, 50.0) == pytest.approx(0.0)  # past the crown (radius 3.8 m)
    assert scene.top(10.0, 50.0, radius=10.0) == pytest.approx(20.0)
    assert scene.top(40.0, 50.0) == pytest.approx(15.0 + obstacles.WIRE_RADIUS_M)
    assert scene.top(40.0, 50.0, lines=False) == 0.0


def test_a_scene_survives_its_folder(tmp_path: Path) -> None:
    xs = np.arange(-100.0, 101.0, 5.0)
    east, north = np.meshgrid(xs, xs[::-1])
    scene = Scene(
        HOME,
        12.5,
        100.0,
        Ground(5.0, 100.0, 0.1 * east + 0.05 * north),
        name="sd_test",
    ).with_trees([SimTree(10.0, 20.0, 18.0, "pine", map_height=0.0)])
    scene = scene.with_lines([LineSpec((0.0, 30.0), (60.0, 30.0), 12.0, "l1")])
    scene = scene.with_areas(water=[Polygon([(50, 50), (80, 50), (80, 80)])])
    back = Scene.load(scene.save(tmp_path / "s").parent)
    assert back.trees == scene.trees and back.trees[0].base == pytest.approx(2.0)
    assert back.lines == scene.lines and back.name == "sd_test"
    assert back.is_water(70.0, 55.0) and not back.is_water(55.0, 70.0)
    assert back.clearance(0.0, 0.0, 20.0) == pytest.approx(scene.clearance(0.0, 0.0, 20.0))
    with pytest.raises(ValueError):
        (tmp_path / "s" / "scene.json").write_text('{"format": 9}')
        Scene.load(tmp_path / "s")


def test_stand_and_power_line_are_world_objects_gazebo_accepts() -> None:
    stand = TreeStand("stand_a", 100.0, 80.0, 20.0, 24.0, seed=3)
    line = PowerLine("line_a", (120.0, 0.0), (120.0, 90.0), 20.0)
    objects.check([stand, line], world.GROUND_SIZE_M / 2)
    for obj in (stand, line):
        world.checked_fragment(obj)
    trees = stand.trees()
    assert trees == TreeStand("stand_a", 100.0, 80.0, 20.0, 24.0, seed=3).trees()
    assert all(math.hypot(t.east - 100, t.north - 80) <= 20 for t in trees)
    assert len(trees) > 20
    scene = obstacles.lake_forest(
        HOME, 584.0, [stand, line, objects.Box("b", 30, 30, (1, 1, 1), (1, 0, 0))]
    )
    assert len(scene.trees) == len(world.trees()) + len(trees)
    assert scene.lines == (LineSpec((120.0, 0.0), (120.0, 90.0), 20.0, "line_a"),)
    assert scene.is_water(*world.LAKE_CENTER)


@pytest.mark.parametrize(
    ("kwargs", "ok"),
    [
        ({"height_m": 4.0}, False),
        ({"height_m": 41.0}, False),
        ({"b": (120.0, 0.5)}, False),
        ({}, True),
    ],
)
def test_power_line_limits(kwargs: dict[str, object], ok: bool) -> None:
    args: dict[str, object] = {"name": "l", "a": (120.0, 0.0), "b": (120.0, 90.0)} | kwargs
    if ok:
        PowerLine(**args)  # type: ignore[arg-type]
    else:
        with pytest.raises(objects.ObjectError):
            PowerLine(**args)  # type: ignore[arg-type]


def test_for_world_knows_the_flat_world_and_refuses_others() -> None:
    scene = obstacles.for_world(world.WORLD_NAME, "-35.0,149.0,600.0,0")
    assert scene.origin == (-35.0, 149.0) and scene.home_alt == 600.0
    with pytest.raises(ValueError):
        obstacles.for_world("sd_nowhere", "-35.0,149.0,600.0,0")


def _site_repo(tmp_path: Path) -> Path:
    src = paths.REPO / "simulation" / "gazebo" / "sites" / "hellasgarden"
    shutil.copytree(src, tmp_path / "simulation" / "gazebo" / "sites" / "hellasgarden")
    return tmp_path


def test_a_site_scene_has_the_worlds_generated_forest_on_its_ground(tmp_path: Path) -> None:
    repo = _site_repo(tmp_path)
    scene = obstacles.for_world("sd_hellasgarden", "0,0,0,0", repo=repo)
    loaded = site.load("hellasgarden", repo)
    grid = terrain.ground_grid(loaded)
    expected = terrain.site_trees(loaded, grid)
    assert len(scene.trees) == len(expected) > 100
    t, e = scene.trees[0], expected[0]
    assert (t.east, t.north, t.height, t.base) == (e.east, e.north, e.height, e.z)
    assert scene.home_alt == loaded.home_alt and scene.origin == (loaded.lat, loaded.lon)
    assert scene.ground.at1(0.0, 0.0) == pytest.approx(0.0, abs=0.05)  # the pad at home
    hit = scene.ray((t.east, t.north, t.base + t.height + 30.0), (0, 0, -1), 250)
    assert hit is not None and hit.kind == "tree" and hit.distance == pytest.approx(30.0, abs=0.01)


def test_a_site_scene_uses_the_real_trees_once_fetched(tmp_path: Path) -> None:
    repo = _site_repo(tmp_path)
    folder = repo / "build" / "trees" / "hellasgarden"
    folder.mkdir(parents=True)
    (folder / "trees.csv").write_text(
        "east_m,north_m,height_m,species\n100.0,-200.0,22.5,spruce\n-150.0,120.0,18.0,deciduous\n",
        encoding="ascii",
    )
    scene = obstacles.site_scene("hellasgarden", repo=repo)
    shapes = {(t.east, t.north): (t.shape, t.height) for t in scene.trees}
    assert shapes == {(100.0, -200.0): ("spruce", 22.5), (-150.0, 120.0): ("birch", 18.0)}
