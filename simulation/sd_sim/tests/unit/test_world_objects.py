"""Objects with facit in the generated world (sd_sim.objects and world.render_world)."""

from __future__ import annotations

import json
import math
import xml.etree.ElementTree as ET
from pathlib import Path

import pytest

from sd_safety.targets import distance_m
from sd_sim import fixture, objects, paths, world

RED_BLOCK = paths.REPO / "simulation" / "scenarios" / "objects" / "red_block.yaml"


def _box(**kw: object) -> objects.Box:
    base: dict[str, object] = {
        "name": "b",
        "east_m": 0.0,
        "north_m": 0.0,
        "size_m": (1.0, 1.0, 1.0),
        "color": (1.0, 0.0, 0.0),
    }
    base.update(kw)
    return objects.Box(**base)  # type: ignore[arg-type]


def test_example_file_loads() -> None:
    placed = objects.load(RED_BLOCK)
    assert [o.name for o in placed] == ["red_block"]
    assert placed[0].label == 1 and placed[0].truth["is_target"] is True


def test_objects_are_in_the_world_with_label(tmp_path: Path) -> None:
    placed = objects.load(RED_BLOCK)
    path = world.render_world(world.WORLD_NAME, 0, tmp_path, placed)
    root = ET.parse(path).getroot()  # noqa: S314 - our own rendered file
    model = root.find("world/model[@name='red_block']")
    assert model is not None
    x, y, z, *_ = (float(v) for v in (model.findtext("pose") or "").split())
    assert (x, y, z) == (10.0, 200.0, 0.0)
    assert model.findtext("plugin[@name='gz::sim::systems::Label']/label") == "1"
    assert model.findtext("link/visual/geometry/box/size") == "1 1 0.5"
    # The vehicle is still the last include and the world has no remote URIs.
    assert [u.text for u in root.iter("uri")] == ["model://sd_quad14_i0"]


def test_truth_file_matches_the_world(tmp_path: Path) -> None:
    placed = [*objects.load(RED_BLOCK), _box(name="decoy", east_m=-30.0, north_m=40.0, up_m=2.0)]
    world.render_world(world.WORLD_NAME, 3, tmp_path, placed)
    truth = json.loads(
        (tmp_path / "build" / "sim" / "worlds" / f"{world.WORLD_NAME}_i3.truth.json").read_text()
    )
    home = fixture.load().home
    assert truth["world"] == world.WORLD_NAME and truth["home"]["lat"] == home.lat
    rows = {r["name"]: r for r in truth["objects"]}
    assert rows["red_block"]["truth"] == {"is_target": True, "kind": "test_block"}
    # The WGS84 position is the world position: 10 m east, 200 m north of home.
    block = rows["red_block"]
    assert (
        abs(distance_m(home.lat, home.lon, block["lat"], block["lon"]) - math.hypot(10, 200)) < 0.05
    )
    assert rows["decoy"]["alt_amsl_m"] == home.alt_amsl_m + 2.0


def test_world_without_objects_is_unchanged(tmp_path: Path) -> None:
    assert world.world_sdf(world.WORLD_NAME, 0, placed=()) == world.world_sdf(world.WORLD_NAME, 0)
    world.render_world(world.WORLD_NAME, 0, tmp_path)
    truth = json.loads((tmp_path / "build/sim/worlds/sd_lake_forest_i0.truth.json").read_text())
    assert truth["objects"] == []


def test_yaw_is_a_compass_heading() -> None:
    assert world.object_pose(_box(yaw_deg=90.0)).yaw_rad == pytest.approx(0.0)
    assert world.object_pose(_box(yaw_deg=0.0)).yaw_rad == pytest.approx(math.pi / 2)


@pytest.mark.parametrize(
    ("placed", "message"),
    [
        ([_box(name="a"), _box(name="a")], "två gånger"),
        ([_box(name="ground")], "används redan"),
        ([_box(name="sd_quad14_i0")], "används redan"),
        ([_box(name='x"/><model name="y')], "bokstäver"),
        ([_box(east_m=math.nan)], "ändliga"),
    ],
)
def test_bad_objects_are_refused(placed: list[objects.Box], message: str) -> None:
    with pytest.raises(objects.ObjectError, match=message):
        world.world_sdf(world.WORLD_NAME, 0, placed=placed)


@pytest.mark.parametrize(
    ("data", "message"),
    [
        ({"name": "a"}, "lista"),
        ([{"name": "a", "kind": "person"}], "saknar"),
        (
            [
                {
                    "name": "a",
                    "kind": "person",
                    "east_m": 0,
                    "north_m": 0,
                    "size_m": [1, 1, 1],
                    "color": [1, 0, 0],
                }
            ],
            "kind ska vara box",
        ),
        (
            [
                {
                    "name": "a",
                    "kind": "box",
                    "east_m": 0,
                    "north_m": 0,
                    "size_m": [1, 1, 1],
                    "color": [2, 0, 0],
                }
            ],
            "mellan 0 och 1",
        ),
        (
            [
                {
                    "name": "a",
                    "kind": "box",
                    "east_m": 0,
                    "north_m": 0,
                    "size_m": [1, 1, 1],
                    "color": [1, 0, 0],
                    "colour": 1,
                }
            ],
            "okända",
        ),
    ],
)
def test_bad_object_files_are_refused(data: object, message: str) -> None:
    with pytest.raises(objects.ObjectError, match=message):
        objects.parse(data)
