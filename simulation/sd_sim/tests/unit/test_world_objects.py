"""Objects with facit in the generated world (sd_sim.objects and world.render_world)."""

from __future__ import annotations

import json
import math
import xml.etree.ElementTree as ET
from collections.abc import Mapping
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import pytest

from sd_safety.targets import distance_m
from sd_sim import fixture, launch, objects, paths, world
from sd_sim.config import SimConfig
from sd_sim.errors import InfraError

RED_BLOCK = paths.REPO / "simulation" / "scenarios" / "objects" / "red_block.yaml"


def _box(**kw: object) -> objects.Box:
    base: dict[str, object] = {
        "name": "b",
        "east_m": 20.0,
        "north_m": 20.0,
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


def test_world_without_objects_has_no_object_block(tmp_path: Path) -> None:
    assert "Scenario objects" not in world.world_sdf(world.WORLD_NAME, 0)
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
        ([_box(east_m=3.0, north_m=0.0)], "närmare hem"),
        ([_box(east_m=5.2, north_m=0.0, size_m=(2.0, 2.0, 1.0))], "närmare hem"),
        ([_box(east_m=599.8)], "utanför världen"),
        ([_box(truth={"when": {1, 2}})], "JSON"),
    ],
)
def test_bad_objects_are_refused(placed: list[objects.Box], message: str) -> None:
    with pytest.raises(objects.ObjectError, match=message):
        world.world_sdf(world.WORLD_NAME, 0, placed=placed)


@pytest.mark.parametrize(
    ("kw", "message"),
    [({"size_m": (0.0, 1.0, 1.0)}, "size_m"), ({"color": (1.0, math.nan, 0.0)}, "color")],
)
def test_box_checks_its_own_values(kw: dict[str, object], message: str) -> None:
    with pytest.raises(objects.ObjectError, match=message):
        _box(**kw)


@dataclass(frozen=True)
class _Custom:
    name: str = "custom"
    east_m: float = 20.0
    north_m: float = 20.0
    up_m: float = 0.0
    yaw_deg: float = 0.0
    label: int = 0
    footprint_m: float = 0.5
    truth: Mapping[str, Any] = field(default_factory=dict)
    text: str = ""

    def sdf(self, pose: objects.Pose) -> str:
        return self.text.format(pose=pose.sdf())


@pytest.mark.parametrize(
    ("text", "message"),
    [
        ('<model name="custom"><pose>{pose}</pose>', "går inte att läsa"),
        ('<model name="other"><pose>{pose}</pose></model>', "objektets namn"),
        ('<light name="custom"/>', "objektets namn"),
        (
            '<model name="custom"><include><uri>https://fuel.example/m</uri></include></model>',
            "model://",
        ),
    ],
)
def test_custom_sdf_is_checked(text: str, message: str) -> None:
    with pytest.raises(objects.ObjectError, match=message):
        world.world_sdf(world.WORLD_NAME, 0, placed=[_Custom(text=text)])


def test_custom_object_is_placed() -> None:
    actor = _Custom(text='<actor name="custom"><pose>{pose}</pose></actor>')
    root = ET.fromstring(world.world_sdf(world.WORLD_NAME, 0, placed=[actor]))  # noqa: S314
    assert root.find("world/actor[@name='custom']") is not None


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


# --- the launcher ------------------------------------------------------------------------


def _stack(tmp_path: Path, **cfg: object) -> launch.Stack:
    return launch.Stack(SimConfig(**cfg), tmp_path / "run", repo=tmp_path)  # type: ignore[arg-type]


def test_launcher_loads_the_file(tmp_path: Path) -> None:
    (tmp_path / "o.yaml").write_text(RED_BLOCK.read_text(encoding="utf-8"), encoding="utf-8")
    assert [o.name for o in _stack(tmp_path, objects="o.yaml").world_objects()] == ["red_block"]
    assert _stack(tmp_path).world_objects() == []


@pytest.mark.parametrize(
    ("cfg", "message"),
    [
        ({"objects": "missing.yaml"}, "missing.yaml"),
        ({"objects": "bad.yaml"}, "lista"),
        ({"objects": "../outside.yaml"}, "i repot"),
        ({"objects": "bad.yaml", "physics": "builtin"}, "physics=gz"),
        ({"objects": "bad.yaml", "vehicle": "stock"}, "physics=gz"),
        ({"objects": "bad.yaml", "world": "other"}, "physics=gz"),
    ],
)
def test_launcher_refuses_objects(tmp_path: Path, cfg: dict[str, str], message: str) -> None:
    (tmp_path / "bad.yaml").write_text("{name: a}\n", encoding="utf-8")
    with pytest.raises(InfraError, match=message):
        _stack(tmp_path, **cfg).world_objects()


def test_clear_truth_removes_only_this_instance(tmp_path: Path) -> None:
    world.render_world(world.WORLD_NAME, 1, tmp_path)
    world.render_world(world.WORLD_NAME, 2, tmp_path)
    world.clear_truth(1, tmp_path)
    assert not world.truth_path(world.WORLD_NAME, 1, tmp_path).exists()
    assert world.truth_path(world.WORLD_NAME, 2, tmp_path).is_file()
