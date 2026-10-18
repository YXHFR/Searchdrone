"""Scenario objects in the Gazebo world: seeded layout, labelled models, ground truth."""

from __future__ import annotations

import json
import math
import xml.etree.ElementTree as ET
from itertools import combinations
from pathlib import Path

import pytest

from sd_detect import default_registry
from sd_safety.targets import Plane
from sd_sim import fixture, render_stills, targets, world


def test_layout_is_seeded() -> None:
    assert targets.layout("SK-RED", 3) == targets.layout("SK-RED", 3)
    assert targets.layout("SK-RED", 3) != targets.layout("SK-RED", 4)


@pytest.mark.parametrize("seed", range(1, 41))
def test_layout_rules(seed: int) -> None:
    objs = targets.layout("SK-RED", seed)
    (x0, y0), _, (x1, y1), _ = fixture.load().survey_box.corners()
    blocks = [t for t in objs if t.cls == "red_block"]
    assert 1 <= len(blocks) <= 3
    assert all(t.role == "target" for t in blocks)
    roles = {(t.cls, t.role) for t in objs}
    assert {
        ("car", "decoy"),
        ("car", "context"),
        ("building", "decoy"),
        ("road", "context"),
    } <= roles
    red_car = next(t for t in objs if t.name == "red_car_0")
    assert red_car.rgb == targets.RED  # the decoy has the targets' colour
    road = next(t for t in objs if t.cls == "road")
    things = [t for t in objs if t.cls != "road"]
    for t in things:
        assert x0 + targets.MARGIN_M <= t.east_m <= x1 - targets.MARGIN_M
        assert y0 + targets.MARGIN_M <= t.north_m <= y1 - targets.MARGIN_M
    for a, b in combinations(things, 2):
        assert math.hypot(a.east_m - b.east_m, a.north_m - b.north_m) >= targets.MIN_GAP_M
    for b in blocks:
        assert abs(b.north_m - road.north_m) >= targets.ROAD_CLEAR_M


def test_every_count_of_blocks_occurs() -> None:
    counts = {sum(t.cls == "red_block" for t in targets.layout("SK-RED", s)) for s in range(1, 20)}
    assert counts == {1, 2, 3}


def test_unknown_scenario() -> None:
    with pytest.raises(ValueError, match="okänt scenario"):
        targets.layout("SK-BLUE", 1)


def test_rendered_models_carry_their_class_label(tmp_path: Path) -> None:
    names, truth_path = targets.render_targets(0, "SK-RED", 11, tmp_path)
    labels = {spec.id: spec.params["gz_label"] for spec in default_registry()}
    objs = {f"sd_targets_i0_{t.name}": t for t in targets.layout("SK-RED", 11)}
    assert set(names) == set(objs)
    for name in names:
        model_dir = tmp_path / "build" / "sim" / "models" / name
        root = ET.parse(model_dir / "model.sdf").getroot()  # noqa: S314 - our own file
        model = root.find("model")
        assert model is not None and model.get("name") == name
        assert model.findtext("static") == "true"
        label = model.find("plugin[@name='gz::sim::systems::Label']")
        assert label is not None
        assert int(label.findtext("label") or 0) == labels[objs[name].cls]
        assert f"<name>{name}</name>" in (model_dir / "model.config").read_text(encoding="utf-8")
    truth = json.loads(truth_path.read_text(encoding="utf-8"))
    home = fixture.load().home
    plane = Plane(home.lat, home.lon)
    assert truth["scenario"] == "SK-RED" and truth["seed"] == 11
    for o in truth["objects"]:
        e, n = plane.xy(o["lat"], o["lon"])
        assert (e, n) == pytest.approx((o["east_m"], o["north_m"]), abs=0.01)
    # A new seed replaces the old objects instead of piling up.
    targets.render_targets(0, "SK-RED", 1, tmp_path)
    left = sorted(p.name for p in (tmp_path / "build" / "sim" / "models").glob("sd_targets_i0_*"))
    assert left == sorted(f"sd_targets_i0_{t.name}" for t in targets.layout("SK-RED", 1))


def test_world_includes_the_objects() -> None:
    names = ("sd_targets_i0_red_block_0", "sd_targets_i0_road_0")
    root = ET.fromstring(world.world_sdf(world.WORLD_NAME, 0, includes=names))  # noqa: S314
    uris = [u.text for u in root.iter("uri")]
    assert uris == ["model://sd_quad14_i0", *(f"model://{n}" for n in names)]


def test_still_rig_and_shot_poses() -> None:
    sdf = ET.fromstring(render_stills._rig_sdf(3))  # noqa: S314 - our own template
    sensors = {s.get("name"): s for s in sdf.iter("sensor")}
    assert set(sensors) == {"stills", "boxes"}
    assert sensors["boxes"].get("type") == "boundingbox_camera"
    for s in sensors.values():
        assert s.findtext("camera/image/width") == "1280"
        assert s.findtext("topic", "").startswith("/sd_rig_i3/")
    poses = render_stills.shot_poses(targets.layout("SK-RED", 11))
    assert len({p["name"] for p in poses}) == len(poses)
    tilted = next(p for p in poses if p["name"] == "block0_tilted")
    # Gazebo's front-left-up pitch is the vehicle's nose-up pitch with the sign flipped.
    assert tilted["gz"]["pitch"] == pytest.approx(-math.radians(tilted["attitude"]["pitch_deg"]))
    assert tilted["gz"]["yaw"] == pytest.approx(math.radians(90.0 - tilted["attitude"]["yaw_deg"]))
