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


def test_objects_are_world_objects_with_their_class_label(tmp_path: Path) -> None:
    """The layout goes into the world through the world-object hook: one static model per
    object at world level (the bounding-box camera boxes each one), its class's label,
    and the facit file with every object's place and class."""
    objs = targets.layout("SK-RED", 11)
    world.render_world(world.WORLD_NAME, 0, tmp_path, placed=objs)
    labels = {spec.id: spec.params["gz_label"] for spec in default_registry()}
    root = ET.parse(tmp_path / "build" / "sim" / "worlds" / f"{world.WORLD_NAME}_i0.sdf")  # noqa: S314 - our own file
    models = {m.get("name"): m for m in root.getroot().iter("model")}
    for t in objs:
        model = models[t.name]
        assert model.findtext("static") == "true"
        label = model.find("plugin[@name='gz::sim::systems::Label']")
        assert label is not None and int(label.findtext("label") or 0) == labels[t.cls]
        # The box stands on the ground: its centre is half its height above the origin.
        body = model.find("link/visual[@name='body']")
        assert body is not None
        assert float((body.findtext("pose") or "").split()[2]) == pytest.approx(t.size_m[2] / 2)
    truth = json.loads(world.truth_path(world.WORLD_NAME, 0, tmp_path).read_text(encoding="utf-8"))
    rows = {o["name"]: o for o in truth["objects"]}
    home = fixture.load().home
    plane = Plane(home.lat, home.lon)
    for t in objs:
        row = rows[t.name]
        assert row["label"] == labels[t.cls] and row["truth"]["cls"] == t.cls
        assert row["truth"]["role"] == t.role
        e, n = plane.xy(row["lat"], row["lon"])
        assert (e, n) == pytest.approx((t.east_m, t.north_m), abs=0.01)


def test_still_rig_and_shot_poses() -> None:
    rig = render_stills.StillRig(3)
    sdf = ET.fromstring(world.checked_fragment(rig))  # noqa: S314 - our own template
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
