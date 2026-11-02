"""The photo farm (``just sim-farm-stills``): the draw, the people's sizes, the clothes, the
detailed ground and the camera's development. Gazebo is not started here."""

# ruff: noqa: S311 - seeded draws of scenery, not secrets
from __future__ import annotations

import math
import random
from dataclasses import replace
from pathlib import Path

import numpy as np
import pytest
import shapely

from sd_sim import objects
from sd_sim.farm import ground, labels, looks, people, stills
from sd_sim.farm.camera import project
from sd_sim.farm.people import BY_ID, Person


def _land() -> stills.Land:
    return stills.Land(
        ground=lambda e, n: 0.02 * e,
        trees=[],
        open_land=shapely.box(-300, -300, 300, 300),
        blocked=shapely.box(100, 100, 140, 140),
        paths=[shapely.LineString([(-200, -50), (200, -50)])],
    )


def _draw(seed: int, **kw: object) -> stills.Scene:
    return stills.draw(seed, "testsite", _land(), 59.3, **kw)  # type: ignore[arg-type]


def test_the_same_seed_gives_the_same_scene() -> None:
    a, b = _draw(7), _draw(7)
    assert replace(a, scene_id="") == replace(b, scene_id="")
    assert replace(_draw(8), scene_id="") != replace(a, scene_id="")


def test_people_stand_free_and_near_home() -> None:
    for seed in range(5):
        scene = _draw(seed)
        low, high = stills.PEOPLE_PER_SCENE
        assert low <= len(scene.people) <= high
        for p in scene.people:
            assert 8.0 < math.hypot(p.east_m, p.north_m) <= stills.RADIUS_M + 1.0
            assert not (100 <= p.east_m <= 140 and 100 <= p.north_m <= 140)  # the house
        objects.check(list(scene.people))


def test_people_have_real_body_heights() -> None:
    rng = random.Random(1)
    heights = [stills.draw_height(rng) for _ in range(4000)]
    adults = [h for h in heights if h >= 1.5]
    assert min(heights) >= stills.CHILD_M[0] and max(heights) <= stills.ADULT_M[3]
    assert 1.68 < float(np.mean(adults)) < 1.76
    assert 0.04 < (len(heights) - len(adults)) / len(heights) < 0.12  # children
    for seed in range(4):
        for p in _draw(seed).people:
            assert 1.0 < p.height_m < 2.0, p


def test_a_scaled_person_has_the_drawn_height() -> None:
    model = BY_ID["sd_person_walking"]
    p = Person("p", model.id, "standing", 20.0, 30.0, 0.0, scale=1.80 / model.body_height_m)
    assert p.height_m == pytest.approx(1.80, abs=0.01)
    low = Person("p", model.id, "standing", 20.0, 30.0, 0.0, scale=1.0)
    assert p.lift_m == pytest.approx(low.lift_m * p.scale, abs=1e-6)
    assert "<scale>" in p.sdf(objects.Pose(20.0, 30.0, 0.0, 0.0))
    with pytest.raises(objects.ObjectError):
        Person("p", model.id, "standing", 20.0, 30.0, 0.0, scale=3.0)


def test_poses_and_places_vary() -> None:
    poses, places = set(), set()
    for seed in range(6):
        scene = _draw(seed)
        poses |= {p.pose for p in scene.people}
        places |= {scene.places[p.name] for p in scene.people}
    assert poses == {"standing", "sitting", "lying"}
    assert {"open", "path"} <= places


def test_shots_are_aimed_at_people_from_the_profiles_heights() -> None:
    scene = _draw(3, profile="heridal")
    prof = stills.PROFILES["heridal"]
    assert scene.profile == "heridal"
    aimed = [s for s in scene.shots if s.aim]
    assert len(aimed) >= len(scene.people)
    for shot in aimed:
        assert prof.heights_m[0] - 0.01 <= shot.height_m <= prof.heights_m[1] + 0.01
        assert abs(shot.roll_deg) <= 10.0 and abs(shot.pitch_deg) <= 10.0
        person = next(p for p in scene.people if p.name == shot.aim)
        g = _land().ground(person.east_m, person.north_m)
        uv = project(prof.intrinsics, shot.camera_pose(), (person.east_m, person.north_m, g))
        assert uv is not None
        assert 0 <= uv[0] <= prof.width and 0 <= uv[1] <= prof.height


def test_profiles_match_the_real_sets_ground_detail() -> None:
    heridal, kios = stills.PROFILES["heridal"], stills.PROFILES["kios"]
    assert 0.9 < heridal.gsd_cm(heridal.heights_m[0]) < heridal.gsd_cm(heridal.heights_m[1]) < 2.0
    assert 3.0 < kios.gsd_cm(kios.heights_m[0]) < kios.gsd_cm(kios.heights_m[1]) < 7.0


def test_a_look_recolours_the_cloth_and_keeps_its_folds() -> None:
    rgb = np.full((4, 4, 3), 0.9, np.float32)
    rgb[0, :] = 0.45  # a fold: half as bright
    weight = np.ones((4, 4), np.float32)
    weight[3, 3] = 0.0  # not cloth
    out = looks.recolour(rgb, weight, (0.8, 0.1, 0.1))
    assert out[1, 1] == pytest.approx([0.8, 0.1, 0.1], abs=0.02)
    assert out[0, 0, 0] == pytest.approx(out[1, 1, 0] * 0.5, abs=0.02)
    assert out[3, 3] == pytest.approx(rgb[3, 3])
    assert looks.name("sd_person_walking", "red", "jeans") == "sd_person_walking-red-jeans"
    with pytest.raises(ValueError):
        looks.name("sd_person_walking", "plaid", "jeans")


def test_a_person_in_a_look_uses_the_looks_folder() -> None:
    look = looks.name("sd_person_walking", sorted(looks.TOPS)[0], "black")
    p = Person("p", "sd_person_walking", "standing", 20.0, 30.0, 0.0, look=look)
    assert f"model://{look}/" in p.sdf(objects.Pose(20.0, 30.0, 0.0, 0.0))
    assert (
        Person.from_truth(
            {"name": "p", "east_m": 20.0, "north_m": 30.0, "yaw_deg": 0.0, "truth": dict(p.truth)}
        )
        == p
    )
    with pytest.raises(objects.ObjectError):
        Person(
            "p",
            "sd_person_walking",
            "standing",
            20.0,
            30.0,
            0.0,
            look="sd_person_sitting-red-black",
        )


def test_ground_patches_cover_every_picture() -> None:
    feet = [
        (ground.footprint(e, n, 40.0, 44.0, 0.75, yaw), 1.5)
        for e, n, yaw in ((30.0, 40.0, 0.0), (35.0, 45.0, 30.0), (-200.0, 150.0, 90.0))
    ]
    patches = ground.plan(feet, "heridal", 6, 1)
    assert len(patches) == 2
    for poly, _ in feet:
        assert any(shapely.box(*p.bounds()).contains(poly) for p in patches)
    for p in patches:
        assert p.side_m <= ground.MAX_SIDE_M["heridal"] + 2.0
        assert p.cell_m <= 0.015
        assert math.hypot(p.east_m, p.north_m) > objects.HOME_KEEP_OUT_M
    assert ground.lift_at(patches, 30.0, 40.0) >= ground.LIFT_M
    assert ground.lift_at(patches, 250.0, -250.0) == 0.0


def test_a_ground_patch_is_not_turned() -> None:
    # The mesh is built in east/north: it must sit in the world unturned (yaw 0), or its
    # hills land beside the terrain's and hide people.
    p = ground.plan([(ground.footprint(30.0, 40.0, 40.0, 44.0, 0.75, 0.0), 1.5)], "heridal", 6, 1)[
        0
    ]
    assert objects.yaw_enu(p.yaw_deg) == pytest.approx(0.0)


def test_ground_patches_share_no_texture_or_material_name() -> None:
    # The renderer keeps textures and materials by name: two patches with the same
    # "patch.jpg" would both show the first one's ground.
    feet = [(ground.footprint(e, 0.0, 40.0, 44.0, 0.75, 0.0), 1.5) for e in (0.0, 300.0)]
    a, b = ground.plan(feet, "heridal", 6, 1)
    assert a.name != b.name
    for p in (a, b):
        assert f"model://{p.name}/{p.name}.obj" in p.sdf(objects.Pose(0.0, 0.0, 0.0, 0.0))


def test_develop_exposes_a_dark_render_like_a_photo() -> None:
    gen = np.random.default_rng(0)
    dark = (gen.random((120, 160, 3)) * 50).astype(np.uint8)
    out, info = stills.develop(dark, random.Random(2))
    assert out.shape == dark.shape and out.dtype == np.uint8
    assert stills.EXPOSURE_MEAN[0] - 15 < out.mean() < stills.EXPOSURE_MEAN[1] + 15
    assert 1.0 < info["gain"] <= stills.MAX_GAIN
    assert stills.develop(dark, random.Random(2))[0].tobytes() == out.tobytes()


def _one_person(pose: str = "lying") -> tuple[dict, stills.Shot, labels.ImageFacit]:
    model = "sd_person_walking"
    person = Person("person_0", model, pose, 0.0, 0.0, 30.0)
    truth = {
        "objects": [
            {
                "name": person.name,
                "east_m": 0.0,
                "north_m": 0.0,
                "up_m": 0.0,
                "yaw_deg": 30.0,
                "ground_m": 0.0,
                "truth": dict(person.truth),
            }
        ]
    }
    shot = stills.Shot("s000", "heridal", 0.0, 0.0, 30.0, 30.0, 0.0, 0.0, 0.0, "person_0")
    cam = stills.PROFILES["heridal"].intrinsics
    fac = labels.facit(truth, cam, shot.camera_pose(), [])
    return truth, shot, fac


def test_how_much_of_a_person_shows_is_counted_in_pixels() -> None:
    truth, shot, fac = _one_person()
    cam = stills.PROFILES["heridal"].intrinsics
    pose = shot.camera_pose()
    (p,) = fac.people
    whole = stills.silhouette_px(*labels.people_in(truth)[0], cam, pose)
    assert 0 < whole < p.box.area
    label_map = np.zeros((cam.height, cam.width), np.uint8)
    x0, y0 = int(p.box.x0) + 1, int(p.box.y0) + 1
    side = int(math.sqrt(whole))
    label_map[y0 : y0 + side, x0 : x0 + side // 2] = people.PERSON_LABEL  # half of it
    shown = stills.shown_shares(truth, fac, cam, pose, label_map)
    assert shown["person_0"] == pytest.approx(0.5, abs=0.05)
    # Specks of the person seen through gaps in a crown are not a person in view.
    specks = np.zeros_like(label_map)
    specks[int(p.box.y0) + 1 : int(p.box.y1) - 1 : 9, int(p.box.x0) + 1 : int(p.box.x1) - 1 : 9] = 2
    speck_share = stills.shown_shares(truth, fac, cam, pose, specks)["person_0"]
    assert speck_share is not None and speck_share < stills.MIN_SHOWN
    assert stills.shown_shares(truth, fac, cam, pose, None) == {"person_0": None}


def test_a_person_mostly_under_a_crown_gets_no_label() -> None:
    _, _, fac = _one_person()
    (p,) = fac.people
    seen = labels.PersonLabel(p.name, p.box, 1.0, p.box, p.model, p.pose)
    fac = labels.ImageFacit([seen])
    assert stills.label_boxes(fac, {p.name: 0.1}) == []
    assert stills.label_boxes(fac, {p.name: 0.6}) == [p.box]
    assert stills.label_boxes(fac, {p.name: None}) == [p.box]  # cannot tell: kept


def test_the_rig_gives_a_label_map_for_the_farm() -> None:
    text = stills.Rig(6, "heridal").sdf(objects.Pose(0.0, 0.0, 40.0, 0.0))
    assert text.count("<sensor ") == 3
    assert "<segmentation_type>semantic</segmentation_type>" in text


def test_a_pgm_label_map_reads_back(tmp_path: Path) -> None:
    (tmp_path / "m.pgm").write_bytes(b"P5 3 2 255\n" + bytes([0, 2, 0, 2, 2, 0]))
    assert stills.read_pgm(tmp_path / "m.pgm").tolist() == [[0, 2, 0], [2, 2, 0]]


def test_ground_photos_give_only_fine_detail() -> None:
    from PIL import Image

    gen = np.random.default_rng(1)
    yy, xx = np.mgrid[0:512, 0:512] / 512
    big = np.sin(2 * np.pi * 4 * xx) * np.sin(2 * np.pi * 4 * yy)  # patches 4 m across
    fine = gen.random((512, 512)) * 0.2
    im = Image.fromarray(((0.45 + 0.3 * big + fine) * 255).astype(np.uint8))
    out = ground.fine_detail(im, size_m=32.0)
    assert out.shape == (512, 512, 3)
    low = out[..., 0].reshape(16, 32, 16, 32).mean(axis=(1, 3))
    before = (0.45 + 0.3 * big).reshape(16, 32, 16, 32).mean(axis=(1, 3))
    assert low.std() < 0.2 * (before / before.mean()).std()  # the patches are gone
    assert float(out[..., 0].std()) > 0.05  # the grain stays
    assert out.min() >= 1 - ground.DETAIL_RANGE and out.max() <= 1 + ground.DETAIL_RANGE
