"""The farm's facit: the camera model, projected person boxes, YOLO/COCO and the round's check."""

from __future__ import annotations

import json
import math
from pathlib import Path
from typing import Any

import numpy as np
import pytest

from sd_sim.farm import labels, qc
from sd_sim.farm import round as farm_round
from sd_sim.farm.camera import (
    DOWN,
    Box,
    CameraPose,
    Intrinsics,
    box,
    project,
    quat_from_rpy,
    rotate,
)
from sd_sim.farm.gazebo import GazeboRunner
from sd_sim.farm.people import PERSON_LABEL, Person

CAM = Intrinsics(1280, 720, math.radians(66.0))


def above(x: float, y: float, z: float, yaw: float = 0.0) -> CameraPose:
    """The camera on a level vehicle at (x, y, z) facing ``yaw`` (ENU, from east)."""
    return CameraPose.mounted((x, y, z), quat_from_rpy(0.0, 0.0, yaw))


def person_row(p: Person, ground: float = 0.0) -> dict[str, Any]:
    return {
        "name": p.name,
        "east_m": p.east_m,
        "north_m": p.north_m,
        "ground_m": ground,
        "up_m": p.up_m,
        "yaw_deg": p.yaw_deg,
        "label": p.label,
        "truth": dict(p.truth),
    }


def truth_of(*people: Person) -> dict[str, Any]:
    return {"world": "sd_lake_forest", "objects": [person_row(p) for p in people]}


def test_mount_looks_down_with_the_front_at_the_top() -> None:
    assert rotate(DOWN, (1.0, 0.0, 0.0)) == pytest.approx((0.0, 0.0, -1.0))
    assert rotate(DOWN, (0.0, 0.0, 1.0)) == pytest.approx((1.0, 0.0, 0.0))
    pose = above(0.0, 0.0, 40.0)  # facing east
    centre = project(CAM, pose, (0.0, 0.0, 0.0))
    assert centre == pytest.approx((640.0, 360.0))
    ahead = project(CAM, pose, (5.0, 0.0, 0.0))  # east: up in the picture
    left = project(CAM, pose, (0.0, 5.0, 0.0))  # north: left of an east-facing vehicle
    assert ahead is not None and left is not None
    assert ahead[1] < 360.0 and ahead[0] == pytest.approx(640.0)
    assert left[0] < 640.0 and left[1] == pytest.approx(360.0)
    # 66 degrees across: the picture's edge is tan(33 deg) * height away from the centre
    edge = project(CAM, pose, (0.0, -40.0 * math.tan(math.radians(33.0)), 0.0))
    assert edge is not None and edge[0] == pytest.approx(1280.0)


def test_person_box_size_matches_the_ground_sample_distance() -> None:
    standing = Person("person_0", "sd_person_standing", "standing", 0.0, 0.0, 0.0)
    lying = Person("person_1", "sd_person_standing", "lying", 0.0, 0.0, 0.0)
    gsd = 2 * 40.0 * math.tan(math.radians(33.0)) / 1280  # metres per pixel at 40 m
    got_s = box(CAM, above(0.0, 0.0, 40.0), standing.corners())
    got_l = box(CAM, above(0.0, 0.0, 40.0), lying.corners())
    assert got_s is not None and got_l is not None
    # From above a lying person is long (1.9 m), a standing one only shoulders wide.
    assert max(got_l[0].w, got_l[0].h) * gsd == pytest.approx(1.93, rel=0.1)
    assert max(got_s[0].w, got_s[0].h) * gsd < 1.0
    assert max(c[2] for c in lying.corners()) < 0.5
    assert min(c[2] for c in lying.corners()) == pytest.approx(0.0, abs=1e-9)


def test_facit_pairs_projection_with_the_label_camera() -> None:
    open_p = Person("person_0", "sd_person_standing", "lying", 3.0, 2.0, 30.0)
    hidden = Person("person_1", "sd_person_walking", "standing", -6.0, -3.0, 0.0)
    far = Person("person_2", "sd_person_walking", "standing", 300.0, 0.0, 0.0)
    pose = above(0.0, 0.0, 40.0)
    got = box(CAM, pose, open_p.corners())
    assert got is not None
    full = got[0]
    seen = Box(full.x0 + 1, full.y0 + 1, full.x1 - 1, full.y1 - 1)
    stray = Box(10, 10, 20, 30)
    f = labels.facit(truth_of(open_p, hidden, far), CAM, pose, [seen, stray])
    assert [p.name for p in f.people] == ["person_0", "person_1"]  # person_2 out of view
    assert [p.name for p in f.labels] == ["person_0"]
    assert [p.name for p in f.hidden] == ["person_1"]
    assert f.unmatched_seen == [stray]
    lab = f.labels[0]
    assert lab.containment == pytest.approx(1.0)
    assert lab.iou is not None and lab.iou > 0.8
    line = labels.yolo_lines(f.labels, CAM).split()
    assert line[0] == "0" and len(line) == 5
    assert float(line[1]) == pytest.approx((full.x0 + full.x1) / 2 / 1280, abs=1e-5)
    coco = labels.Coco({"round_id": "r"})
    image_id = coco.add("images/a.png", CAM, f.labels)
    ann = coco.data["annotations"][0]
    assert ann["image_id"] == image_id and ann["category_id"] == 1
    assert ann["bbox"][2] == pytest.approx(full.w, abs=0.01)
    assert coco.data["categories"] == [labels.COCO_CATEGORY]


def test_camera_boxes_keep_only_people() -> None:
    raw = [
        {"label": PERSON_LABEL, "x0": 1, "y0": 2, "x1": 3, "y1": 4},
        {"label": 1, "x0": 5, "y0": 6, "x1": 7, "y1": 8},
    ]
    assert labels.camera_boxes(raw) == [Box(1, 2, 3, 4)]
    assert labels.camera_boxes(None) == []


def write_ppm(path: Path, image: np.ndarray) -> None:
    h, w, _ = image.shape
    path.write_bytes(b"P6 %d %d 255\n" % (w, h) + image.astype(np.uint8).tobytes())


def test_ppm_and_black_pictures(tmp_path: Path) -> None:
    rng = np.random.default_rng(1)
    noisy = rng.integers(40, 200, size=(8, 12, 3), dtype=np.uint8)
    write_ppm(tmp_path / "a.ppm", noisy)
    assert np.array_equal(labels.read_ppm(tmp_path / "a.ppm"), noisy)
    assert not labels.is_black(noisy)
    assert labels.is_black(np.zeros((8, 12, 3), dtype=np.uint8))
    assert labels.is_black(np.full((8, 12, 3), 120, dtype=np.uint8))  # one flat colour
    (tmp_path / "b.ppm").write_bytes(b"P5 1 1 255\n\x00")
    with pytest.raises(ValueError):
        labels.read_ppm(tmp_path / "b.ppm")


class FakeRunner:
    """Stills the way gz_stills writes them, over people in a flat world."""

    def __init__(self, people: list[Person], heights: list[float], black: int = 0) -> None:
        self.people = people
        self.heights = heights
        self.black = black

    def run(self, spec: farm_round.RoundSpec, work: Path) -> farm_round.RawRound:
        stills = work / "stills"
        stills.mkdir(parents=True)
        rng = np.random.default_rng(spec.seed)
        truth = truth_of(*self.people)
        for i, z in enumerate(self.heights):
            pose = above(0.0, 0.0, z)
            image = rng.integers(30, 220, size=(CAM.height, CAM.width, 3), dtype=np.uint8)
            if i < self.black:
                image[:] = 0
            seen = []
            for p in self.people:
                got = box(CAM, pose, p.corners())
                if got is not None:
                    b = got[0]
                    seen.append(
                        {"label": PERSON_LABEL, "x0": b.x0, "y0": b.y0, "x1": b.x1, "y1": b.y1}
                    )
            name = f"{i:08d}"
            write_ppm(stills / f"{name}.ppm", image)
            meta = {
                "name": name,
                "sim_time_ns": i * 1_000_000_000,
                "width": CAM.width,
                "height": CAM.height,
                "boxes": seen,
                "camera": pose.as_dict(),
            }
            (stills / f"{name}.json").write_text(json.dumps(meta), encoding="utf-8")
        return farm_round.RawRound(
            stills,
            truth,
            {"world": "sd_lake_forest", "weather": "clear", "height_m": 40.0},
            CAM,
            extra={"run_dir": "runs/x"},
        )


def spec(tmp_path: Path, seed: int = 7) -> Path:
    s = farm_round.RoundSpec.new(seed, "script", "0123456789abcdef0123")
    path = tmp_path / "spec.json"
    s.write(path)
    return path


def test_round_writes_a_checked_dataset(tmp_path: Path) -> None:
    people = [
        Person("person_0", "sd_person_rescue_randy", "lying", 4.0, 1.0, 10.0, is_target=True),
        Person("person_1", "sd_person_sitting", "sitting", -5.0, 3.0, 200.0, tint=(0.8, 0.1, 0.1)),
    ]
    runner = FakeRunner(people, [0.3, 1.0, 30, 35, 40, 40, 45, 50], black=0)
    path = spec(tmp_path)
    assert farm_round.execute(path, runner) == farm_round.EXIT_OK
    result = farm_round.read_result(tmp_path)
    assert result is not None and result["status"] == "ok", result
    assert result["images"] == 6  # the two pictures on the ground are left out
    assert result["qc"]["on_ground"] == 2 and result["qc"]["labels"] == 12
    assert result["qc"]["median_iou"] == pytest.approx(1.0)
    assert result["run_dir"] == "runs/x"
    assert {c["license"] for c in result["credits"]} == {"CC-BY-4.0", "CC0-1.0"}
    images = sorted((tmp_path / "images").glob("*.png"))
    assert len(images) == 6
    name = images[0].stem
    assert len((tmp_path / "labels" / f"{name}.txt").read_text().splitlines()) == 2
    meta = json.loads((tmp_path / "meta" / f"{name}.json").read_text())
    assert meta["schema"] == labels.IMAGE_SCHEMA
    assert meta["commit"].startswith("0123456789")
    assert meta["height_m"] == 30.0 and meta["world"] == "sd_lake_forest"
    assert len(meta["camera"]["orientation_wxyz"]) == 4
    assert any(c["model"].startswith("Rescue Randy v3") for c in meta["credits"])
    coco = json.loads((tmp_path / "coco.json").read_text())
    assert len(coco["images"]) == 6 and len(coco["annotations"]) == 12
    assert (tmp_path / "truth.json").is_file()


def test_round_that_never_took_off_fails(tmp_path: Path) -> None:
    runner = FakeRunner([Person("person_0", "sd_person_walking", "standing", 3, 3, 0)], [0.3] * 6)
    assert farm_round.execute(spec(tmp_path), runner) == farm_round.EXIT_QC
    result = farm_round.read_result(tmp_path)
    assert result is not None and result["status"] == "failed"
    assert "lyfte inte" in result["error"]


def test_black_pictures_fail_the_round(tmp_path: Path) -> None:
    runner = FakeRunner(
        [Person("person_0", "sd_person_walking", "standing", 3, 3, 0)], [40.0] * 8, black=3
    )
    assert farm_round.execute(spec(tmp_path), runner) == farm_round.EXIT_QC
    result = farm_round.read_result(tmp_path)
    assert result is not None and "svarta" in result["error"]
    assert result["images"] == 5  # black pictures are never kept


def test_facit_that_misses_fails_the_check() -> None:
    t = qc.RoundTally(max_height_m=40.0, pictures=10, labels=10)
    p = labels.PersonLabel("p", Box(0, 0, 10, 10), 1.0, Box(50, 50, 60, 60), "m", "standing")
    f = labels.ImageFacit(people=[p])
    t.add(f)
    result = qc.check(t)
    assert not result.ok
    assert any("utanför facit" in r for r in result.reasons)


def test_gazebo_round_is_not_built_yet(tmp_path: Path) -> None:
    assert farm_round.execute(spec(tmp_path), GazeboRunner()) == farm_round.EXIT_NOT_BUILT
    result = farm_round.read_result(tmp_path)
    assert result is not None and result["status"] == "not_built"
    assert "inte byggt än" in result["error"]


def test_spec_round_trip_and_bad_commander(tmp_path: Path) -> None:
    path = spec(tmp_path, seed=42)
    got = farm_round.RoundSpec.read(path)
    assert got.seed == 42 and got.round_id.endswith("-s42-01234567")
    with pytest.raises(ValueError):
        farm_round.RoundSpec.new(1, "pilot", "abc")


def test_ppm_pixels_that_look_like_whitespace(tmp_path: Path) -> None:
    image = np.array([[[32, 10, 9], [200, 100, 50]]], dtype=np.uint8)
    (tmp_path / "a.ppm").write_bytes(b"P6\n# gz\n2 1\n255\n" + image.tobytes())
    assert np.array_equal(labels.read_ppm(tmp_path / "a.ppm"), image)
    (tmp_path / "b.ppm").write_bytes(b"P6 2 1 255")
    with pytest.raises(ValueError):
        labels.read_ppm(tmp_path / "b.ppm")


def test_person_at_the_picture_edge_is_no_stray() -> None:
    pose = above(0.0, 0.0, 40.0)
    edge = None
    for i in range(200):  # walk the person out over the picture's right edge
        p = Person("person_0", "sd_person_standing", "lying", 0.0, -25.0 - i * 0.02, 90.0)
        got = box(CAM, pose, p.corners())
        if got is not None and 0.05 < got[1] < labels.MIN_IN_VIEW:
            edge = (p, got[0])
            break
    assert edge is not None
    person, part = edge
    f = labels.facit(truth_of(person), CAM, pose, [part])
    assert f.unmatched_seen == [] and f.labels == []
    tally = qc.RoundTally()
    tally.add(f)
    assert tally.stray == 0


def test_spec_carries_the_worlds(tmp_path: Path) -> None:
    s = farm_round.RoundSpec.new(3, "sd-brain", "abcdef0123", worlds=["w1", "w2"])
    s.write(tmp_path / "spec.json")
    back = farm_round.RoundSpec.read(tmp_path / "spec.json")
    assert back == s and back.worlds == ("w1", "w2")
    with pytest.raises(ValueError):
        farm_round.RoundSpec.new(3, "script", "abc", worlds=[])


def test_grade_hook(tmp_path: Path) -> None:
    assert farm_round.grade(tmp_path, available=lambda: False) is None
    calls: list[list[str]] = []

    def run(cmd: list[str], **_kw: Any) -> None:
        calls.append(cmd)
        (tmp_path / "grade.json").write_text(json.dumps({"is_exam": True, "cost_sek": 0.4}))

    got = farm_round.grade(tmp_path, available=lambda: True, run=run)
    assert got == {"is_exam": True, "cost_sek": 0.4}
    assert calls[0][-5:] == ["sd_brain", "eval", "grade", "--round", str(tmp_path)]
    (tmp_path / "grade.json").unlink()
    broken = farm_round.grade(tmp_path, available=lambda: True, run=lambda *a, **k: None)
    assert broken is not None and "error" in broken
