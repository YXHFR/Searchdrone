"""The parts of the Gazebo stills script that need no Gazebo, and the box conversion."""

from __future__ import annotations

import json
import math
import threading
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import pytest

from sd_sim import gz_stills, render_stills


def _image(sec: int, nsec: int = 0, fmt: int = gz_stills.RGB_INT8) -> Any:
    return SimpleNamespace(
        header=SimpleNamespace(stamp=SimpleNamespace(sec=sec, nsec=nsec)),
        pixel_format_type=fmt,
        width=2,
        height=1,
        data=b"\x01\x02\x03\x04\x05\x06",
    )


def _boxes(sec: int, *labels: int) -> Any:
    corner = SimpleNamespace
    boxes = [
        SimpleNamespace(
            label=lab, box=SimpleNamespace(min_corner=corner(x=1, y=2), max_corner=corner(x=5, y=8))
        )
        for lab in labels
    ]
    return SimpleNamespace(
        header=SimpleNamespace(stamp=SimpleNamespace(sec=sec, nsec=0)), annotated_box=boxes
    )


def test_recorder_keeps_pictures_and_boxes_by_time() -> None:
    rec = gz_stills.Recorder()
    rec.on_image(_image(1))
    rec.on_image(_image(2, 500))
    rec.on_image(_image(3, fmt=99))  # not RGB: ignored
    assert rec.newest == 2_000_000_500
    assert rec.wait_image_after(1_000_000_000, 0.1) == 2_000_000_500
    assert rec.wait_image_after(2_000_000_500, 0.05) is None  # nothing newer
    rec.on_boxes(_boxes(2, 1, 10))
    assert rec.boxes_at(2_000_000_000, 0.05) == [
        {"label": 1, "x0": 1.0, "y0": 2.0, "x1": 5.0, "y1": 8.0},
        {"label": 10, "x0": 1.0, "y0": 2.0, "x1": 5.0, "y1": 8.0},
    ]
    assert rec.boxes_at(9, 0.05) is None
    rec.on_image(_image(20))  # pictures older than 10 s are dropped
    assert sorted(rec.images) == [20_000_000_000]


def test_recorder_wakes_a_waiting_reader() -> None:
    rec = gz_stills.Recorder()
    threading.Timer(0.05, lambda: rec.on_image(_image(4))).start()
    assert rec.wait_image_after(-1, 5.0) == 4_000_000_000


def test_write_still_writes_the_picture_before_its_description(tmp_path: Path) -> None:
    gz_stills.write_still(
        tmp_path, "a", 7, (2, 1, b"\x01\x02\x03\x04\x05\x06"), [], {"pose": {"x": 1}}
    )
    assert (tmp_path / "a.ppm").read_bytes() == b"P6 2 1 255\n\x01\x02\x03\x04\x05\x06"
    meta = json.loads((tmp_path / "a.json").read_text(encoding="utf-8"))
    assert meta == {
        "name": "a",
        "sim_time_ns": 7,
        "width": 2,
        "height": 1,
        "boxes": [],
        "pose": {"x": 1},
    }
    assert not list(tmp_path.glob(".*"))  # no temporary files left


@pytest.mark.parametrize(
    ("rpy", "expected"),
    [
        ((0.0, 0.0, 0.0), (1.0, 0.0, 0.0, 0.0)),
        ((0.0, 0.0, math.pi / 2), (math.cos(math.pi / 4), 0.0, 0.0, math.sin(math.pi / 4))),
        ((0.0, math.pi / 2, 0.0), (math.cos(math.pi / 4), 0.0, math.sin(math.pi / 4), 0.0)),
        ((math.pi, 0.0, 0.0), (0.0, 1.0, 0.0, 0.0)),
    ],
)
def test_quaternion(rpy: tuple[float, float, float], expected: tuple[float, ...]) -> None:
    assert gz_stills.quat(*rpy) == pytest.approx(expected, abs=1e-12)


def test_needs_loopback_transport(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("GZ_IP", "192.0.2.1")  # any address but loopback
    assert gz_stills.main(["shots"]) == 2


def test_boxes_by_class() -> None:
    raw = [
        {"label": 1, "x0": 10.0, "y0": 20.0, "x1": 30.0, "y1": 25.0},
        {"label": 99, "x0": 0.0, "y0": 0.0, "x1": 1.0, "y1": 1.0},  # not in the register
    ]
    assert render_stills.boxes_by_class(raw, {1: "red_block"}) == [
        {"cls": "red_block", "x": 10.0, "y": 20.0, "w": 20.0, "h": 5.0}
    ]


def test_prune_keeps_the_newest(tmp_path: Path) -> None:
    for n in range(5):
        gz_stills.write_still(tmp_path, f"{n:08d}", n, (1, 1, b"\x00\x00\x00"), [])
    assert gz_stills.prune(tmp_path, 2) == 3
    assert sorted(p.name for p in tmp_path.iterdir()) == [
        "00000003.json",
        "00000003.ppm",
        "00000004.json",
        "00000004.ppm",
    ]
    assert gz_stills.prune(tmp_path, 2) == 0


def _pose(name: str, xyz: tuple[float, float, float], wxyz: tuple[float, ...]) -> Any:
    w, x, y, z = wxyz
    return SimpleNamespace(
        name=name,
        position=SimpleNamespace(x=xyz[0], y=xyz[1], z=xyz[2]),
        orientation=SimpleNamespace(w=w, x=x, y=y, z=z),
    )


def _poses(sec: int, nsec: int, *poses: Any) -> Any:
    return SimpleNamespace(
        header=SimpleNamespace(stamp=SimpleNamespace(sec=sec, nsec=nsec)), pose=list(poses)
    )


def test_camera_pose_is_the_farms_camera_frame() -> None:
    from sd_sim.farm import camera

    yaw = camera.quat_from_rpy(0.0, 0.0, math.pi / 2)  # nose to the north
    got = gz_stills.camera_pose(((10.0, 20.0, 40.0), yaw), ((0.06, 0.0, -0.06), (1, 0, 0, 0)))
    assert got["position_m"] == pytest.approx([10.0, 20.06, 39.94])
    want = camera.quat_mul(yaw, camera.DOWN)
    assert got["orientation_wxyz"] == pytest.approx(list(want), abs=1e-6)
    # Looking straight down: the optical axis (camera x) points to -z in the world.
    q = tuple(got["orientation_wxyz"])
    assert camera.rotate(q, (1.0, 0.0, 0.0)) == pytest.approx((0.0, 0.0, -1.0), abs=1e-6)  # type: ignore[arg-type]


def test_recorder_matches_a_picture_with_the_pose_of_its_moment() -> None:
    rec = gz_stills.Recorder("sd_quad14", "camera_link")
    level = (1.0, 0.0, 0.0, 0.0)
    link = _pose("camera_link", (0.0, 0.0, 0.0), level)
    rec.on_poses(_poses(1, 10_000_000, _pose("sd_quad14", (1.0, 2.0, 30.0), level), link))
    rec.on_poses(_poses(2, 5_000_000, _pose("sd_quad14", (3.0, 2.0, 31.0), level), link))
    rec.on_poses(_poses(3, 0, _pose("rotor_0", (0.0, 0.0, 0.0), level)))  # no model: ignored
    assert rec.camera_at(2_000_000_000, 0.05)["position_m"] == [3.0, 2.0, 31.0]  # type: ignore[index]
    assert rec.camera_at(1_000_000_000, 0.05)["position_m"] == [1.0, 2.0, 30.0]  # type: ignore[index]
    assert rec.camera_at(1_500_000_000, 0.05) is None  # nothing near enough


def test_capture_needs_the_model_with_the_poses(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("GZ_IP", "127.0.0.1")
    argv = ["capture", "--image-topic", "a", "--boxes-topic", "b", "--out", "c"]
    with pytest.raises(SystemExit):
        gz_stills.main([*argv, "--poses-topic", "/world/w/dynamic_pose/info"])
