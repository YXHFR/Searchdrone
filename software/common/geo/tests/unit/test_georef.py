"""Georeferencing against synthetic poses: pixel -> ground -> pixel, and known geometry."""

from __future__ import annotations

import math
import random

import pytest
from hypothesis import given, settings
from hypothesis import strategies as st

from sd_geo import Attitude, PinholeCamera, footprint, ground_offset, project, px_per_m, sigma95_m

CAM = PinholeCamera(1280, 720, 66.0)


def test_centre_pixel_is_straight_below() -> None:
    assert ground_offset(640, 360, CAM, Attitude(yaw_deg=123.0), 40.0) == pytest.approx(
        (0.0, 0.0), abs=1e-9
    )


@pytest.mark.parametrize(
    ("yaw", "expected"),
    [(0.0, (0.0, 1.0)), (90.0, (1.0, 0.0)), (180.0, (0.0, -1.0)), (270.0, (-1.0, 0.0))],
)
def test_top_of_picture_points_forward(yaw: float, expected: tuple[float, float]) -> None:
    east, north = ground_offset(640, 0, CAM, Attitude(yaw_deg=yaw), 40.0) or (0.0, 0.0)
    half_height = 40.0 * math.tan(math.radians(CAM.vfov_deg / 2))
    assert (east / half_height, north / half_height) == pytest.approx(expected, abs=1e-9)


def test_right_of_picture_is_the_vehicles_right() -> None:
    east, north = ground_offset(1280, 360, CAM, Attitude(yaw_deg=0.0), 40.0) or (0.0, 0.0)
    assert north == pytest.approx(0.0, abs=1e-9)
    assert east == pytest.approx(40.0 * math.tan(math.radians(33.0)))


def test_nose_up_pitch_moves_the_centre_forward() -> None:
    _, north = ground_offset(640, 360, CAM, Attitude(pitch_deg=10.0), 40.0) or (0.0, 0.0)
    assert north == pytest.approx(40.0 * math.tan(math.radians(10.0)))


def test_right_roll_moves_the_centre_left() -> None:
    east, _ = ground_offset(640, 360, CAM, Attitude(roll_deg=10.0), 40.0) or (0.0, 0.0)
    assert east == pytest.approx(-40.0 * math.tan(math.radians(10.0)))


def test_a_ray_above_the_horizon_misses_the_ground() -> None:
    forward = PinholeCamera(640, 480, 80.0, mount_pitch_deg=0.0)
    assert ground_offset(320, 0, forward, Attitude(), 10.0) is None
    assert ground_offset(320, 479, forward, Attitude(), 10.0) is not None


def test_measured_in_the_rendered_simulator_picture() -> None:
    # The simulator's stills camera 30 m up, its front towards east: a 0.5 m cube whose
    # top is 3 m east and 2 m north of the camera was rendered at pixel (573, 260).
    u, v = project(3.0, 2.0, 29.5, CAM, Attitude(yaw_deg=90.0)) or (0.0, 0.0)
    assert (u, v) == pytest.approx((573.0, 260.0), abs=1.0)


def test_footprint_and_px_per_m() -> None:
    corners = footprint(CAM, Attitude(), 40.0)
    assert corners is not None
    width = corners[1][0] - corners[0][0]
    assert width == pytest.approx(2 * 40.0 * math.tan(math.radians(33.0)))
    assert px_per_m(CAM, 40.0) == pytest.approx(1280 / width)


@settings(max_examples=300, deadline=None)
@given(
    u=st.floats(0, 1280),
    v=st.floats(0, 720),
    roll=st.floats(-25, 25),
    pitch=st.floats(-25, 25),
    yaw=st.floats(0, 360),
    height=st.floats(5, 120),
    mount=st.floats(-90, -50),
)
def test_roundtrip_without_noise(
    u: float, v: float, roll: float, pitch: float, yaw: float, height: float, mount: float
) -> None:
    cam = PinholeCamera(1280, 720, 66.0, mount_pitch_deg=mount)
    att = Attitude(roll, pitch, yaw)
    ground = ground_offset(u, v, cam, att, height)
    if ground is None:
        return
    back = project(ground[0], ground[1], height, cam, att)
    assert back is not None
    assert back == pytest.approx((u, v), abs=1e-6)
    # ...and the ground point itself, back again: well under 0.1 m.
    again = ground_offset(back[0], back[1], cam, att, height)
    assert again is not None
    assert math.dist(again, ground) < 0.1


def test_noisy_pose_stays_inside_the_budget() -> None:
    """1 000 random cases with the pose noise of the error budget: median below 5 m at
    50 m height, and the 95 % radius covers about 95 % of the cases."""
    rng = random.Random(7)  # noqa: S311 - test data
    errors: list[float] = []
    inside = 0
    for _ in range(1000):
        truth = Attitude(rng.uniform(-8, 8), rng.uniform(-8, 8), rng.uniform(0, 360))
        u, v = rng.uniform(0, 1280), rng.uniform(0, 720)
        exact = ground_offset(u, v, CAM, truth, 50.0)
        assert exact is not None
        noisy = Attitude(
            truth.roll_deg + rng.gauss(0, math.hypot(1.0, 1.5)),
            truth.pitch_deg + rng.gauss(0, math.hypot(1.0, 1.5)),
            truth.yaw_deg + rng.gauss(0, 3.0),
        )
        got = ground_offset(u, v, CAM, noisy, 50.0 + rng.gauss(0, 2.0))
        assert got is not None
        gnss = (rng.gauss(0, 1.5), rng.gauss(0, 1.5))
        err = math.dist((got[0] + gnss[0], got[1] + gnss[1]), exact)
        errors.append(err)
        if err <= sigma95_m(math.hypot(*exact), 50.0, 0.0):
            inside += 1
    errors.sort()
    assert errors[len(errors) // 2] < 5.0
    assert 920 <= inside <= 990


def test_camera_rejects_nonsense() -> None:
    with pytest.raises(ValueError, match="hfov"):
        PinholeCamera(10, 10, 0.0)
    with pytest.raises(ValueError, match="height"):
        ground_offset(0, 0, CAM, Attitude(), 0.0)
