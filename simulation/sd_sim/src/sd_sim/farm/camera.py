"""The stills camera as Gazebo models it: a pinhole looking along its own x axis.

Gazebo's camera frame has x forward (the optical axis), y left and z up; the picture's
column grows to the right (towards -y) and its row grows downwards (towards -z), with
square pixels and the focal length set by the horizontal field of view. On the vehicle
the camera is pitched +90 degrees, so it looks straight down with the top of the picture
towards the vehicle's front.

``CameraPose`` is the camera frame's pose in the world (x east, y north, z up) as a
position and a unit quaternion (w, x, y, z), which is what the farm records for every
picture. ``box()`` projects a set of world points (a person's box corners) and returns
the axis-aligned picture box around them, clipped to the picture.
"""

from __future__ import annotations

import math
from collections.abc import Iterable, Sequence
from dataclasses import dataclass

Vec = tuple[float, float, float]
Quat = tuple[float, float, float, float]  # w, x, y, z

# The mount on the vehicle: pitched +90 degrees about y, looking down.
DOWN = (math.cos(math.pi / 4), 0.0, math.sin(math.pi / 4), 0.0)
MIN_DEPTH_M = 0.1  # the camera's near clip


def quat_from_rpy(roll: float, pitch: float, yaw: float) -> Quat:
    cr, sr = math.cos(roll / 2), math.sin(roll / 2)
    cp, sp = math.cos(pitch / 2), math.sin(pitch / 2)
    cy, sy = math.cos(yaw / 2), math.sin(yaw / 2)
    return (
        cr * cp * cy + sr * sp * sy,
        sr * cp * cy - cr * sp * sy,
        cr * sp * cy + sr * cp * sy,
        cr * cp * sy - sr * sp * cy,
    )


def quat_mul(a: Quat, b: Quat) -> Quat:
    aw, ax, ay, az = a
    bw, bx, by, bz = b
    return (
        aw * bw - ax * bx - ay * by - az * bz,
        aw * bx + ax * bw + ay * bz - az * by,
        aw * by - ax * bz + ay * bw + az * bx,
        aw * bz + ax * by - ay * bx + az * bw,
    )


def rotate(q: Quat, v: Vec) -> Vec:
    """``v`` turned by the unit quaternion ``q``."""
    w, x, y, z = q
    vx, vy, vz = v
    # t = 2 * cross(q.xyz, v); v' = v + w * t + cross(q.xyz, t)
    tx = 2 * (y * vz - z * vy)
    ty = 2 * (z * vx - x * vz)
    tz = 2 * (x * vy - y * vx)
    return (
        vx + w * tx + (y * tz - z * ty),
        vy + w * ty + (z * tx - x * tz),
        vz + w * tz + (x * ty - y * tx),
    )


def conj(q: Quat) -> Quat:
    return (q[0], -q[1], -q[2], -q[3])


@dataclass(frozen=True)
class Intrinsics:
    width: int
    height: int
    hfov_rad: float

    @property
    def focal_px(self) -> float:
        return (self.width / 2) / math.tan(self.hfov_rad / 2)


@dataclass(frozen=True)
class CameraPose:
    position: Vec
    orientation: Quat

    @classmethod
    def mounted(cls, body_position: Vec, body_orientation: Quat, mount: Quat = DOWN) -> CameraPose:
        """The camera's pose from the vehicle's pose and the camera's mount on it (the
        camera sits at the body's origin, as on the farm's vehicle and still rig)."""
        return cls(body_position, quat_mul(body_orientation, mount))

    def to_camera(self, p: Vec) -> Vec:
        d = (p[0] - self.position[0], p[1] - self.position[1], p[2] - self.position[2])
        return rotate(conj(self.orientation), d)

    def as_dict(self) -> dict[str, list[float]]:
        return {
            "position_m": [round(v, 4) for v in self.position],
            "orientation_wxyz": [round(v, 6) for v in self.orientation],
        }


def project(cam: Intrinsics, pose: CameraPose, p: Vec) -> tuple[float, float] | None:
    """Picture coordinates (column, row; pixel centres at .5) of a world point, or
    ``None`` when it is behind the near clip."""
    x, y, z = pose.to_camera(p)
    if x < MIN_DEPTH_M:
        return None
    f = cam.focal_px
    return cam.width / 2 - f * y / x, cam.height / 2 - f * z / x


@dataclass(frozen=True)
class Box:
    """An axis-aligned picture box in pixels: x0, y0 top left, x1, y1 bottom right."""

    x0: float
    y0: float
    x1: float
    y1: float

    @property
    def w(self) -> float:
        return max(0.0, self.x1 - self.x0)

    @property
    def h(self) -> float:
        return max(0.0, self.y1 - self.y0)

    @property
    def area(self) -> float:
        return self.w * self.h

    def clip(self, width: int, height: int) -> Box:
        return Box(
            min(max(self.x0, 0.0), width),
            min(max(self.y0, 0.0), height),
            min(max(self.x1, 0.0), width),
            min(max(self.y1, 0.0), height),
        )

    def iou(self, other: Box) -> float:
        inter = self.intersection(other)
        union = self.area + other.area - inter
        return inter / union if union > 0 else 0.0

    def intersection(self, other: Box) -> float:
        w = min(self.x1, other.x1) - max(self.x0, other.x0)
        h = min(self.y1, other.y1) - max(self.y0, other.y0)
        return max(0.0, w) * max(0.0, h)


def bounding(points: Iterable[tuple[float, float]]) -> Box | None:
    pts = list(points)
    if not pts:
        return None
    us = [p[0] for p in pts]
    vs = [p[1] for p in pts]
    return Box(min(us), min(vs), max(us), max(vs))


def box(cam: Intrinsics, pose: CameraPose, corners: Sequence[Vec]) -> tuple[Box, float] | None:
    """The clipped picture box of the points and the share of the unclipped box that is
    inside the picture (1.0: wholly in view). ``None`` when nothing is in view or a point
    is behind the camera (never the case for a camera that looks down from above)."""
    projected = [project(cam, pose, c) for c in corners]
    if any(p is None for p in projected):
        return None
    full = bounding(p for p in projected if p is not None)
    if full is None or full.area <= 0:
        return None
    clipped = full.clip(cam.width, cam.height)
    if clipped.area <= 0:
        return None
    return clipped, clipped.area / full.area
