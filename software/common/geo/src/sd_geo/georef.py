"""Georeferencing: from a pixel in a still to a point on the ground, and back.

A pixel becomes a ray through a pinhole camera, the ray is turned by the camera mount and
the vehicle's attitude at the moment of exposure, and the point where it meets the ground
is the answer. The ground is flat for now: a level plane ``height_m`` below the camera
(the height above the ground under the vehicle). A height model can replace the plane
later without changing the callers.

Everything is in a local frame centred on the camera: metres east and north (the same
plane as ``sd_safety.targets.Plane``), plus down. Turning the offset into latitude and
longitude is the caller's job, with the gate's geometry.

Conventions: pixel ``(u, v)`` has its origin in the top-left corner, ``u`` to the right
and ``v`` down; pixel centres are at half-integers. The attitude is ArduPilot's: roll
right wing down, pitch nose up, yaw clockwise from north, all in degrees, turning the
vehicle's front-right-down axes into north-east-down. A camera with
``mount_pitch_deg = -90`` looks straight down with the top of the picture towards the
vehicle's front and the right of the picture towards its right.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

__all__ = [
    "RADIUS_95",
    "Attitude",
    "ErrorBudget",
    "PinholeCamera",
    "footprint",
    "ground_offset",
    "project",
    "px_per_m",
    "sigma95_m",
]

RADIUS_95 = math.sqrt(-2.0 * math.log(0.05))  # 2.448: 95 % of a 2-D normal error
Vec3 = tuple[float, float, float]
Xy = tuple[float, float]


@dataclass(frozen=True)
class PinholeCamera:
    """A distortion-free camera with square pixels."""

    width_px: int
    height_px: int
    hfov_deg: float
    cx: float | None = None  # None = the middle of the picture
    cy: float | None = None
    mount_pitch_deg: float = -90.0  # 0 = forward, -90 = straight down

    def __post_init__(self) -> None:
        if self.width_px < 1 or self.height_px < 1:
            raise ValueError("the picture must have at least one pixel")
        if not 0 < self.hfov_deg < 180:
            raise ValueError("hfov_deg must be between 0 and 180")

    @property
    def focal_px(self) -> float:
        return (self.width_px / 2) / math.tan(math.radians(self.hfov_deg) / 2)

    @property
    def centre(self) -> Xy:
        return (
            self.width_px / 2 if self.cx is None else self.cx,
            self.height_px / 2 if self.cy is None else self.cy,
        )

    @property
    def vfov_deg(self) -> float:
        return math.degrees(2 * math.atan((self.height_px / 2) / self.focal_px))


@dataclass(frozen=True)
class Attitude:
    roll_deg: float = 0.0
    pitch_deg: float = 0.0
    yaw_deg: float = 0.0


def _body_to_ned(att: Attitude) -> tuple[Vec3, Vec3, Vec3]:
    """Rows of the rotation matrix R = Rz(yaw) Ry(pitch) Rx(roll)."""
    cr, sr = math.cos(math.radians(att.roll_deg)), math.sin(math.radians(att.roll_deg))
    cp, sp = math.cos(math.radians(att.pitch_deg)), math.sin(math.radians(att.pitch_deg))
    cy, sy = math.cos(math.radians(att.yaw_deg)), math.sin(math.radians(att.yaw_deg))
    return (
        (cy * cp, cy * sp * sr - sy * cr, cy * sp * cr + sy * sr),
        (sy * cp, sy * sp * sr + cy * cr, sy * sp * cr - cy * sr),
        (-sp, cp * sr, cp * cr),
    )


def _camera_axes(cam: PinholeCamera) -> tuple[Vec3, Vec3, Vec3]:
    """The picture's right, down and viewing axes in the vehicle's front-right-down frame."""
    t = math.radians(cam.mount_pitch_deg)
    right = (0.0, 1.0, 0.0)
    view = (math.cos(t), 0.0, -math.sin(t))
    down = (math.sin(t), 0.0, math.cos(t))  # view x right
    return right, down, view


def _ray_ned(u: float, v: float, cam: PinholeCamera, att: Attitude) -> Vec3:
    cx, cy = cam.centre
    f = cam.focal_px
    a, b = (u - cx) / f, (v - cy) / f
    right, down, view = _camera_axes(cam)
    body = tuple(a * right[i] + b * down[i] + view[i] for i in range(3))
    rot = _body_to_ned(att)
    return (
        sum(rot[0][i] * body[i] for i in range(3)),
        sum(rot[1][i] * body[i] for i in range(3)),
        sum(rot[2][i] * body[i] for i in range(3)),
    )


def ground_offset(
    u: float, v: float, cam: PinholeCamera, att: Attitude, height_m: float
) -> Xy | None:
    """(east, north) in metres from the camera to where pixel ``(u, v)`` meets the
    ground ``height_m`` below it; ``None`` when the ray does not reach the ground
    (it points at or above the horizon)."""
    if not height_m > 0:
        raise ValueError("height_m must be positive")
    north, east, down = _ray_ned(u, v, cam, att)
    if down <= 1e-6:
        return None
    t = height_m / down
    return east * t, north * t


def project(
    east: float, north: float, down_m: float, cam: PinholeCamera, att: Attitude
) -> Xy | None:
    """The inverse of :func:`ground_offset`: the pixel ``(u, v)`` where a point ``east``,
    ``north`` and ``down_m`` metres from the camera appears; ``None`` when it is behind
    the camera (outside the picture is still returned, as coordinates off the picture)."""
    rot = _body_to_ned(att)
    ned = (north, east, down_m)
    body = tuple(sum(rot[r][c] * ned[r] for r in range(3)) for c in range(3))  # R^T x
    right, down, view = _camera_axes(cam)
    z = sum(body[i] * view[i] for i in range(3))
    if z <= 1e-9:
        return None
    x = sum(body[i] * right[i] for i in range(3))
    y = sum(body[i] * down[i] for i in range(3))
    cx, cy = cam.centre
    return cx + cam.focal_px * x / z, cy + cam.focal_px * y / z


def footprint(cam: PinholeCamera, att: Attitude, height_m: float) -> list[Xy] | None:
    """The four picture corners on the ground (top-left, top-right, bottom-right,
    bottom-left) as (east, north) offsets; ``None`` when a corner misses the ground."""
    corners: list[Xy] = []
    for u, v in ((0, 0), (cam.width_px, 0), (cam.width_px, cam.height_px), (0, cam.height_px)):
        p = ground_offset(u, v, cam, att, height_m)
        if p is None:
            return None
        corners.append(p)
    return corners


def px_per_m(cam: PinholeCamera, height_m: float) -> float:
    """Picture pixels per metre on level ground straight below a camera looking down."""
    if not height_m > 0:
        raise ValueError("height_m must be positive")
    return cam.focal_px / height_m


@dataclass(frozen=True)
class ErrorBudget:
    """The 1-sigma error sources of a georeferenced point."""

    gnss_m: float = 1.5
    attitude_deg: float = 1.0
    heading_deg: float = 3.0
    timing_s: float = 0.05
    height_m: float = 2.0
    mount_deg: float = 1.5


def sigma95_m(
    offset_m: float, height_m: float, speed_ms: float, budget: ErrorBudget | None = None
) -> float:
    """The radius that holds the true ground position 95 % of the time: the 1-sigma error
    sources added in quadrature, times 2.45 (the 95 % radius of a round two-dimensional
    normal error; 2 would hold only 86 %). ``offset_m`` is the point's horizontal
    distance from the camera."""
    b = ErrorBudget() if budget is None else budget
    slant = math.hypot(offset_m, height_m)
    terms = (
        b.gnss_m,
        slant * math.tan(math.radians(b.attitude_deg)),
        slant * math.tan(math.radians(b.mount_deg)),
        offset_m * math.sin(math.radians(b.heading_deg)),
        speed_ms * b.timing_s,
        offset_m * b.height_m / height_m if height_m > 0 else 0.0,
    )
    return RADIUS_95 * math.sqrt(sum(t * t for t in terms))
