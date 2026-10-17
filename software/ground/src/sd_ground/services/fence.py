"""The flight controller's fence: its polygon, its parameters and how they are checked.

Only the ground station changes the fence, and only for a charter or limits the pilot
approved by click; the AI has no way to start any of it (no route, no skill).

* **Polygon** (``fence_polygon``): ``buffer(area, geofence_buffer_m + 10 m)`` in the local
  plane around home, at most 70 corners. A simplification is accepted only while it
  still covers ``buffer(area, geofence_buffer_m)``, so it never moves an edge inside the
  onboard computer's own area guard. Uploaded as inclusion vertices (``mission_type = 1``)
  by ``sd_ground.mav.fence`` while the vehicle stands disarmed, then read back; only a
  matching ``fence_hash`` marks the charter verified.
* **Height, radius and LOITER speed** (``fence_param_targets``): the standard values, or
  just outside the mission's limits once the pilot has raised them above the standard
  (owner decision: limits may be raised per mission by click, up to the outer cap). Never
  below the standard values: the onboard computer refuses to fly with a fence inside them.
* **Check** (``check_fc_safety_params``): every safety parameter read back must hold;
  one violation keeps the charter unverified, so ``takeoff`` is refused with
  ``FENCE_NOT_VERIFIED``. A fence that is too loose stops READY as much as one too tight.
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from typing import Final

from shapely.geometry import Polygon as ShapelyPolygon

from sd_ground.services.charters import polygon_xy
from sd_safety.constants import FC_SAFETY_EXPECTED, FENCE_POLYGON_MARGIN_M, FS_GCS_TIMEOUT_S
from sd_safety.limits import DEFAULT_LIMITS, fc_fence_params
from sd_safety.targets import Plane
from sd_schemas.charter import MissionCharter
from sd_schemas.safety import SafetyLimits

__all__ = [
    "FENCE_MAX_VERTICES",
    "FENCE_PARAMS",
    "SAFETY_PARAMS",
    "FenceSimplifyError",
    "ParamViolation",
    "check_fc_safety_params",
    "fence_param_targets",
    "fence_polygon",
]

FENCE_MAX_VERTICES: Final = 70
SIMPLIFY_START_M: Final = 0.5
SIMPLIFY_MAX_M: Final = 4.0
PARAM_TOLERANCE: Final = 1e-3
FENCE_PARAMS: Final = ("FENCE_ALT_MAX", "FENCE_RADIUS", "LOIT_SPEED_MS")
SAFETY_PARAMS: Final = (
    "FENCE_ENABLE",
    "FENCE_TYPE",
    "FENCE_ACTION",
    *FENCE_PARAMS,
    "FS_GCS_ENABLE",
    "FS_GCS_TIMEOUT",
    "MAV_GCS_SYSID",
    "BATT_FS_LOW_ACT",
    "BATT_FS_CRT_ACT",
)


class FenceSimplifyError(ValueError):
    """No polygon with at most 70 corners covers the area with its buffer."""


@dataclass(frozen=True, slots=True)
class ParamViolation:
    param: str
    value: float | None
    expected: str


def fence_polygon(charter: MissionCharter) -> list[tuple[float, float]]:
    """The inclusion polygon's corners as (lat, lon), first corner not repeated."""
    plane = Plane(charter.home.lat, charter.home.lon)
    area = polygon_xy(charter.area, plane)
    must_cover = area.buffer(charter.geofence_buffer_m)
    fence = area.buffer(charter.geofence_buffer_m + FENCE_POLYGON_MARGIN_M, quad_segs=8)
    shape = _within_corners(fence, must_cover)
    coords = list(shape.exterior.coords)[:-1]
    return [plane.latlon(x, y) for x, y in coords]


def _corners(poly: ShapelyPolygon) -> int:
    return len(poly.exterior.coords) - 1


def _within_corners(fence: ShapelyPolygon, must_cover: ShapelyPolygon) -> ShapelyPolygon:
    if _corners(fence) <= FENCE_MAX_VERTICES:
        return fence
    tol = SIMPLIFY_START_M
    while tol <= SIMPLIFY_MAX_M:
        simple = fence.simplify(tol, preserve_topology=True)
        if (
            isinstance(simple, ShapelyPolygon)
            and _corners(simple) <= FENCE_MAX_VERTICES
            and simple.covers(must_cover)
        ):
            return simple
        tol *= 2
    raise FenceSimplifyError("simplify")


def fence_param_targets(limits: SafetyLimits) -> dict[str, float]:
    """Height, radius and LOITER speed for ``limits``: the standard fence, or just
    outside the limits where the pilot raised them above the standard."""
    widest = SafetyLimits(
        max_alt_agl_m=max(limits.max_alt_agl_m, DEFAULT_LIMITS.max_alt_agl_m),
        max_speed_ms=max(limits.max_speed_ms, DEFAULT_LIMITS.max_speed_ms),
        max_range_from_pilot_m=max(
            limits.max_range_from_pilot_m, DEFAULT_LIMITS.max_range_from_pilot_m
        ),
        battery_rtl_pct=limits.battery_rtl_pct,
    )
    return {k: float(v) for k, v in fc_fence_params(widest).items()}


def check_fc_safety_params(
    params: Mapping[str, float | None], limits: SafetyLimits
) -> list[ParamViolation]:
    """Every safety parameter that does not hold (empty = all good)."""
    out: list[ParamViolation] = []
    targets = fence_param_targets(limits)

    def need(name: str, ok: bool, expected: str) -> None:
        if not ok:
            out.append(ParamViolation(name, params.get(name), expected))

    def value(name: str) -> float | None:
        v = params.get(name)
        return None if v is None else float(v)

    enable, ftype = value("FENCE_ENABLE"), value("FENCE_TYPE")
    need("FENCE_ENABLE", enable == 1, "== 1")
    need("FENCE_TYPE", ftype is not None and (int(ftype) & 7) == 7, "& 7 == 7")
    need("FENCE_ACTION", value("FENCE_ACTION") == FC_SAFETY_EXPECTED["FENCE_ACTION"], "== 3")
    for name in FENCE_PARAMS:
        v = value(name)
        need(
            name,
            v is not None and abs(v - targets[name]) <= PARAM_TOLERANCE,
            f"== {targets[name]:g}",
        )
    gcs = value("FS_GCS_ENABLE")
    need("FS_GCS_ENABLE", gcs is not None and gcs != 0, "!= 0")
    need("FS_GCS_TIMEOUT", value("FS_GCS_TIMEOUT") == FS_GCS_TIMEOUT_S, f"== {FS_GCS_TIMEOUT_S:g}")
    need("MAV_GCS_SYSID", value("MAV_GCS_SYSID") == FC_SAFETY_EXPECTED["MAV_GCS_SYSID"], "== 250")
    low, crit = value("BATT_FS_LOW_ACT"), value("BATT_FS_CRT_ACT")
    need("BATT_FS_LOW_ACT", low is not None and low != 0, "!= 0")
    need("BATT_FS_CRT_ACT", crit is not None and crit != 0, "!= 0")
    return out
