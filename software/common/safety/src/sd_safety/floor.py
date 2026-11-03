"""The height floor of a closer look: the lowest the drone may be over a point right now.

The floor comes from a map pack loaded before the flight (the ground's height, the trees'
heights, when they were scanned, known power lines), the owner's margins
(``sd_safety.constants``, decisions 14-17) and, on board, an optional downward laser.
The ground's gate and the onboard executor call the same pure functions, so they agree;
the ground has no laser and always counts without it. Nothing here reads files or
clocks.

For a point P, every 1 m cell within ``FLOOR_TREE_RADIUS_M`` counts with its tree:

- the map's height + ``FLOOR_UNDERESTIMATE_M`` + growth since the scan
  (``FLOOR_GROWTH_M_PER_Y`` a year, at most ``FLOOR_GROWTH_MAX_M``, which an unknown scan
  year also gets);
- at least ``FLOOR_UNKNOWN_TREE_M`` + the same margins (the most growth) where the map
  has no height (any value below 0), the cell has no ground height, the scan is older
  than ``FLOOR_SCAN_MAX_AGE_Y`` or dated after the flight (a data or clock error), or
  the cell is outside the pack. Unknown is a minimum, not a replacement: where such a
  cell still has a height, that height + the margins counts when it is higher.

The top of the highest such tree, above the ground at P, is ``tree_m``. A tree on ground
higher than P's counts from its own ground, so a slope never hides one; on flat ground
this is exactly the owner's "highest tree within 10 m". Then:

- without the laser the floor is the trees + ``FLOOR_CLEARANCE_NO_LASER_M``;
- with a confirmed laser it is the trees + ``FLOOR_ALT_ERROR_M``, and never nearer what the
  laser sees than ``LASER_FLOOR_MIN_M`` or its tested minimum range +
  ``LASER_RANGE_MARGIN_M``. The laser can raise the floor (a tree the map missed) but never
  take it below the map's. Its reading is of the ground under the drone: it counts at the
  point :func:`floor_at` is asked about (the caller passes it only for the drone's own
  position) and, along a path, within ``LASER_VALID_RADIUS_M`` of the path's first point
  (where the drone is). A laser lost during the look counts as none (``laser_lost``), and
  so does a reading that is not a finite number;
- within ``POWERLINE_ZONE_M`` of a known line, horizontally, the floor is the height the
  look started at (a line of one point is a pole, with the zone around it);
- the auto floor (no pilot's click) is also at least ``AUTO_FLOOR_ABOVE_GROUND_M`` above
  the ground at P, and never below the lowest floor.

The most cautious value always wins. Without a pack, or at a point outside it, the floor
is the height the look started at (``no_map``): the drone looks but does not come down.
The floors are never lowered to fit the charter's ceiling: a floor above it is the
caller's ``FLOOR_ABOVE_CEILING``. A point or start height that is not a finite number is
refused (``ValueError``), since ``max`` would quietly drop it.

The pack and its loader are in :mod:`sd_safety.floor_pack` (re-exported here).
"""

from __future__ import annotations

import math
from collections.abc import Sequence
from dataclasses import dataclass
from datetime import date
from itertools import pairwise
from typing import Final

import numpy as np
from numpy.typing import NDArray

from sd_safety.constants import (
    FLOOR_ALT_ERROR_M,
    FLOOR_CLEARANCE_NO_LASER_M,
    FLOOR_GROWTH_M_PER_Y,
    FLOOR_GROWTH_MAX_M,
    FLOOR_PATH_STEP_M,
    FLOOR_SCAN_MAX_AGE_Y,
    FLOOR_TREE_RADIUS_M,
    FLOOR_UNDERESTIMATE_M,
    FLOOR_UNKNOWN_TREE_M,
    LASER_FLOOR_MIN_M,
    LASER_RANGE_MARGIN_M,
    LASER_VALID_RADIUS_M,
    POWERLINE_ZONE_M,
)
from sd_safety.floor_pack import (
    DTM_NODATA_CM,
    SCAN_YEAR_UNKNOWN,
    FloorPack,
    FloorPackError,
    Line,
    load_floor_pack,
)
from sd_safety.targets import Xy, point_segment_distance
from sd_schemas.base import LatLon
from sd_schemas.inspect import AUTO_FLOOR_ABOVE_GROUND_M, FloorReason

__all__ = [
    "FloorPack",
    "FloorPackError",
    "FloorResult",
    "LaserState",
    "floor_along",
    "floor_at",
    "load_floor_pack",
]

Ints = NDArray[np.int64]

_UNKNOWN_TOP_M: Final = FLOOR_UNKNOWN_TREE_M + FLOOR_UNDERESTIMATE_M + FLOOR_GROWTH_MAX_M
_REACH: Final = math.ceil(FLOOR_TREE_RADIUS_M) + 1  # cells either side of P's to look at


@dataclass(frozen=True, slots=True)
class LaserState:
    """The downward laser as the executor sees it; ``None`` instead of one: none fitted.

    Structurally identical to ``sd_schemas.rangefinder.LaserState`` (TN-4, #181), which
    replaces this one when it lands. ``surface_amsl_m`` (the highest surface it saw over
    the confirmation window) is set exactly when ``confirmed``; ``min_range_m`` is the
    tested minimum range from the configuration (GRF-250: 10, TF02-Pro: 0).
    """

    confirmed: bool
    surface_amsl_m: float | None
    min_range_m: float
    distance_m: float | None = None

    def __post_init__(self) -> None:
        if self.confirmed != (self.surface_amsl_m is not None):
            raise ValueError("surface_amsl_m is set exactly when the laser is confirmed")


@dataclass(frozen=True, slots=True)
class FloorResult:
    """The floor at a point (or the highest along a path), heights above sea level.

    ``ground_amsl_m`` and ``tree_m`` are None without a map (``no_map``). ``tree_m`` is
    the trees' adjusted height above the ground (the margins included).
    """

    ground_amsl_m: float | None
    tree_m: float | None
    floor_auto_amsl_m: float  # the lowest without the pilot's click
    floor_lowest_amsl_m: float  # the lowest with the click (and the laser if confirmed)
    reason: FloorReason
    laser_confirmed: bool


def floor_at(
    pack: FloorPack | None,
    lat: float,
    lon: float,
    *,
    flight_date: date,
    laser: LaserState | None,
    start_alt_amsl_m: float,
    laser_lost: bool = False,
    extra_lines: Sequence[Sequence[LatLon]] = (),
) -> FloorResult:
    """The floor over (lat, lon). ``laser``: the laser's state when (lat, lon) is under
    the drone, else None; ``laser_lost``: it was confirmed during this look and has
    stopped answering. ``start_alt_amsl_m``: the height the look started at;
    ``extra_lines``: power lines the pilot drew, on top of the pack's."""
    _finite(lat=lat, lon=lon, start_alt_amsl_m=start_alt_amsl_m)
    if pack is None:
        return _no_map(start_alt_amsl_m)
    lines = _all_lines(pack, extra_lines)
    p = pack.plane.xy(lat, lon)
    return _floor_xy(pack, p, flight_date, laser, p, laser_lost, start_alt_amsl_m, lines)


def floor_along(
    pack: FloorPack | None,
    path: Sequence[LatLon],
    *,
    flight_date: date,
    laser: LaserState | None,
    start_alt_amsl_m: float,
    laser_lost: bool = False,
    extra_lines: Sequence[Sequence[LatLon]] = (),
) -> FloorResult:
    """The highest floor along a path from where the drone is (``path[0]``), sampled
    every ``FLOOR_PATH_STEP_M`` (both ends included): the result at the point with the
    highest lowest floor, with the highest auto floor anywhere on the path. The laser
    counts within ``LASER_VALID_RADIUS_M`` of ``path[0]``."""
    if not path:
        raise ValueError("a path has at least one point")
    _finite(start_alt_amsl_m=start_alt_amsl_m)
    if pack is None:
        return _no_map(start_alt_amsl_m)
    lines = _all_lines(pack, extra_lines)
    xy = [pack.plane.point(p) for p in path]
    samples = [xy[0]]
    for a, b in pairwise(xy):
        n = max(1, math.ceil(math.dist(a, b) / FLOOR_PATH_STEP_M))
        samples += [
            (a[0] + (b[0] - a[0]) * i / n, a[1] + (b[1] - a[1]) * i / n) for i in range(1, n + 1)
        ]
    results = [
        _floor_xy(pack, p, flight_date, laser, xy[0], laser_lost, start_alt_amsl_m, lines)
        for p in samples
    ]
    worst = max(results, key=lambda r: r.floor_lowest_amsl_m)
    auto = max(r.floor_auto_amsl_m for r in results)
    return FloorResult(
        worst.ground_amsl_m,
        worst.tree_m,
        auto,
        worst.floor_lowest_amsl_m,
        worst.reason,
        worst.laser_confirmed,
    )


# --- the computation ------------------------------------------------------------------------


def _finite(**values: float) -> None:
    for name, value in values.items():
        if not math.isfinite(value):
            raise ValueError(f"{name} is not a finite number: {value}")


def _no_map(start_alt_amsl_m: float) -> FloorResult:
    return FloorResult(None, None, start_alt_amsl_m, start_alt_amsl_m, "no_map", False)


def _all_lines(pack: FloorPack, extra: Sequence[Sequence[LatLon]]) -> tuple[Line, ...]:
    return pack.lines + tuple(pack.line_xy(line) for line in extra)


def _floor_xy(
    pack: FloorPack,
    p: Xy,
    flight_date: date,
    laser: LaserState | None,
    laser_xy: Xy,
    laser_lost: bool,
    start_alt_amsl_m: float,
    lines: tuple[Line, ...],
) -> FloorResult:
    trees = _trees(pack, p, flight_date.year)
    if trees is None:
        return _no_map(start_alt_amsl_m)
    ground, tree_m, unknown = trees
    near_line = any(
        point_segment_distance(p, a, b) <= POWERLINE_ZONE_M
        for line in lines
        for a, b in pairwise(line)
    )
    line_floor = start_alt_amsl_m if near_line else -math.inf
    no_laser = ground + tree_m + FLOOR_CLEARANCE_NO_LASER_M
    laser_floor = None if laser_lost else _laser_floor(laser, laser_xy, p)
    map_lowest = ground + tree_m + FLOOR_ALT_ERROR_M
    if laser_floor is None:
        lowest = max(no_laser, line_floor)
    else:
        lowest = max(map_lowest, laser_floor, line_floor)
    auto = max(no_laser, ground + AUTO_FLOOR_ABOVE_GROUND_M, line_floor, lowest)
    reason: FloorReason
    if near_line:
        reason = "powerline"
    elif laser_lost:
        reason = "laser_lost"
    elif laser_floor is not None and laser_floor >= map_lowest:
        reason = "laser"
    else:
        reason = "unknown" if unknown else "map"
    return FloorResult(ground, tree_m, auto, lowest, reason, laser_floor is not None)


def _laser_floor(laser: LaserState | None, laser_xy: Xy, p: Xy) -> float | None:
    """The floor the laser sets at P, or None when it does not count there: none fitted,
    not confirmed, a reading that is not a finite number (``max`` would drop it and keep
    the map's lower floor), or it saw the ground too far from P."""
    if laser is None or laser.surface_amsl_m is None or not laser.confirmed:
        return None
    if not (math.isfinite(laser.surface_amsl_m) and math.isfinite(laser.min_range_m)):
        return None
    if math.dist(p, laser_xy) > LASER_VALID_RADIUS_M:
        return None
    return laser.surface_amsl_m + max(LASER_FLOOR_MIN_M, laser.min_range_m + LASER_RANGE_MARGIN_M)


def _trees(pack: FloorPack, p: Xy, flight_year: int) -> tuple[float, float, bool] | None:
    """(ground at P, the adjusted trees' top above it, whether an unknown cell set it), or
    None when P is outside the pack or has no ground height."""
    rows, cols = pack.dtm_cm.shape
    r0 = math.floor(rows / 2 - p[1])
    c0 = math.floor(p[0] + cols / 2)
    if not (0 <= r0 < rows and 0 <= c0 < cols) or int(pack.dtm_cm[r0, c0]) == DTM_NODATA_CM:
        return None
    ground = int(pack.dtm_cm[r0, c0]) / 100.0

    # Every cell of the window around P; those outside the pack stay unknown.
    window = np.arange(-_REACH, _REACH + 1, dtype=np.int64)
    rr: Ints = (r0 + window)[:, None].repeat(window.size, axis=1)
    cc: Ints = (c0 + window)[None, :].repeat(window.size, axis=0)
    # The nearest point of each cell to P: a cell that reaches into the circle counts.
    dx = np.maximum(np.abs(cc - cols / 2 + 0.5 - p[0]) - 0.5, 0.0)
    dy = np.maximum(np.abs(rows / 2 - rr - 0.5 - p[1]) - 0.5, 0.0)
    near = np.hypot(dx, dy) <= FLOOR_TREE_RADIUS_M
    inside = near & (rr >= 0) & (rr < rows) & (cc >= 0) & (cc < cols)
    ri: Ints = rr[inside]
    ci: Ints = cc[inside]

    dtm = pack.dtm_cm[ri, ci].astype(np.int64)
    chm = pack.chm_dm[ri, ci].astype(np.float64)
    year = pack.scan_year[ri // pack.scan_cell_m, ci // pack.scan_cell_m].astype(np.int64)
    has_ground = dtm != DTM_NODATA_CM
    # CHM_UNKNOWN_DM (-1) is the format's "unknown"; any other value below 0 is no height
    # either, so a writer's other nodata value can never count as a negative tree.
    measured = chm >= 0
    dated = year != SCAN_YEAR_UNKNOWN
    age = flight_year - year
    # A scan dated after the flight is a data or clock error: no scan, not a fresh one.
    in_date = ~dated | ((age >= 0) & (age <= FLOOR_SCAN_MAX_AGE_Y))
    known = has_ground & measured & in_date
    growth = np.where(
        known & dated,
        np.minimum(age * FLOOR_GROWTH_M_PER_Y, FLOOR_GROWTH_MAX_M),
        FLOOR_GROWTH_MAX_M,
    )
    own = np.where(measured, chm / 10.0 + FLOOR_UNDERESTIMATE_M + growth, -math.inf)
    # Unknown is 30 m forest at the least: a height the map still has counts when higher.
    height = np.where(known, own, np.maximum(own, _UNKNOWN_TOP_M))
    # A tree stands on its own ground when that is higher than P's; a cell without a
    # ground height (or outside the pack) on the highest ground in reach.
    highest = max(ground, float(dtm[has_ground].max()) / 100.0)
    base = np.where(has_ground, np.maximum(dtm / 100.0, ground), highest)
    tops = base + height
    known_top = float(tops[known].max()) if known.any() else -math.inf
    unknown_top = float(tops[~known].max()) if (~known).any() else -math.inf
    outside = bool((near & ~inside).any())
    if outside:
        unknown_top = max(unknown_top, highest + _UNKNOWN_TOP_M)
    top = max(known_top, unknown_top)
    return ground, top - ground, unknown_top >= known_top
