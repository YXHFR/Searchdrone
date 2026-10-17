"""Charters: building a proposal into a charter, checking it, splitting it into sectors.

Everything here is pure (no database, no clock): ``services.missions`` loads the mission,
the active charter and home, calls these and stores the result.

* ``build`` turns a ``CharterProposal`` into a ``MissionCharter`` draft: the server fills
  in home, the pilot position (always home in v1), the limits for every field left out,
  the default skills and, when the proposal has none, the sectors.
* ``validate`` lists every reason the draft cannot be shown to the pilot (``code`` +
  ``detail_sv``); an empty list means it can. The same check runs again at the click.
* ``resolve_autonomy``: an explicit choice on the card wins; otherwise the active
  charter's mode is inherited (a proposal can never change it); the first charter gets
  the profile's default.

Distances are geodesic (``sd_safety.targets.distance_m``), areas are measured in a local
east/north plane around home (``sd_safety.targets.Plane``), the same geometry the gate
uses.
"""

from __future__ import annotations

import math
import string
from collections.abc import Iterable, Mapping, Sequence
from typing import Any, Final

import shapely
from shapely.geometry import MultiPolygon, Point, box
from shapely.geometry import Polygon as ShapelyPolygon

from sd_safety.constants import FENCE_MARGIN_M
from sd_safety.limits import (
    ABSOLUTE_MIN_ALT_AGL_M,
    area_reach_from_home_m,
    fc_area_max_from_home_m,
    fc_fence_params,
)
from sd_safety.targets import Plane, distance_m
from sd_schemas.base import GeoJsonPolygon, LatLon, LatLonAlt
from sd_schemas.charter import (
    MAX_SECTORS,
    SECTOR_MAX_AREA_M2,
    AutonomyMode,
    CharterProposal,
    CharterValidationError,
    MissionCharter,
    Sector,
)
from sd_schemas.reason_codes import CHARTER_VALIDATION_CODES
from sd_schemas.safety import SafetyLimits
from sd_schemas.skills import SkillSpec

__all__ = [
    "MAX_VERTICES",
    "SECTOR_CELL_M",
    "CharterInvalidError",
    "build",
    "default_skills",
    "polygon_xy",
    "resolve_autonomy",
    "sector_area_m2",
    "split_sectors",
    "validate",
]

MAX_VERTICES: Final = 200
MIN_VERTICES: Final = 3
# One grid cell per sector. 1 m under 500 m so that a sliver (< 100 m²) merged into a
# full cell still fits in one survey_sector (250 000 m²).
SECTOR_CELL_M: Final = 499.0
SLIVER_M2: Final = 100.0
_SECTOR_IDS: Final = [*string.ascii_uppercase] + [
    a + b for a in string.ascii_uppercase for b in string.ascii_uppercase
]


class CharterInvalidError(Exception):
    """The proposal cannot become a charter (422 with ``errors``)."""

    def __init__(self, errors: list[CharterValidationError]) -> None:
        super().__init__(", ".join(e.code for e in errors))
        self.errors = errors


def error(
    code: str, *, field: str | None = None, limit: float | None = None
) -> CharterValidationError:
    return CharterValidationError(
        code=code, detail_sv=CHARTER_VALIDATION_CODES[code], field=field, limit=limit
    )


# --- geometry -------------------------------------------------------------------------


def _ring(polygon: Mapping[str, Any]) -> list[tuple[float, float]]:
    """The outer ring as (lon, lat), closing corner dropped."""
    coords = polygon.get("coordinates")
    if not isinstance(coords, list) or not coords or not isinstance(coords[0], list):
        raise ValueError("not a polygon")
    ring = [(float(p[0]), float(p[1])) for p in coords[0]]
    if len(ring) > 1 and ring[0] == ring[-1]:
        ring = ring[:-1]
    return ring


def polygon_xy(polygon: Mapping[str, Any], plane: Plane) -> ShapelyPolygon:
    """A GeoJSON polygon (outer ring) in the local plane."""
    return ShapelyPolygon([plane.xy(lat, lon) for lon, lat in _ring(polygon)])


def geojson_from_xy(poly: ShapelyPolygon, plane: Plane) -> GeoJsonPolygon:
    ring = []
    for x, y in poly.exterior.coords:
        lat, lon = plane.latlon(x, y)
        ring.append([round(lon, 9), round(lat, 9)])
    if ring[0] != ring[-1]:
        ring.append(ring[0])
    return {"type": "Polygon", "coordinates": [ring]}


def sector_area_m2(polygon: Mapping[str, Any], plane: Plane) -> float:
    return float(polygon_xy(polygon, plane).area)


# --- sectors --------------------------------------------------------------------------


def _parts(geom: Any) -> list[ShapelyPolygon]:
    if isinstance(geom, ShapelyPolygon):
        return [geom] if not geom.is_empty else []
    if isinstance(geom, MultiPolygon):
        return [g for g in geom.geoms if not g.is_empty]
    if hasattr(geom, "geoms"):
        return [g for part in geom.geoms for g in _parts(part)]
    return []


def split_sectors(area: Mapping[str, Any], plane: Plane) -> list[Sector]:
    """The area in a north-south / east-west grid of cells (at most ``SECTOR_CELL_M``),
    clipped to the area. Named A, B, ... row by row from the north-west; pieces under
    100 m² join the neighbour they share the longest edge with."""
    poly = polygon_xy(area, plane)
    if poly.area <= SECTOR_MAX_AREA_M2:
        pieces = [poly]
    else:
        minx, miny, maxx, maxy = poly.bounds
        cols = max(1, math.ceil((maxx - minx) / SECTOR_CELL_M))
        rows = max(1, math.ceil((maxy - miny) / SECTOR_CELL_M))
        cells: list[tuple[int, int, ShapelyPolygon]] = []
        for r in range(rows):
            top = maxy - r * SECTOR_CELL_M
            for c in range(cols):
                left = minx + c * SECTOR_CELL_M
                clipped = poly.intersection(
                    box(left, top - SECTOR_CELL_M, left + SECTOR_CELL_M, top)
                )
                cells += [(r, c, p) for p in _parts(clipped)]
        pieces = _merge_slivers([p for _, _, p in sorted(cells, key=lambda t: (t[0], t[1]))])
    return [
        Sector(
            sector_id=_SECTOR_IDS[i],
            name=_SECTOR_IDS[i],
            area=geojson_from_xy(p, plane),
            order_idx=i,
        )
        for i, p in enumerate(pieces)
    ]


def _merge_slivers(pieces: list[ShapelyPolygon]) -> list[ShapelyPolygon]:
    big = [p for p in pieces if p.area >= SLIVER_M2]
    for sliver in (p for p in pieces if p.area < SLIVER_M2):
        if not big:
            big.append(sliver)
            continue
        shared = [sliver.boundary.intersection(b.boundary).length for b in big]
        best = max(range(len(big)), key=lambda i: shared[i])
        merged = shapely.union(big[best], sliver)
        big[best] = merged if isinstance(merged, ShapelyPolygon) else big[best]
    return big


# --- building and checking ------------------------------------------------------------


def default_skills(registry: Iterable[SkillSpec]) -> list[str]:
    """Every implemented stage-1 skill."""
    return sorted(s.name for s in registry if s.stage == 1 and s.status == "implemented")


def build(
    proposal: CharterProposal,
    *,
    area: GeoJsonPolygon,
    charter_id: str,
    mission_id: str,
    vehicle_id: str,
    version: int,
    home: LatLonAlt,
    limits: SafetyLimits,
    registry: Iterable[SkillSpec],
    autonomy_mode: AutonomyMode,
) -> MissionCharter:
    """The charter a proposal describes. ``area`` is the proposal's polygon or the drawn
    area it names. Raises ``CharterInvalidError`` when the values cannot form a charter
    at all (e.g. the height band is the wrong way round)."""
    plane = Plane(home.lat, home.lon)
    try:
        sectors = proposal.sectors if proposal.sectors is not None else split_sectors(area, plane)
    except (ValueError, IndexError, shapely.errors.GEOSException):
        raise CharterInvalidError([error("AREA_INVALID", field="area")]) from None
    data: dict[str, Any] = {
        "charter_id": charter_id,
        "mission_id": mission_id,
        "vehicle_id": vehicle_id,
        "version": version,
        "area": area,
        "geofence_buffer_m": proposal.geofence_buffer_m,
        "home": home,
        "pilot_position": LatLon(lat=home.lat, lon=home.lon),
        "alt_min_agl_m": proposal.alt_min_agl_m,
        "alt_max_agl_m": proposal.alt_max_agl_m
        if proposal.alt_max_agl_m is not None
        else limits.max_alt_agl_m,
        "default_alt_agl_m": proposal.default_alt_agl_m,
        "max_speed_ms": proposal.max_speed_ms or limits.max_speed_ms,
        "max_range_from_pilot_m": proposal.max_range_from_pilot_m or limits.max_range_from_pilot_m,
        "battery_rtl_pct": proposal.battery_rtl_pct or limits.battery_rtl_pct,
        "allowed_skills": proposal.allowed_skills
        if proposal.allowed_skills is not None
        else default_skills(registry),
        "autonomy_mode": autonomy_mode,
        "sectors": sectors,
        "fence_hash": None,
        "approved_by": None,
        "approved_at": None,
        "notes": proposal.reason_sv or None,
    }
    band = _band_errors(data)
    if band:
        raise CharterInvalidError(band)
    try:
        return MissionCharter.model_validate(data)
    except ValueError:
        raise CharterInvalidError([error("VALIDATION")]) from None


def _band_errors(data: Mapping[str, Any]) -> list[CharterValidationError]:
    lo, hi, mid = data["alt_min_agl_m"], data["alt_max_agl_m"], data["default_alt_agl_m"]
    if lo < ABSOLUTE_MIN_ALT_AGL_M or lo >= hi or not lo <= mid <= hi:
        return [error("ALT_BAND_INVALID", field="alt_min_agl_m", limit=ABSOLUTE_MIN_ALT_AGL_M)]
    return []


def validate(
    charter: MissionCharter,
    *,
    limits: SafetyLimits,
    registry: Mapping[str, SkillSpec],
    max_area_km2: float,
    active: MissionCharter | None,
    base_version: int | None,
    airborne: bool,
) -> list[CharterValidationError]:
    """Every reason the charter cannot be offered (empty list = it can)."""
    out: list[CharterValidationError] = []
    plane = Plane(charter.home.lat, charter.home.lon)
    try:
        ring = _ring(charter.area)
        poly = polygon_xy(charter.area, plane)
        geometry_ok = MIN_VERTICES <= len(ring) <= MAX_VERTICES and poly.is_valid and poly.area > 0
    except (ValueError, TypeError, shapely.errors.GEOSException):
        ring, poly, geometry_ok = [], ShapelyPolygon(), False
    if not geometry_ok or poly.area > max_area_km2 * 1e6:
        out.append(error("AREA_INVALID", field="area", limit=max_area_km2))
        return out + _limit_errors(charter, limits) + _skill_errors(charter, registry)

    if not poly.covers(Point(0.0, 0.0)):  # home is the plane's origin
        out.append(error("HOME_OUTSIDE_AREA", field="area"))
    for sector in charter.sectors:
        try:
            sp = polygon_xy(sector.area, plane)
        except (ValueError, TypeError):
            out.append(error("SECTOR_OUTSIDE_AREA", field=f"sectors.{sector.sector_id}"))
            continue
        if not sp.is_valid or not poly.buffer(0.5).covers(sp):
            out.append(error("SECTOR_OUTSIDE_AREA", field=f"sectors.{sector.sector_id}"))
        if sp.area > SECTOR_MAX_AREA_M2:
            out.append(
                error(
                    "SECTOR_TOO_LARGE",
                    field=f"sectors.{sector.sector_id}",
                    limit=SECTOR_MAX_AREA_M2,
                )
            )
    if (
        len({s.sector_id for s in charter.sectors}) != len(charter.sectors)
        or len(charter.sectors) > MAX_SECTORS
    ):
        out.append(error("SECTOR_OUTSIDE_AREA", field="sectors"))

    pilot = charter.pilot_position
    reach = min(charter.max_range_from_pilot_m, limits.max_range_from_pilot_m)
    far = max(distance_m(pilot.lat, pilot.lon, lat, lon) for lon, lat in ring)
    if far > reach:
        out.append(error("AREA_BEYOND_RANGE", field="area", limit=reach))
    fc_reach = fc_area_max_from_home_m(limits)
    if area_reach_from_home_m(charter) > fc_reach:
        out.append(error("AREA_BEYOND_FC_FENCE", field="area", limit=fc_reach))
    radius = fc_fence_params(limits)["FENCE_RADIUS"]
    home_reach = max(distance_m(charter.home.lat, charter.home.lon, lat, lon) for lon, lat in ring)
    if home_reach + charter.geofence_buffer_m > radius - FENCE_MARGIN_M:
        out.append(
            error("BUFFER_INVALID", field="geofence_buffer_m", limit=radius - FENCE_MARGIN_M)
        )

    out += _band_errors(charter.model_dump())
    out += _limit_errors(charter, limits)
    out += _skill_errors(charter, registry)

    if airborne and active is not None:
        old = polygon_xy(active.area, Plane(active.home.lat, active.home.lon))
        new = polygon_xy(charter.area, Plane(active.home.lat, active.home.lon))
        if not old.buffer(0.01).covers(new):
            out.append(error("AREA_CHANGE_REQUIRES_LANDING", field="area"))
    current = None if active is None else active.version
    if base_version != current:
        out.append(error("BASE_VERSION_STALE", field="base_version", limit=current))
    return _unique(out)


def _limit_errors(charter: MissionCharter, limits: SafetyLimits) -> list[CharterValidationError]:
    out = []
    pairs: Sequence[tuple[str, bool, float]] = (
        ("alt_max_agl_m", charter.alt_max_agl_m > limits.max_alt_agl_m, limits.max_alt_agl_m),
        ("max_speed_ms", charter.max_speed_ms > limits.max_speed_ms, limits.max_speed_ms),
        (
            "max_range_from_pilot_m",
            charter.max_range_from_pilot_m > limits.max_range_from_pilot_m,
            limits.max_range_from_pilot_m,
        ),
        (
            "battery_rtl_pct",
            charter.battery_rtl_pct < limits.battery_rtl_pct,
            limits.battery_rtl_pct,
        ),
    )
    for name, looser, limit in pairs:
        if looser:
            out.append(error("CHARTER_EXCEEDS_LIMITS", field=name, limit=limit))
    return out


def _skill_errors(
    charter: MissionCharter, registry: Mapping[str, SkillSpec]
) -> list[CharterValidationError]:
    out = []
    for name in charter.allowed_skills:
        spec = registry.get(name)
        if spec is None:
            out.append(error("SKILL_UNKNOWN", field=f"allowed_skills.{name}"))
        elif spec.status != "implemented":
            out.append(error("SKILL_NOT_IMPLEMENTED", field=f"allowed_skills.{name}"))
    return out


def _unique(errors: list[CharterValidationError]) -> list[CharterValidationError]:
    seen: set[tuple[str, str | None]] = set()
    out = []
    for e in errors:
        key = (e.code, e.field)
        if key not in seen:
            seen.add(key)
            out.append(e)
    return out


def resolve_autonomy(
    active: MissionCharter | None,
    chosen: AutonomyMode | None,
    *,
    profile_default: AutonomyMode = AutonomyMode.WITHIN_CHARTER,
) -> AutonomyMode:
    """The mode a newly approved charter gets: the pilot's explicit choice, else the
    active charter's (inherited), else the profile's default for a first charter."""
    if chosen is not None:
        return chosen
    if active is not None:
        return active.autonomy_mode
    return profile_default
