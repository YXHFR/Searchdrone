"""How much of each sector has been seen: ``mission_sectors.covered``, once a second.

The covered area is the union of a ``spacing_m / 2`` buffer around the telemetry points
flown during a ``survey_sector`` order at its search height (within ±3 m of the order's
``alt_m``), clipped to the sector: the same definition the onboard computer uses. The
onboard computer sends no coverage of its own.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any, Final

import shapely
from shapely.geometry import MultiPolygon, Point, mapping
from shapely.geometry import Polygon as ShapelyPolygon

from sd_ground.services.charters import geojson_from_xy, polygon_xy
from sd_safety.constants import DEFAULT_SURVEY_SPACING_M
from sd_safety.targets import Plane
from sd_schemas.api import Mission

if TYPE_CHECKING:
    from sd_ground.services.missions import MissionFlow

__all__ = ["ALT_BAND_M", "EVERY_S", "covered_area", "tick"]

EVERY_S: Final = 1.0
ALT_BAND_M: Final = 3.0
_last: dict[str, float] = {}


def covered_area(
    points: list[tuple[float, float, float]],
    sector: dict[str, Any],
    plane: Plane,
    *,
    alt_m: float,
    spacing_m: float,
) -> MultiPolygon | None:
    """The covered part of ``sector`` from (lat, lon, alt above home) points."""
    near = [plane.xy(lat, lon) for lat, lon, alt in points if abs(alt - alt_m) <= ALT_BAND_M]
    if not near:
        return None
    swath = shapely.union_all([Point(x, y).buffer(spacing_m / 2, quad_segs=4) for x, y in near])
    seen = swath.intersection(polygon_xy(sector, plane))
    parts = [
        g
        for g in getattr(seen, "geoms", [seen])
        if isinstance(g, ShapelyPolygon) and not g.is_empty
    ]
    return MultiPolygon(parts) if parts else None


def _multi_geojson(area: MultiPolygon, plane: Plane) -> dict[str, Any]:
    return {
        "type": "MultiPolygon",
        "coordinates": [geojson_from_xy(p, plane)["coordinates"] for p in area.geoms],
    }


async def tick(flow: MissionFlow, mission: Mission) -> None:
    mid = str(mission.id)
    now = flow.g.clock.monotonic()
    if now - _last.get(mid, 0.0) < EVERY_S:
        return
    _last[mid] = now
    charter = await flow.active_charter(mission)
    if charter is None:
        return
    surveys = [
        c
        for c in await flow.repo.commands(mid, None)
        if c.skill == "survey_sector"
        and str(c.status) in ("running", "done")
        and c.params.get("sector_id")
    ]
    if not surveys:
        return
    plane = Plane(charter.home.lat, charter.home.lon)
    sectors = {s.sector_id: s.area for s in charter.sectors}
    track = await flow.repo.track(mid)
    for cmd in surveys:
        sector_id = str(cmd.params["sector_id"])
        area = sectors.get(sector_id)
        if area is None:
            continue
        alt = float(cmd.params.get("alt_m") or charter.default_alt_agl_m)
        spacing = float(cmd.params.get("spacing_m") or DEFAULT_SURVEY_SPACING_M)
        points = [
            (float(p["lat"]), float(p["lon"]), float(p["alt_rel_home_m"] or 0.0))
            for p in track
            if p["t"] >= cmd.created_at
        ]
        seen = covered_area(points, area, plane, alt_m=alt, spacing_m=spacing)
        await flow.repo.set_sector_covered(
            mid, sector_id, None if seen is None else _multi_geojson(seen, plane), flow.now()
        )


def as_geojson(area: MultiPolygon) -> dict[str, Any]:
    return dict(mapping(area))
