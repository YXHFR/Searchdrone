"""How much of each sector has been seen: ``mission_sectors.covered``, once a second.

The covered area is the union of discs of radius ``spacing_m / 2 + 2 m`` around the
telemetry points flown during a ``survey_sector`` order within ±3 m of its search height,
clipped to the sector: the same reach and height band as ``sd_geo.coverage``. The onboard
computer sends no coverage of its own.

The work is incremental: each tick reads only the track points newer than the last one
counted and adds their discs to what each sector already has. The geometry runs in a
worker thread so the main loop (which the liveness watchdog watches) never waits for it.
"""

from __future__ import annotations

import asyncio
import datetime as dt
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any, Final

import shapely
from shapely.geometry import MultiPolygon, Point
from shapely.geometry import Polygon as ShapelyPolygon
from shapely.geometry.base import BaseGeometry

from sd_geo.coverage import ALT_TOLERANCE_M, REACH_EXTRA_M
from sd_ground.services.charters import geojson_from_xy, polygon_xy
from sd_safety.constants import DEFAULT_SURVEY_SPACING_M
from sd_safety.targets import Plane
from sd_schemas.api import Mission

if TYPE_CHECKING:
    from sd_ground.services.missions import MissionFlow

__all__ = ["EVERY_S", "Coverage", "covered_area", "tick"]

EVERY_S: Final = 1.0


@dataclass
class _Mission:
    last_tick: float = 0.0
    since: dt.datetime | None = None  # the newest track point counted
    seen: dict[str, BaseGeometry] = field(default_factory=dict[str, BaseGeometry])
    finished: set[str] = field(default_factory=set[str])  # surveys counted to their end


@dataclass
class Coverage:
    """What each open mission's sectors have seen so far (kept by the mission flow)."""

    missions: dict[str, _Mission] = field(default_factory=dict[str, _Mission])

    def of(self, mission_id: str) -> _Mission:
        return self.missions.setdefault(mission_id, _Mission())

    def forget(self, mission_id: str) -> None:
        self.missions.pop(mission_id, None)


def covered_area(
    points: list[tuple[float, float, float]],
    sector: ShapelyPolygon,
    plane: Plane,
    *,
    alt_m: float,
    spacing_m: float,
) -> BaseGeometry | None:
    """The part of ``sector`` (local plane) seen from (lat, lon, height above home)
    points; ``None`` when no point counts."""
    near = [plane.xy(lat, lon) for lat, lon, alt in points if abs(alt - alt_m) <= ALT_TOLERANCE_M]
    if not near:
        return None
    reach = spacing_m / 2 + REACH_EXTRA_M
    swath = shapely.union_all([Point(x, y).buffer(reach, quad_segs=4) for x, y in near])
    seen = swath.intersection(sector)
    return None if seen.is_empty else seen


def _polygons(area: BaseGeometry) -> MultiPolygon | None:
    parts = [
        g
        for g in getattr(area, "geoms", [area])
        if isinstance(g, ShapelyPolygon) and not g.is_empty
    ]
    return MultiPolygon(parts) if parts else None


def _multi_geojson(area: MultiPolygon, plane: Plane) -> dict[str, Any]:
    return {
        "type": "MultiPolygon",
        "coordinates": [geojson_from_xy(p, plane)["coordinates"] for p in area.geoms],
    }


def _grow(
    old: BaseGeometry | None,
    points: list[tuple[float, float, float]],
    sector: ShapelyPolygon,
    plane: Plane,
    alt_m: float,
    spacing_m: float,
) -> BaseGeometry | None:
    new = covered_area(points, sector, plane, alt_m=alt_m, spacing_m=spacing_m)
    if new is None:
        return old
    return new if old is None else old.union(new)


async def tick(flow: MissionFlow, mission: Mission) -> None:
    mid = str(mission.id)
    state = flow.coverage.of(mid)
    now = flow.g.clock.monotonic()
    if now - state.last_tick < EVERY_S:
        return
    state.last_tick = now
    charter = await flow.active_charter(mission)
    if charter is None:
        return
    surveys = [
        c
        for c in await flow.repo.commands(mid, None)
        if c.skill == "survey_sector"
        and str(c.status) in ("running", "done")
        and c.params.get("sector_id")
        and str(c.id) not in state.finished
    ]
    if not surveys:
        return
    new_points = await flow.repo.track(mid, state.since)
    if new_points:
        state.since = new_points[-1]["t"]
    plane = Plane(charter.home.lat, charter.home.lon)
    sectors = {s.sector_id: s.area for s in charter.sectors}
    for cmd in surveys:
        if str(cmd.status) == "done":
            state.finished.add(str(cmd.id))  # this tick counts its last points
        sector_id = str(cmd.params["sector_id"])
        area = sectors.get(sector_id)
        if area is None:
            continue
        points = [
            (float(p["lat"]), float(p["lon"]), float(p["alt_rel_home_m"] or 0.0))
            for p in new_points
            if p["t"] >= cmd.created_at
        ]
        if not points:
            continue
        alt = float(cmd.params.get("alt_m") or charter.default_alt_agl_m)
        spacing = float(cmd.params.get("spacing_m") or DEFAULT_SURVEY_SPACING_M)
        grown = await asyncio.to_thread(
            _grow,
            state.seen.get(sector_id),
            points,
            polygon_xy(area, plane),
            plane,
            alt,
            spacing,
        )
        if grown is None:
            continue
        state.seen[sector_id] = grown
        multi = _polygons(grown)
        await flow.repo.set_sector_covered(
            mid, sector_id, None if multi is None else _multi_geojson(multi, plane), flow.now()
        )
