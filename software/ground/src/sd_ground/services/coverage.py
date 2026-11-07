"""How much of each sector has been seen: ``mission_sectors.covered``, once a second.

The covered area is the union of discs of radius ``spacing_m / 2 + 2 m`` around the
telemetry points flown during a ``survey_sector`` order within ±3 m of its search height,
clipped to the sector: the same reach and height band as ``sd_geo.coverage``. The onboard
computer sends no coverage of its own. A search the drone flies on after a closer look
(``inspections.flown_on``: its order stays ``superseded`` on the ground) counts from the
drone's report that it flies on until that run ends.

When a sector turns done (its search's done, or the end of a search flown on after a look),
what of it no track point saw is its gap (``gap_m2``): told once to the operator
(``SECTOR_GAP``) and kept for the AI's situation (``_Mission.gaps``). A gap behind the
drone (a cylinder lifted after the legs it cut were passed), a straighten refused, wind or a
lost stretch of track all show here, whatever the cause. Only the parts at least
``GAP_MIN_WIDTH_M`` wide and ``GAP_MIN_M2`` large count: the thin edges every search leaves
do not.

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
from sd_schemas.api import AlertCode, Command, Mission
from sd_schemas.base import GeoJsonPolygon
from sd_schemas.charter import MissionCharter

if TYPE_CHECKING:
    from sd_ground.services.missions import MissionFlow

__all__ = ["EVERY_S", "GAP_MIN_M2", "GAP_MIN_WIDTH_M", "Coverage", "covered_area", "gap_m2", "tick"]

EVERY_S: Final = 1.0
# A part of a done sector no track point saw is a gap when it is at least this wide (it
# holds a disc this wide) and this large. A full search leaves only slivers at the sector's
# edges and corners (about 850 m² in all, none of it 10 m wide); a search leg skipped for
# 90 m at 30 m spacing leaves a part about 26 m wide and about 1 600 m².
GAP_MIN_WIDTH_M: Final = 10.0
GAP_MIN_M2: Final = 300.0


@dataclass
class _Mission:
    last_tick: float = 0.0
    since: dt.datetime | None = None  # the newest track point counted
    seen: dict[str, BaseGeometry] = field(default_factory=dict[str, BaseGeometry])
    finished: set[str] = field(default_factory=set[str])  # surveys counted to their end
    # Done sectors whose gap was worked out: (m², % of the sector), (0, 0) for none.
    gaps: dict[str, tuple[float, float]] = field(default_factory=dict[str, tuple[float, float]])
    # False until the first tick has looked at the sectors: those already done then (the
    # ground restarted) were told before, so their gaps are kept for the AI, not told again.
    primed: bool = False


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


def gap_m2(sector: ShapelyPolygon, seen: BaseGeometry | None) -> float:
    """How much of ``sector`` (local plane) ``seen`` left out, counting only the parts at
    least ``GAP_MIN_WIDTH_M`` wide and ``GAP_MIN_M2`` large."""
    left = sector if seen is None else sector.difference(seen)
    total = 0.0
    for part in getattr(left, "geoms", [left]):
        if not isinstance(part, ShapelyPolygon) or part.area < GAP_MIN_M2:
            continue
        if part.buffer(-GAP_MIN_WIDTH_M / 2).is_empty:
            continue  # a sliver: no disc of that width fits
        total += part.area
    return total


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
    from sd_ground.services import inspections as inspections_service

    mid = str(mission.id)
    state = flow.coverage.of(mid)
    now = flow.g.clock.monotonic()
    if now - state.last_tick < EVERY_S:
        return
    state.last_tick = now
    charter = await flow.active_charter(mission)
    if charter is None:
        return
    flown_on = {f.command_id: f.since for f in inspections_service.flown_on(flow, mid)}
    surveys = [
        c
        for c in await flow.repo.commands(mid, None)
        if c.skill == "survey_sector"
        and (str(c.status) in ("running", "done") or str(c.id) in flown_on)
        and c.params.get("sector_id")
        and str(c.id) not in state.finished
    ]
    plane = Plane(charter.home.lat, charter.home.lon)
    sectors = {s.sector_id: s.area for s in charter.sectors}
    if surveys:
        await _count(flow, mid, state, surveys, flown_on, sectors, plane, charter)
    counting = {str(c.params["sector_id"]) for c in surveys if str(c.id) not in state.finished}
    await _gaps(flow, mission, state, sectors, plane, counting)
    state.primed = True


async def _count(
    flow: MissionFlow,
    mid: str,
    state: _Mission,
    surveys: list[Command],
    flown_on: dict[str, dt.datetime],
    sectors: dict[str, GeoJsonPolygon],
    plane: Plane,
    charter: MissionCharter,
) -> None:
    new_points = await flow.repo.track(mid, state.since)
    if new_points:
        state.since = new_points[-1]["t"]
    for cmd in surveys:
        if str(cmd.status) == "done":
            state.finished.add(str(cmd.id))  # this tick counts its last points
        sector_id = str(cmd.params["sector_id"])
        area = sectors.get(sector_id)
        if area is None:
            continue
        since = max(cmd.created_at, flown_on.get(str(cmd.id), cmd.created_at))
        points = [
            (float(p["lat"]), float(p["lon"]), float(p["alt_rel_home_m"] or 0.0))
            for p in new_points
            if p["t"] >= since
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


async def _gaps(
    flow: MissionFlow,
    mission: Mission,
    state: _Mission,
    sectors: dict[str, GeoJsonPolygon],
    plane: Plane,
    counting: set[str],
) -> None:
    """Each sector that turned done, once no search of it is still being counted: its gap,
    told to the operator when there is one. A sector that is no longer done (searched
    again) is worked out again when it is. Sectors already done at the first tick (the
    ground restarted) are worked out but not told again."""
    mid = str(mission.id)
    for sid, status in (await flow.repo.sector_statuses(mid)).items():
        if status != "done":
            state.gaps.pop(sid, None)
            continue
        area = sectors.get(sid)
        if sid in state.gaps or sid in counting or area is None:
            continue
        poly = polygon_xy(area, plane)
        m2 = await asyncio.to_thread(gap_m2, poly, state.seen.get(sid))
        pct = 100.0 * m2 / poly.area if poly.area > 0 else 0.0
        state.gaps[sid] = (m2, pct)
        if m2 <= 0.0 or not state.primed:
            continue
        flow.alert(
            str(AlertCode.SECTOR_GAP),
            "warning",
            f"Sektor {sid} är klar men cirka {_m2_sv(m2)} m² ({pct:.0f} %) är inte sökt.",
            vehicle_id=mission.vehicle_id,
            mission_id=mid,
            details={"sector_id": sid, "unsearched_m2": round(m2), "unsearched_pct": round(pct, 1)},
        )


def _m2_sv(m2: float) -> str:
    """Square metres the Swedish way, to the nearest hundred: "2 300"."""
    return f"{round(m2, -2):,.0f}".replace(",", " ")
