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

The track reaches the database in batches (``telemetry.BATCH`` samples, about 5 s), so at a
search's end its last stretch may not be there yet. A search that ended still counts the
points up to its end until the track has caught up past it (``_Ending``; at most
``CATCH_UP_S``), and its sector's gap is worked out only then. Whether a gap was told is
read from the event log (``alarm.SECTOR_GAP`` since the sector's newest search order), so a
ground restart neither tells it again nor leaves it untold.

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
from sd_ground.logging import get_logger
from sd_ground.services.charters import geojson_from_xy, polygon_xy
from sd_safety.constants import DEFAULT_SURVEY_SPACING_M
from sd_safety.targets import Plane
from sd_schemas.api import AlertCode, Command, Mission
from sd_schemas.base import GeoJsonPolygon
from sd_schemas.charter import MissionCharter

if TYPE_CHECKING:
    from sd_ground.services.missions import MissionFlow

__all__ = [
    "CATCH_UP_S",
    "EVERY_S",
    "GAP_MIN_M2",
    "GAP_MIN_WIDTH_M",
    "Coverage",
    "covered_area",
    "gap_m2",
    "tick",
]

EVERY_S: Final = 1.0
# A part of a done sector no track point saw is a gap when it is at least this wide (it
# holds a disc this wide) and this large. A full search leaves only slivers at the sector's
# edges and corners (about 850 m² in all, none of it 10 m wide); a search leg skipped for
# 90 m at 30 m spacing leaves a part about 26 m wide and about 1 600 m².
GAP_MIN_WIDTH_M: Final = 10.0
GAP_MIN_M2: Final = 300.0
# How long an ended search waits for the track to catch up past its end: two batches of
# telemetry (``telemetry.BATCH`` at ``SAMPLE_S``, 5 s each) and a tick. Only when no newer
# point comes (the samples stopped) is the gap worked out without them.
CATCH_UP_S: Final = 12.0
# The sectors' statuses are read when a search's state changed, else this often: a status
# written just after the order's (the same report) is never missed for long.
GAPS_REFRESH_S: Final = 10.0
log = get_logger("sd_ground.coverage")


@dataclass
class _Ending:
    """A search that ended (done, broken off, or its run flown on after a look over): the
    points from ``since`` to ``until`` still count until the track has caught up past
    ``until`` or ``deadline`` (monotonic) has passed."""

    cmd: Command
    since: dt.datetime
    until: dt.datetime
    deadline: float


@dataclass
class _Mission:
    last_tick: float = 0.0
    since: dt.datetime | None = None  # the newest track point counted
    seen: dict[str, BaseGeometry] = field(default_factory=dict[str, BaseGeometry])
    finished: set[str] = field(default_factory=set[str])  # done searches counted to their end
    # The searches counted on the last tick (running, or flown on after a look: since when).
    open: dict[str, dt.datetime] = field(default_factory=dict[str, dt.datetime])
    flying_on: set[str] = field(default_factory=set[str])  # of those, flown on after a look
    ending: dict[str, _Ending] = field(default_factory=dict[str, _Ending])
    # Done sectors whose gap was worked out: (m², % of the sector), (0, 0) for none.
    gaps: dict[str, tuple[float, float]] = field(default_factory=dict[str, tuple[float, float]])
    # The searches' state the gaps were last worked out for: unchanged, nothing to do.
    gaps_for: object = None


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
    searches = [
        c
        for c in await flow.repo.commands(mid, None)
        if c.skill == "survey_sector" and c.params.get("sector_id")
    ]
    _follow(flow, state, searches, flown_on, now)
    plane = Plane(charter.home.lat, charter.home.lon)
    sectors = {s.sector_id: s.area for s in charter.sectors}
    by_id = {str(c.id): c for c in searches}
    try:
        if state.open or state.ending:
            await _count(flow, mid, state, by_id, sectors, plane, charter, now)
        counting = {str(by_id[cid].params["sector_id"]) for cid in state.open} | {
            str(e.cmd.params["sector_id"]) for e in state.ending.values()
        }
        key = (
            frozenset((str(c.id), str(c.status)) for c in searches),
            frozenset(state.open),
            frozenset(state.ending),
            int(now // GAPS_REFRESH_S),
        )
        if key != state.gaps_for:
            await _gaps(flow, mission, state, sectors, plane, counting, searches)
            state.gaps_for = key
    except Exception as exc:  # the rest of the vehicle's tick (link loss, looks) goes on
        log.error("coverage.tick_failed", mission_id=mid, error=type(exc).__name__)


def _follow(
    flow: MissionFlow,
    state: _Mission,
    searches: list[Command],
    flown_on: dict[str, dt.datetime],
    now: float,
) -> None:
    """Which searches count now (running, or flown on after a look), and which ended since
    the last tick or ended done unseen (after a restart): those count up to their end."""
    opened: dict[str, dt.datetime] = {}
    for c in searches:
        cid = str(c.id)
        if cid in state.finished:
            continue
        if str(c.status) == "running" or cid in flown_on:
            opened[cid] = max(c.created_at, flown_on.get(cid, c.created_at))
            if cid not in state.open:
                state.gaps.pop(str(c.params["sector_id"]), None)  # searched (again)
        elif cid not in state.ending and (cid in state.open or str(c.status) == "done"):
            # A run flown on after a look ends with the drone's report (this tick); the
            # search's own order with its last status.
            until = flow.now() if cid in state.flying_on else c.updated_at
            since = state.open.get(cid, c.created_at)
            state.ending[cid] = _Ending(c, since, until, now + CATCH_UP_S)
            # Worked out again from its end, also when it was never seen running (its
            # running and its done between two ticks).
            state.gaps.pop(str(c.params["sector_id"]), None)
    state.open = opened
    state.flying_on = {cid for cid in opened if cid in flown_on}


async def _count(
    flow: MissionFlow,
    mid: str,
    state: _Mission,
    by_id: dict[str, Command],
    sectors: dict[str, GeoJsonPolygon],
    plane: Plane,
    charter: MissionCharter,
    now: float,
) -> None:
    new_points = await flow.repo.track(mid, state.since)
    if new_points:
        state.since = new_points[-1]["t"]
    spans: list[tuple[Command, dt.datetime, dt.datetime | None]] = [
        (by_id[cid], since, None) for cid, since in state.open.items()
    ]
    spans += [(e.cmd, e.since, e.until) for e in state.ending.values()]
    for cmd, since, until in spans:
        sector_id = str(cmd.params["sector_id"])
        area = sectors.get(sector_id)
        if area is None:
            continue
        points = [
            (float(p["lat"]), float(p["lon"]), float(p["alt_rel_home_m"] or 0.0))
            for p in new_points
            if p["t"] >= since and (until is None or p["t"] <= until)
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
    # The track rows go in, in order, a batch at a time: a point past a search's end means
    # every point up to it is in and counted.
    for cid, e in list(state.ending.items()):
        if (state.since is not None and state.since >= e.until) or now >= e.deadline:
            if now >= e.deadline and (state.since is None or state.since < e.until):
                log.info("coverage.track_behind", mission_id=mid, command_id=cid)
            del state.ending[cid]
            if str(e.cmd.status) == "done":
                state.finished.add(cid)


async def _gaps(
    flow: MissionFlow,
    mission: Mission,
    state: _Mission,
    sectors: dict[str, GeoJsonPolygon],
    plane: Plane,
    counting: set[str],
    searches: list[Command],
) -> None:
    """Each sector that turned done, once no search of it is still being counted: its gap,
    told to the operator when there is one and it was not told already. A sector that is
    no longer done (searched again) is worked out again when it is."""
    mid = str(mission.id)
    for sid, status in (await flow.repo.sector_statuses(mid)).items():
        if status != "done" or sid in counting:  # searched (again): worked out at its end
            state.gaps.pop(sid, None)
            continue
        area = sectors.get(sid)
        if sid in state.gaps or area is None:
            continue
        try:
            poly = polygon_xy(area, plane)
            m2 = await asyncio.to_thread(gap_m2, poly, state.seen.get(sid))
            pct = 100.0 * m2 / poly.area if poly.area > 0 else 0.0
        except Exception as exc:  # worked out as none, not again every second
            log.error(
                "coverage.gap_failed", mission_id=mid, sector_id=sid, error=type(exc).__name__
            )
            state.gaps[sid] = (0.0, 0.0)
            continue
        newest = max(
            (c.created_at for c in searches if c.params.get("sector_id") == sid), default=None
        )
        told = m2 > 0.0 and await _told(flow, mid, sid, newest)
        state.gaps[sid] = (m2, pct)
        if m2 <= 0.0 or told:
            continue
        flow.alert(
            str(AlertCode.SECTOR_GAP),
            "warning",
            f"Sektor {sid} är klar men cirka {_m2_sv(m2)} m² ({pct:.0f} %) är inte sökt.",
            vehicle_id=mission.vehicle_id,
            mission_id=mid,
            details={"sector_id": sid, "unsearched_m2": round(m2), "unsearched_pct": round(pct, 1)},
        )


async def _told(flow: MissionFlow, mid: str, sid: str, since: dt.datetime | None) -> bool:
    """Whether the sector's gap was told since its newest search order (before a ground
    restart, say): the event log's ``alarm.SECTOR_GAP``."""
    repo = flow.g.repo
    if repo is None:
        return False
    await flow.g.flush_events()
    events = await repo.events(mid, since, f"alarm.{AlertCode.SECTOR_GAP}", 1000)
    return any(e.payload.get("sector_id") == sid for e in events)


def _m2_sv(m2: float) -> str:
    """Square metres the Swedish way, to the nearest hundred: "2 300"."""
    return f"{round(m2, -2):,.0f}".replace(",", " ")
