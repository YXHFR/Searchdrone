"""How much of each sector has been seen: ``mission_sectors.covered``, once a second.

The covered area is the union of discs of radius ``spacing_m / 2 + 2 m`` around the
telemetry points flown during a ``survey_sector`` order within ±3 m of its search height,
clipped to the sector: the same reach and height band as ``sd_geo.coverage``. The onboard
computer sends no coverage of its own. A search the drone flies on after a closer look
(``inspections.flown_on``: its order stays ``superseded`` on the ground) counts from the
drone's report that it flies on until that run ends. After a ground restart (the look's end
no longer in memory) such runs are rebuilt from the event log (``_restore``): from the
look's ``inspection.closed`` (``next`` resume) to the run's ``inspection.resume_ended``, or
to the next order the drone ran that moves it. A rebuilt run still flown also ends as the
live one does when that end is lost: when the drone's state no longer names the look's
resume (once seen, or not within ``inspections.CLOSED_WAIT_S`` of the look's end). Until the
runs are rebuilt, nothing is counted and no gap is worked out.

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
``CATCH_UP_S``), and its sector's gap is worked out only then. When the track had not
caught up by then, the alert says so (the gap may be smaller). Whether a gap was told is
read from the event log: an ``alarm.SECTOR_GAP`` naming the search whose end made the
sector done (``command_id``), so a ground restart neither tells it again nor leaves it
untold, and another search's alert never stands for it.

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
class _Rebuilt:
    """A run flown on after a look, rebuilt from the event log, whose end has not come:
    from ``start`` (the look's end), flown by the drone's resume ``resume_id``."""

    start: dt.datetime
    resume_id: str | None
    seen: bool = False  # the drone's state named the resume
    snapshot_ms: int | None = None  # the drone's state last looked at


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
    # Runs flown on after a look were rebuilt from the event log (once per process).
    restored: bool = False
    # Of those, the runs whose end has not come yet.
    restored_open: dict[str, _Rebuilt] = field(default_factory=dict[str, _Rebuilt])
    # Per sector, the search whose end was counted last: (its end, its order id).
    last_end: dict[str, tuple[dt.datetime, str]] = field(
        default_factory=dict[str, tuple[dt.datetime, str]]
    )
    # Sectors whose last end was worked out before the track caught up past it.
    behind: set[str] = field(default_factory=set[str])


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
    commands = await flow.repo.commands(mid, None)
    searches = [c for c in commands if c.skill == "survey_sector" and c.params.get("sector_id")]
    try:
        if not state.restored:
            await _restore(flow, mid, state, commands, searches, flown_on, now)
            state.restored = True
        if state.restored_open:
            await _restored_ends(flow, mission, state, commands, searches, flown_on, now)
    except Exception as exc:  # tried again on the next tick
        log.error("coverage.restore_failed", mission_id=mid, error=type(exc).__name__)
    if not state.restored:  # a run not rebuilt would count as unsearched: wait for it
        return
    rebuilt = {cid: run.start for cid, run in state.restored_open.items()}
    _follow(flow, state, searches, {**rebuilt, **flown_on}, now)
    plane = Plane(charter.home.lat, charter.home.lon)
    sectors = {s.sector_id: s.area for s in charter.sectors}
    by_id = {str(c.id): c for c in searches}
    try:
        if state.open or state.ending:
            await _count(flow, mid, state, by_id, sectors, plane, charter)
    except Exception as exc:  # the ends still close by their deadline (_expire)
        log.error("coverage.count_failed", mission_id=mid, error=type(exc).__name__)
    _expire(state, mid, now)
    try:
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
            await _gaps(flow, mission, state, sectors, plane, counting, by_id)
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
                _reopen(state, str(c.params["sector_id"]))  # searched (again)
        elif cid not in state.ending and (cid in state.open or str(c.status) == "done"):
            # A run flown on after a look ends with the drone's report (this tick); the
            # search's own order with its last status.
            until = flow.now() if cid in state.flying_on else c.updated_at
            since = state.open.get(cid, c.created_at)
            _ended(state, cid, _Ending(c, since, until, now + CATCH_UP_S))
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


def _expire(state: _Mission, mid: str, now: float) -> None:
    """The ends counted to their end. The track rows go in, in order, a batch at a time: a
    point past a search's end means every point up to it is in and counted. At its
    deadline an end closes all the same (the samples stopped, or the track could not be
    read or counted): its sector's gap is then marked as worked out with the track behind."""
    for key, e in list(state.ending.items()):
        caught_up = state.since is not None and state.since >= e.until
        if not caught_up and now < e.deadline:
            continue
        if not caught_up:
            log.warning("coverage.track_behind", mission_id=mid, command_id=str(e.cmd.id))
            state.behind.add(str(e.cmd.params["sector_id"]))
        del state.ending[key]
        if str(e.cmd.status) == "done":
            state.finished.add(str(e.cmd.id))


def _reopen(state: _Mission, sector_id: str) -> None:
    """A search of the sector starts or ends: its gap is worked out again from its end."""
    state.gaps.pop(sector_id, None)
    state.behind.discard(sector_id)


def _ended(state: _Mission, key: str, e: _Ending) -> None:
    """A search's end, counted up to it until the track has caught up (``_expire``). Its
    sector's gap is worked out again from this end, also when the search was never seen
    running (its running and its done between two ticks), and belongs to the search whose
    end came last."""
    sector_id = str(e.cmd.params["sector_id"])
    state.ending[key] = e
    _reopen(state, sector_id)
    last = state.last_end.get(sector_id)
    if last is None or e.until >= last[0]:
        state.last_end[sector_id] = (e.until, str(e.cmd.id))


# A run flown on after a look ends onboard with the next order the drone runs that moves it.
_RAN: Final = frozenset({"running", "done", "failed", "superseded"})


def _run_end(
    flow: MissionFlow,
    cid: str,
    start: dt.datetime,
    ends: list[Any],
    commands: list[Command],
) -> dt.datetime | None:
    """When a run flown on after a look, from ``start``, ended: its ``inspection.resume_ended``
    or the next order the drone ran that moves it, whichever came first; None: not yet."""
    told = [e.t for e in ends if e.payload.get("command_id") == cid and e.t >= start]
    later = [
        c.created_at
        for c in commands
        if c.created_at > start
        and str(c.status) in _RAN
        and (spec := flow.registry.get(c.skill)) is not None
        and spec.moves_vehicle
    ]
    return min([*told, *later], default=None)


async def _restore(
    flow: MissionFlow,
    mid: str,
    state: _Mission,
    commands: list[Command],
    searches: list[Command],
    flown_on: dict[str, dt.datetime],
    now: float,
) -> None:
    """Runs flown on after a look that this process did not see begin (a ground restart):
    from the event log. One that ended counts up to its end, one still flown counts on
    (``restored_open``) until its end comes."""
    repo = flow.g.repo
    if repo is None:
        return
    await flow.g.flush_events()
    closed = await repo.events(mid, None, "inspection.closed", 1000)
    ends = await repo.events(mid, None, "inspection.resume_ended", 1000)
    by_id = {str(c.id): c for c in searches}
    for e in closed:
        details = e.payload.get("details") or {}
        cid = str(details.get("resumed_command_id"))
        if details.get("next") != "resume" or cid not in by_id or cid in flown_on:
            continue
        until = _run_end(flow, cid, e.t, ends, commands)
        if until is None:
            resume = details.get("resume_command_id")
            state.restored_open[cid] = _Rebuilt(e.t, None if resume is None else str(resume))
            _reopen(state, str(by_id[cid].params["sector_id"]))
        else:
            key = f"{cid}@{e.t.isoformat()}"
            _ended(state, key, _Ending(by_id[cid], e.t, until, now + CATCH_UP_S))


async def _restored_ends(
    flow: MissionFlow,
    mission: Mission,
    state: _Mission,
    commands: list[Command],
    searches: list[Command],
    flown_on: dict[str, dt.datetime],
    now: float,
) -> None:
    """A rebuilt run still flown: counted on until its end comes (or the look's end is
    known in memory again: then it is followed from there). Its end is its
    ``inspection.resume_ended`` or the next order that moves the drone (``_run_end``), or,
    when that is lost, the drone's state as for a run followed live
    (``inspections.on_snapshot``): no longer naming the look's resume once it was seen, or
    never naming it within ``CLOSED_WAIT_S`` of the look's end."""
    from sd_ground.services import inspections as inspections_service

    repo = flow.g.repo
    if repo is None:
        return
    await flow.g.flush_events()
    first = min(run.start for run in state.restored_open.values())
    after = first - dt.timedelta(microseconds=1)  # the events from the earliest run's start
    ends = await repo.events(str(mission.id), after, "inspection.resume_ended", 1000)
    snap, snap_ms = flow.snapshot(mission.vehicle_id)
    by_id = {str(c.id): c for c in searches}
    for cid, run in list(state.restored_open.items()):
        until = None if cid in flown_on else _run_end(flow, cid, run.start, ends, commands)
        if (
            until is None
            and cid not in flown_on
            and snap is not None
            and snap_ms != run.snapshot_ms
        ):
            run.snapshot_ms = snap_ms  # each state report once, as when followed live
            active = None if snap.active_command_id is None else str(snap.active_command_id)
            wait = dt.timedelta(seconds=inspections_service.CLOSED_WAIT_S)
            if inspections_service.flies_resume(run.resume_id, active, snap.active_skill):
                run.seen = True
            elif run.seen or flow.now() >= run.start + wait:
                until = flow.now()
                log.info("coverage.rebuilt_run_left", command_id=cid, seen=run.seen)
        if cid in flown_on or until is not None:
            del state.restored_open[cid]
        if until is not None and cid in by_id:
            state.open.pop(cid, None)
            state.flying_on.discard(cid)
            _ended(state, cid, _Ending(by_id[cid], run.start, until, now + CATCH_UP_S))


async def _gaps(
    flow: MissionFlow,
    mission: Mission,
    state: _Mission,
    sectors: dict[str, GeoJsonPolygon],
    plane: Plane,
    counting: set[str],
    searches: dict[str, Command],
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
        last = state.last_end.get(sid)
        search = None if last is None else searches.get(last[1])
        search_id = None if last is None else last[1]
        since = None if search is None else search.created_at
        told = m2 > 0.0 and await _told(flow, mid, sid, search_id, since)
        state.gaps[sid] = (m2, pct)
        if m2 <= 0.0 or told:
            continue
        behind = sid in state.behind
        text = f"Sektor {sid} är klar men cirka {_m2_sv(m2)} m² ({pct:.0f} %) är inte sökt."
        if behind:
            text += " Spåret hann inte fram helt, så luckan kan vara mindre."
        flow.alert(
            str(AlertCode.SECTOR_GAP),
            "warning",
            text,
            vehicle_id=mission.vehicle_id,
            mission_id=mid,
            details={
                "sector_id": sid,
                "command_id": search_id,
                "unsearched_m2": round(m2),
                "unsearched_pct": round(pct, 1),
                "track_behind": behind,
            },
        )


async def _told(
    flow: MissionFlow, mid: str, sid: str, search_id: str | None, since: dt.datetime | None
) -> bool:
    """Whether the gap left by the search whose end made the sector done was told (before a
    ground restart, say): an ``alarm.SECTOR_GAP`` in the event log since that search was
    ordered (``since``) naming it. An alert written before alerts named their search (no
    ``command_id``) stands for it when it is of the sector and that recent."""
    repo = flow.g.repo
    if repo is None:
        return False
    await flow.g.flush_events()
    events = await repo.events(mid, since, f"alarm.{AlertCode.SECTOR_GAP}", 1000)
    return any(
        e.payload.get("sector_id") == sid
        and ("command_id" not in e.payload or e.payload["command_id"] == search_id)
        for e in events
    )


def _m2_sv(m2: float) -> str:
    """Square metres the Swedish way, to the nearest hundred: "2 300"."""
    return f"{round(m2, -2):,.0f}".replace(",", " ")
