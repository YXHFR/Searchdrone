"""A sector that turns done with a part no track point saw (``services.coverage``): the
ground tells the operator once (alert ``SECTOR_GAP``) and the AI in its situation, both
in square metres and as a share of the sector, over the real SD-link server with the
fake drone."""

from __future__ import annotations

import asyncio
import dataclasses
import datetime as dt
import itertools
import math
import sys
import uuid
from collections.abc import AsyncIterator, Callable
from pathlib import Path
from typing import TYPE_CHECKING, Any

import pytest
from shapely.geometry import Point
from structlog.testing import capture_logs

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "helpers"))
sys.path.insert(0, str(Path(__file__).resolve().parent))

from flow_rig import Rig, rig  # pyright: ignore[reportMissingImports]
from test_inspect_closed import _flown_on_a, _resume_ended  # pyright: ignore[reportMissingImports]
from test_orders import (  # pyright: ignore[reportMissingImports]
    driver,
    order,
    ready_with_sector,
    take_off,
    wait_order_status,
)

from sd_ground.services import coverage, inspections
from sd_ground.services.missions import MissionFlow
from sd_ground.services.telemetry import BATCH, SAMPLE_S
from sd_schemas.api import AlertCode

if TYPE_CHECKING:
    from conftest import Db

pytestmark = [pytest.mark.contract, pytest.mark.db]


@pytest.fixture
async def r(migrated_db: Db, tmp_path: Path) -> AsyncIterator[Rig]:
    async with rig(migrated_db.name, migrated_db.app_password, tmp_path) as built:
        yield built


def gap_alerts(r: Rig) -> list[dict[str, Any]]:
    return [
        e.data
        for e in r.g.bus.replay(r.g.bus.server_epoch, 0) or []
        if e.type == "alert" and e.data.get("code") == str(AlertCode.SECTOR_GAP)
    ]


async def fly(r: Rig, path: list[tuple[float, float]], step_m: float = 4.0) -> None:
    """The drone along ``path`` (east, north from home) at 16 m/s: a track point every 8 m
    at the ground's 2 Hz."""
    assert r.fc is not None
    for (e0, n0), (e1, n1) in itertools.pairwise(path):
        steps = max(1, math.ceil(math.dist((e0, n0), (e1, n1)) / step_m))
        for i in range(1, steps + 1):
            lat, lon = r.at(e0 + (e1 - e0) * i / steps, n0 + (n1 - n0) * i / steps)
            r.fc.lat, r.fc.lon = lat, lon
            r.fc.target = (lat, lon, r.fc.alt_rel)
            await asyncio.sleep(0.25)


def lawnmower(east: tuple[float, float], norths: list[float]) -> list[tuple[float, float]]:
    """Legs from ``east[0]`` to ``east[1]`` and back, one at each of ``norths``."""
    out: list[tuple[float, float]] = []
    for i, n in enumerate(norths):
        a, b = east if i % 2 == 0 else (east[1], east[0])
        out += [(a, n), (b, n)]
    return out


async def decided(r: Rig, mid: str, sector_id: str) -> tuple[float, float]:
    """The sector's gap once worked out (the track caught up past the search's end)."""
    await r.until(
        lambda: sector_id in r.flow.coverage.of(mid).gaps,
        coverage.CATCH_UP_S + 5,
        f"sector {sector_id}'s gap worked out",
    )
    return r.flow.coverage.of(mid).gaps[sector_id]


async def hover_search(r: Rig, before: Callable[[str], None] | None = None) -> tuple[str, Any, str]:
    """A search of sector A (80 x 80 m) running, the drone hovering in its middle until that
    is counted: the mission, the AI's driver and the order. ``before(mission)`` runs just
    before the order."""
    mid, _sector = await ready_with_sector(r)
    d = driver(r, mid)
    await take_off(r, mid, d)
    if before is not None:
        before(mid)
    r.onboard.order_outcome = "running"
    out = await order(r, d, "survey_sector", {"sector_id": "A", "alt_m": 40, "spacing_m": 20})
    sid = str(out["command_id"])
    await wait_order_status(r, mid, sid, "running")
    assert r.fc is not None
    r.fc.lat, r.fc.lon = r.at(50.0, 0.0)

    async def covered() -> bool:
        features = (await r.mission(mid))["sectors"]["features"]
        return next(f for f in features if f["id"] == "A")["properties"]["covered"] is not None

    await r.until(covered, 20, "hover point covered")
    return mid, d, sid


_TICK = coverage.tick  # the tick as built, for a test that pauses it


def paused(monkeypatch: pytest.MonkeyPatch) -> None:
    """No coverage tick runs (the ground is down or busy) until ``monkeypatch.undo()``."""

    async def down(*_: Any) -> None:
        return None

    monkeypatch.setattr(coverage, "tick", down)


# What a ground restart keeps of the mission flow: its wiring. What it held in memory is
# gone and built again from the database and the event log. Each field is one or the other
# (``restarted`` checks), so a field added to the flow is sorted here.
_WIRING = frozenset({"g", "repo", "caps", "registry", "link", "mav", "actions", "permit"})
_WIRING |= {"_lock", "_tasks", "_tlog_mission", "_tlog_done"}
_MEMORY = frozenset({"coverage", "clearances", "rates", "finds", "reports", "link_watch"})
_MEMORY |= {"looks", "resume_told", "_watch", "_status", "_release_waited"}


async def restarted(r: Rig) -> None:
    """The ground restarts: the mission flow's memory (coverage, looks, resumes told,
    statuses, ...) is that of a new flow, which then starts as at boot."""
    names = {f.name for f in dataclasses.fields(MissionFlow)}
    assert names == _WIRING | _MEMORY and not _WIRING & _MEMORY, names ^ (_WIRING | _MEMORY)
    fresh = MissionFlow(r.g, r.flow.repo)
    for name in _MEMORY:
        setattr(r.flow, name, getattr(fresh, name))
    await r.flow.start()


async def test_a_sector_done_with_an_unsearched_part_is_told_to_the_operator_and_the_ai(
    r: Rig,
) -> None:
    """The drone hovers in the middle of sector A (80 x 80 m) and the search ends done:
    only the disc it saw (20 / 2 + 2 m around the hover point, about 450 m²) is searched,
    so about 5 950 m² (93 %) is told as not searched."""
    mid, _sector = await ready_with_sector(r)
    d = driver(r, mid)
    await take_off(r, mid, d)
    r.onboard.order_outcome = "running"
    out = await order(r, d, "survey_sector", {"sector_id": "A", "alt_m": 40, "spacing_m": 20})
    sid = str(out["command_id"])
    await wait_order_status(r, mid, sid, "running")
    assert r.fc is not None
    r.fc.lat, r.fc.lon = r.at(50.0, 0.0)

    async def covered() -> bool:
        features = (await r.mission(mid))["sectors"]["features"]
        return next(f for f in features if f["id"] == "A")["properties"]["covered"] is not None

    await r.until(covered, 20, "hover point covered")
    assert not gap_alerts(r)  # nothing is told while the search runs

    assert r.onboard.session is not None
    await r.onboard.session.ack(uuid.UUID(sid), "done")
    await wait_order_status(r, mid, sid, "done")
    await r.until(lambda: gap_alerts(r), 10, "the gap told to the operator")

    (alert,) = gap_alerts(r)
    assert alert["severity"] == "warning"
    details = alert  # the details sit flat in the payload
    assert details["sector_id"] == "A"
    assert 5_500 <= details["unsearched_m2"] <= 6_100, details
    assert 86.0 <= details["unsearched_pct"] <= 95.0, details
    assert "Sektor A är klar men cirka" in alert["message_sv"]
    assert "m²" in alert["message_sv"] and "%" in alert["message_sv"]

    rows = {s["sector_id"]: s for s in (await d.situation())["sectors"]}
    assert rows["A"]["status"] == "done"
    assert rows["A"]["unsearched_m2"] == details["unsearched_m2"]
    assert rows["A"]["unsearched_pct"] == details["unsearched_pct"]

    # Told once: later ticks (one a second) leave it be.
    await asyncio.sleep(2.5)
    assert len(gap_alerts(r)) == 1

    # The ground restarts (its coverage memory is gone): the track is counted again and
    # the gap is kept for the AI, but the operator was told already.
    r.flow.coverage.forget(mid)

    async def rebuilt() -> bool:
        return r.flow.coverage.of(mid).gaps.get("A", (0.0, 0.0))[0] > 0

    await r.until(rebuilt, 10, "the gap worked out again")
    rows = {s["sector_id"]: s for s in (await d.situation())["sectors"]}
    assert rows["A"]["unsearched_m2"] == details["unsearched_m2"]
    await asyncio.sleep(1.5)
    assert len(gap_alerts(r)) == 1

    # Searched again: the sector is no longer done and the AI no longer hears of a gap;
    # done again with the same hole, the operator is told again.
    out = await order(r, d, "survey_sector", {"sector_id": "A", "alt_m": 40, "spacing_m": 20})
    again = str(out["command_id"])
    await wait_order_status(r, mid, again, "running")
    rows = {s["sector_id"]: s for s in (await d.situation())["sectors"]}
    assert rows["A"]["status"] == "active" and "unsearched_m2" not in rows["A"]
    await r.onboard.session.ack(uuid.UUID(again), "done")
    await wait_order_status(r, mid, again, "done")
    await r.until(lambda: len(gap_alerts(r)) == 2, coverage.CATCH_UP_S + 5, "told again")
    assert gap_alerts(r)[1]["unsearched_m2"] == details["unsearched_m2"]


async def test_a_search_again_between_two_ticks_is_worked_out_again_and_told(
    r: Rig, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The sector is searched again and its running and done both come while no tick runs,
    its track in the database past its end by the next tick: the sector's gap is worked
    out again from that end and told again (before, the old gap was kept as it was)."""
    mid, d, sid = await hover_search(r)
    assert r.onboard.session is not None
    await r.onboard.session.ack(uuid.UUID(sid), "done")
    await r.until(lambda: gap_alerts(r), coverage.CATCH_UP_S + 5, "the gap told")
    paused(monkeypatch)
    out = await order(r, d, "survey_sector", {"sector_id": "A", "alt_m": 40, "spacing_m": 20})
    again = str(out["command_id"])
    await wait_order_status(r, mid, again, "running")
    await r.onboard.session.ack(uuid.UUID(again), "done")
    await wait_order_status(r, mid, again, "done")
    await asyncio.sleep(BATCH * SAMPLE_S + 1.0)  # a batch of track past its end is in
    monkeypatch.undo()
    await r.until(lambda: len(gap_alerts(r)) == 2, coverage.CATCH_UP_S + 5, "told again")


async def test_a_complete_search_whose_last_points_come_after_its_end_has_no_gap(
    r: Rig,
) -> None:
    """The drone flies sector A in full and its done comes at once: the last seconds of the
    track reach the database only with the next batch of telemetry. They are waited for
    and counted, so no gap is told (before, the last leg was missing: a gap of about one
    leg)."""
    mid, _sector = await ready_with_sector(r)
    d = driver(r, mid)
    await take_off(r, mid, d)
    r.onboard.order_outcome = "running"
    out = await order(r, d, "survey_sector", {"sector_id": "A", "alt_m": 40, "spacing_m": 20})
    sid = str(out["command_id"])
    await wait_order_status(r, mid, sid, "running")
    await fly(r, lawnmower((10.0, 90.0), [-30.0, -10.0, 10.0, 30.0]))
    assert r.onboard.session is not None
    await r.onboard.session.ack(uuid.UUID(sid), "done")  # at the end of the last leg
    await wait_order_status(r, mid, sid, "done")
    assert await decided(r, mid, "A") == (0.0, 0.0)
    assert not gap_alerts(r)
    rows = {s["sector_id"]: s for s in (await d.situation())["sectors"]}
    assert rows["A"]["status"] == "done" and "unsearched_m2" not in rows["A"]


async def test_a_gap_not_told_before_a_ground_restart_is_told_after_it(
    r: Rig, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The search ends done with a hole and the ground restarts before telling it: the
    first ticks after the restart tell it, once (the event log says it was not told)."""
    mid, _d, sid = await hover_search(r)
    paused(monkeypatch)  # down: nothing counted, nothing told
    assert r.onboard.session is not None
    await r.onboard.session.ack(uuid.UUID(sid), "done")
    await wait_order_status(r, mid, sid, "done")
    r.flow.coverage.forget(mid)  # back up, its memory gone
    monkeypatch.undo()
    await r.until(lambda: gap_alerts(r), coverage.CATCH_UP_S + 5, "the gap told after the restart")
    await asyncio.sleep(2.5)
    (alert,) = gap_alerts(r)
    assert alert["sector_id"] == "A" and alert["unsearched_m2"] > 5_000


async def test_a_search_flown_on_after_a_look_and_ended_by_the_drone_counts_to_its_end(
    r: Rig,
) -> None:
    """The search broken off by a look is flown on by the drone's own resume over all of
    sector A, and its end (``inspect.resume_ended``) comes as the last leg ends: the points
    up to that end are counted once the track has caught up, so no gap is told."""
    mid, sid, sent, d, _resume = await _flown_on_a(r)
    await fly(r, lawnmower((30.0, 90.0), [-20.0, 0.0, 20.0]))
    await _resume_ended(r, sent, sid)
    rows = {s["sector_id"]: s for s in (await d.situation())["sectors"]}
    assert rows["A"]["status"] == "done"
    assert await decided(r, mid, "A") == (0.0, 0.0)
    assert not gap_alerts(r)


@pytest.mark.parametrize("when", ["while flown on", "after its end"])
async def test_a_search_flown_on_after_a_look_counts_to_its_end_across_a_ground_restart(
    r: Rig, monkeypatch: pytest.MonkeyPatch, when: str
) -> None:
    """As above, but the ground restarts while the drone flies the search on, or after its
    end and before its track was counted: the run is found again in the event log (the
    look's ``inspection.closed`` and the run's end), so its points count and no gap is told
    (before, the sector was told up to all of it unsearched)."""
    mid, sid, sent, _d, _resume = await _flown_on_a(r)
    if when == "while flown on":
        await fly(r, lawnmower((30.0, 90.0), [-20.0]))
        await restarted(r)
        await fly(r, lawnmower((30.0, 90.0), [0.0, 20.0]))
        await _resume_ended(r, sent, sid)
    else:
        await fly(r, lawnmower((30.0, 90.0), [-20.0, 0.0, 20.0]))
        paused(monkeypatch)
        await _resume_ended(r, sent, sid)
        await restarted(r)
        monkeypatch.undo()
    assert await decided(r, mid, "A") == (0.0, 0.0)
    await asyncio.sleep(2.5)
    assert not gap_alerts(r)


async def test_a_gap_told_for_an_older_search_after_a_new_order_does_not_hide_its_own(
    r: Rig,
) -> None:
    """The sector is ordered searched again before the first search's gap is told (the
    track still catching up), and that order runs later: the first gap is told, and the
    second search's gap too, each naming its own search (before, the first alert came
    after the newer order and was taken as its own)."""
    mid, d, sid = await hover_search(r)
    assert r.onboard.session is not None
    await r.onboard.session.ack(uuid.UUID(sid), "done")
    await wait_order_status(r, mid, sid, "done")
    r.onboard.order_outcome = "silent"  # sent, not yet answered
    out = await order(r, d, "survey_sector", {"sector_id": "A", "alt_m": 40, "spacing_m": 20})
    again = str(out["command_id"])
    await r.until(lambda: gap_alerts(r), coverage.CATCH_UP_S + 5, "the first gap told")
    assert gap_alerts(r)[0]["command_id"] == sid
    await r.onboard.session.ack(uuid.UUID(again), "accepted")
    await r.onboard.session.ack(uuid.UUID(again), "running")
    await wait_order_status(r, mid, again, "running")
    await r.onboard.session.ack(uuid.UUID(again), "done")
    await wait_order_status(r, mid, again, "done")
    await r.until(lambda: len(gap_alerts(r)) == 2, coverage.CATCH_UP_S + 5, "the second told")
    assert gap_alerts(r)[1]["command_id"] == again


async def test_a_search_order_the_drone_refuses_does_not_tell_the_gap_again(r: Rig) -> None:
    """The gap is told; then a new search of the sector is refused by the drone, so the
    sector stays done with the same gap, and the ground restarts: the gap is worked out
    again but not told again (before, any newer order of the sector made it count as
    untold)."""
    mid, d, sid = await hover_search(r)
    assert r.onboard.session is not None
    await r.onboard.session.ack(uuid.UUID(sid), "done")
    await r.until(lambda: gap_alerts(r), coverage.CATCH_UP_S + 5, "the gap told")
    r.onboard.order_outcome = "rejected"
    out = await order(r, d, "survey_sector", {"sector_id": "A", "alt_m": 40, "spacing_m": 20})
    await wait_order_status(r, mid, str(out["command_id"]), "rejected_onboard")
    await restarted(r)
    assert await decided(r, mid, "A") != (0.0, 0.0)
    await asyncio.sleep(2.5)
    assert len(gap_alerts(r)) == 1


async def test_a_track_that_cannot_be_counted_still_ends_by_the_deadline_and_says_so(
    r: Rig, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The track cannot be read or counted after the search ends: its end still closes at
    the deadline and the gap is told with what was counted, its alert saying the track was
    behind (so the hole may be smaller), and a warning is logged."""
    _mid, _d, sid = await hover_search(r)

    async def broken(*_: Any) -> None:
        raise RuntimeError("db")

    monkeypatch.setattr(coverage, "_count", broken)
    assert r.onboard.session is not None
    with capture_logs() as logs:
        await r.onboard.session.ack(uuid.UUID(sid), "done")
        await r.until(lambda: gap_alerts(r), coverage.CATCH_UP_S + 5, "the gap told")
    (alert,) = gap_alerts(r)
    assert alert["track_behind"] is True and alert["command_id"] == sid
    assert alert["unsearched_m2"] > 5_000
    assert "Spåret hann inte fram helt" in alert["message_sv"]
    behind = [e for e in logs if e["event"] == "coverage.track_behind"]
    assert behind and behind[0]["log_level"] == "warning", logs


async def test_a_rebuilt_run_ends_with_the_next_order_that_moves_the_drone(r: Rig) -> None:
    """The ground restarts while the drone flies a search on after a look. The drone then
    runs an order that moves it (a goto) before the run's end is told: the rebuilt run ends
    with that order, so the leg flown during it does not count, and the leg the search
    skipped is told once as its gap."""
    mid, sid, sent, d, _resume = await _flown_on_a(r)
    await fly(r, lawnmower((30.0, 90.0), [-20.0]))
    await restarted(r)
    await fly(r, lawnmower((30.0, 90.0), [0.0]))
    await _caught_up(r, mid)
    r.onboard.order_outcome = "silent"  # its state still names the resume (no other end)
    lat, lon = r.at(0.0, 60.0)
    out = await order(r, d, "goto", {"lat": lat, "lon": lon, "alt_m": 40})
    goto = str(out["command_id"])
    assert r.onboard.session is not None
    await r.onboard.session.ack(uuid.UUID(goto), "accepted")
    await r.onboard.session.ack(uuid.UUID(goto), "running")
    await wait_order_status(r, mid, goto, "running")
    await fly(r, lawnmower((30.0, 90.0), [20.0]))  # along the skipped leg, on the goto
    await _resume_ended(r, sent, sid)  # late: the sector is done
    m2, _pct = await decided(r, mid, "A")
    assert 600.0 < m2 < 1_600.0, m2  # the leg at 20 m north: about 18 x 60 m
    await asyncio.sleep(2.5)
    (alert,) = gap_alerts(r)
    assert alert["command_id"] == sid


@pytest.mark.parametrize("seen", [True, False])
async def test_a_rebuilt_run_whose_end_is_lost_ends_when_the_drone_leaves_its_resume(
    r: Rig, monkeypatch: pytest.MonkeyPatch, seen: bool
) -> None:
    """The ground restarts while the drone flies a search on after a look, and the run's
    end never comes. As for a run followed live, the drone's state ends it: no longer
    naming the look's resume once it did, or never naming it within ``CLOSED_WAIT_S`` of
    the look's end. What it flies after that (home, by hand) at search height over the
    sector does not count as searched (before, the run counted on for ever)."""
    mid, sid, _sent, _d, _resume = await _flown_on_a(r)
    await fly(r, lawnmower((30.0, 90.0), [-20.0]))
    if not seen:
        r.onboard.idle()  # the resume already over, its end lost
        monkeypatch.setattr(inspections, "CLOSED_WAIT_S", 0.0)
    await restarted(r)
    state = r.flow.coverage.of(mid)
    if seen:
        await r.until(
            lambda: sid in state.restored_open and state.restored_open[sid].seen,
            5,
            "the resume named by the drone",
        )
        r.onboard.idle()  # it flies home
    await r.until(lambda: not state.restored_open, 5, "the rebuilt run ended")
    await fly(r, lawnmower((30.0, 90.0), [0.0, 20.0]))
    await asyncio.sleep(BATCH * SAMPLE_S + 2 * coverage.EVERY_S)  # its batches in and counted
    seen_area = state.seen.get("A")
    for north in (0.0, 20.0):
        x, y = r.plane.xy(*r.at(60.0, north))
        assert seen_area is None or not seen_area.contains(Point(x, y)), north


async def test_no_gap_is_worked_out_until_the_runs_flown_on_are_rebuilt(
    r: Rig, monkeypatch: pytest.MonkeyPatch
) -> None:
    """After a restart the event log cannot be read for a while: nothing is counted and no
    gap worked out until the runs flown on after a look are rebuilt (before, the sector
    was told all unsearched and the count moved past the run's points)."""
    mid, sid, sent, _d, _resume = await _flown_on_a(r)
    await fly(r, lawnmower((30.0, 90.0), [-20.0, 0.0, 20.0]))
    paused(monkeypatch)
    await _resume_ended(r, sent, sid)
    restore = coverage._restore  # pyright: ignore[reportPrivateUsage]

    async def unreadable(*_: Any) -> None:
        raise RuntimeError("db")

    monkeypatch.setattr(coverage, "_restore", unreadable)
    monkeypatch.setattr(coverage, "tick", _TICK)
    await restarted(r)
    await asyncio.sleep(3.0)
    state = r.flow.coverage.of(mid)
    assert "A" not in state.gaps and not state.seen and state.since is None
    assert not gap_alerts(r)
    monkeypatch.setattr(coverage, "_restore", restore)
    assert await decided(r, mid, "A") == (0.0, 0.0)
    await asyncio.sleep(2.5)
    assert not gap_alerts(r)


async def _caught_up(r: Rig, mid: str) -> None:
    """The track counted up to now (its batches in)."""
    now = r.flow.now() - dt.timedelta(seconds=SAMPLE_S)
    state = r.flow.coverage.of(mid)
    await r.until(
        lambda: state.since is not None and state.since >= now,
        coverage.CATCH_UP_S,
        "the track counted",
    )


@pytest.mark.parametrize("ordered", ["after", "before"])
async def test_a_gap_told_before_alerts_named_their_search_is_not_told_again(
    r: Rig, ordered: str
) -> None:
    """A ``SECTOR_GAP`` written before an upgrade has no ``command_id``. Written after the
    search was ordered, it stands for that search's gap and the gap is not told again; one
    written before the order is of an older search and does not."""

    def legacy(mid: str) -> None:
        payload = {"code": str(AlertCode.SECTOR_GAP), "severity": "warning", "sector_id": "A"}
        payload["message_sv"] = "Sektor A är klar men cirka 6 400 m² (100 %) är inte sökt."
        r.g.add_event(
            f"alarm.{AlertCode.SECTOR_GAP}", severity="warning", mission_id=mid, payload=payload
        )

    mid, _d, sid = await hover_search(r, legacy if ordered == "before" else None)
    if ordered == "after":
        legacy(mid)
    assert r.onboard.session is not None
    await r.onboard.session.ack(uuid.UUID(sid), "done")
    assert await decided(r, mid, "A") != (0.0, 0.0)
    await asyncio.sleep(2.5)
    assert len(gap_alerts(r)) == (1 if ordered == "before" else 0)
