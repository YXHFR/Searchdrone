"""Owner decisions 2026-11-03 on the cylinders of unconfirmed persons, onboard.

A drone's own cylinder needs a second look (two stills that agree on the place) unless one
detection is very sure; a cylinder the operator rejected goes; a search in sim or an
exercise (overflight allowed) passes straight through an unconfirmed cylinder, in live
without the permit it goes around.
"""

from __future__ import annotations

import asyncio
import datetime as dt
import json
import math
import random
from collections.abc import AsyncIterator
from contextlib import asynccontextmanager
from pathlib import Path
from types import SimpleNamespace
from typing import Any
from uuid import UUID

import pytest
from onboard_helpers import PLANE, Rig, at, charter, rig, square
from test_inspect import (
    SKILLS,
    _exiting,  # pyright: ignore[reportPrivateUsage]
    _outside_all,  # pyright: ignore[reportPrivateUsage]
    _search_over_a_person,  # pyright: ignore[reportPrivateUsage]
    detection,
    install,
    keepout,
    track_xy,
    xy,
)

from sd_mav.mission import MISSION
from sd_mav.modes import CopterMode
from sd_onboard.executor import executor as executor_module
from sd_onboard.executor.keepout_reflex import remaining
from sd_onboard.io.persistence import StateStore
from sd_onboard.skills import goto as goto_module
from sd_onboard.skills.goto import GotoHandler
from sd_onboard.skills.resume import ResumeHandler
from sd_onboard.skills.survey_sector import BLOCKED_MIN_M, SurveyRun
from sd_safety.constants import (
    GOTO_ALT_STEP_TOL_M,
    KEEPOUT_PAIR_MIN_M,
    KEEPOUT_PAIR_WINDOW_S,
    KEEPOUT_PASS_GAP_S,
    KEEPOUT_PASS_MAX_S,
    KEEPOUT_RESTART_CAUTION_S,
    KEEPOUT_SCORE_SURE,
)
from sd_schemas.base import LatLon, LatLonAlt, new_id
from sd_schemas.executor import ExecutorState as S
from sd_schemas.executor import SurveyResumePoint
from sd_schemas.keepout import Keepout, leg_enters
from sd_schemas.link import KeepoutInstall
from sd_schemas.skills import ResumeParams

# --- a second look before a cylinder ----------------------------------------------------


async def _ready(r: Rig) -> None:
    await r.install(charter(allowed_skills=SKILLS))


async def test_one_unsure_detection_makes_no_cylinder_and_a_second_still_does() -> None:
    async with rig() as r:
        await _ready(r)
        first = detection(50, 120, 0.6)
        r.ex.on_detections([first])
        assert r.ex.local_keepouts == {}
        second = detection(53, 122, 0.6)
        r.ex.on_detections([second])
        ((k, _),) = r.ex.local_keepouts.values()
        assert k.source == "detection" and k.source_id == second.detection_id
        assert "executor.keepout_created" in r.up.kinds()


async def test_two_detections_in_one_still_are_not_a_second_look() -> None:
    async with rig() as r:
        await _ready(r)
        r.ex.on_detections([detection(50, 120, 0.6), detection(52, 120, 0.6)])
        assert r.ex.local_keepouts == {}


async def test_a_second_look_must_be_near_and_recent() -> None:
    async with rig() as r:
        await _ready(r)
        r.ex.on_detections([detection(50, 120, 0.6, sigma=2.0)])
        r.ex.on_detections([detection(50 + KEEPOUT_PAIR_MIN_M + 5, 120, 0.6, sigma=2.0)])
        assert r.ex.local_keepouts == {}  # too far apart: another place
        r.ex.on_detections([detection(-80, 0, 0.6)])
        await r.run(KEEPOUT_PAIR_WINDOW_S + 1)
        r.ex.on_detections([detection(-80, 2, 0.6)])
        assert r.ex.local_keepouts == {}  # the first look is too old


async def test_a_very_sure_detection_makes_a_cylinder_at_once() -> None:
    async with rig() as r:
        await _ready(r)
        r.ex.on_detections([detection(50, 120, KEEPOUT_SCORE_SURE)])
        assert len(r.ex.local_keepouts) == 1
        r.ex.on_detections([detection(-50, 120, KEEPOUT_SCORE_SURE - 0.01)])
        assert len(r.ex.local_keepouts) == 1  # just under: waits for a second look


# --- the operator rejects the candidate -------------------------------------------------


async def test_a_rejected_candidate_takes_the_drones_own_cylinder_with_it() -> None:
    """Decision 5: the ground names the detections of rejected candidates
    (``KeepoutInstall.released``); a cylinder made from any of them goes."""
    async with rig() as r:
        await _ready(r)
        first, second = detection(50, 120, 0.6), detection(52, 121, 0.6)
        r.ex.on_detections([first])
        r.ex.on_detections([second])
        sure = detection(-60, 40, 0.9)
        r.ex.on_detections([sure])
        assert len(r.ex.local_keepouts) == 2
        seq = r.ex.keepout_seq + 1
        unrelated = KeepoutInstall(install_seq=seq, keepouts=[], released=[new_id()])
        r.ex.on_keepouts(unrelated, new_id())
        assert len(r.ex.local_keepouts) == 2
        # The first look's detection is enough: it is part of the cylinder.
        rejected = KeepoutInstall(install_seq=seq + 1, keepouts=[], released=[first.detection_id])
        r.ex.on_keepouts(rejected, new_id())
        ((k, _),) = r.ex.local_keepouts.values()
        assert k.source_id == sure.detection_id
        released = [d for t, _, d in r.up.events if t == "executor.keepout_released"]
        assert [d["detection_id"] for d in released] == [str(second.detection_id)]
        # A released detection never pairs again.
        r.ex.on_detections([detection(50, 120, 0.6)])
        rejected2 = KeepoutInstall(
            install_seq=seq + 2, keepouts=[], released=[first.detection_id, second.detection_id]
        )
        r.ex.on_keepouts(rejected2, new_id())
        assert len(r.ex.local_keepouts) == 1


# --- a search passes an unconfirmed cylinder only with overflight -----------------------


def _sim_charter() -> object:
    return charter(allowed_skills=SKILLS, operation_mode="sim", overflight_allowed=True)


async def _cylinder_on_the_leg(r: Rig, source: str) -> tuple[tuple[float, float], float]:
    box = square(60, east=0, north=120)
    await r.takeoff()
    await r.send(r.order("survey_sector", {"polygon": box, "spacing_m": 30, "speed_ms": 8}), 1)

    def to() -> tuple[float, float]:
        ahead = r.fc.missions[MISSION][r.fc.mission_current]
        return PLANE.xy(ahead.x / 1e7, ahead.y / 1e7)

    # On a long search leg, with room past the cylinder to its end (no corner inside).
    await r.until(lambda: r.fc.mission_current >= 4 and math.dist(xy(r), to()) > 80, 300)
    here, end = xy(r), to()
    d = math.dist(here, end)
    centre = (here[0] + 25 * (end[0] - here[0]) / d, here[1] + 25 * (end[1] - here[1]) / d)
    install(r, keepout(*centre, 10.0, source=source))
    return centre, 10.0


async def _fly_and_watch(
    r: Rig, centre: tuple[float, float], radius: float
) -> tuple[list[float], float]:
    """Fly the search to its end: the times (s) of each stay inside the cylinder, and the
    lowest speed while inside (from the track: the fake reports no groundspeed)."""
    stays: list[float] = []
    since: float | None = None
    slowest = math.inf
    t0 = r.clock.monotonic()
    last = xy(r)
    while r.ex.state == S.SURVEY:
        await r.run(0.1)
        p = xy(r)
        speed, last = math.dist(p, last) / 0.1, p
        assert r.clock.monotonic() - t0 < 600, "the search never ended"
        inside = math.dist(xy(r), centre) < radius
        now = r.clock.monotonic()
        if inside:
            since = now if since is None else since
            slowest = min(slowest, speed)
        elif since is not None:
            stays.append(now - since)
            since = None
    return stays, slowest


async def test_in_sim_a_search_flies_straight_through_an_unconfirmed_cylinder() -> None:
    async with rig() as r:
        await r.install(_sim_charter())  # type: ignore[arg-type]
        centre, radius = await _cylinder_on_the_leg(r, "candidate")
        stays, slowest = await _fly_and_watch(r, centre, radius)
        kinds = r.up.kinds()
        assert "executor.keepout_reroute" not in kinds and "executor.keepout_hold" not in kinds
        assert "executor.survey_blocked" not in kinds  # searched through: nothing left out
        assert len(stays) == 1, stays  # through once
        assert stays[0] <= KEEPOUT_PASS_MAX_S
        assert slowest > 3.0  # at search speed: no hover, no turn inside


@pytest.mark.parametrize(
    ("sim", "source"), [(False, "candidate"), (False, "detection"), (True, "confirmed")]
)
async def test_live_without_the_permit_or_a_confirmed_find_is_gone_around(
    sim: bool, source: str
) -> None:
    async with rig() as r:
        await r.install(_sim_charter() if sim else charter(allowed_skills=SKILLS))  # type: ignore[arg-type]
        centre, radius = await _cylinder_on_the_leg(r, source)
        stays, _ = await _fly_and_watch(r, centre, radius)
        assert "executor.keepout_reroute" in r.up.kinds()
        assert stays == []
        blocked = [d["blocked"] for t, _, d in r.up.events if t == "executor.survey_blocked"]
        if source == "confirmed":
            assert blocked == []  # a find: nobody has to look at it before the search goes on
        else:
            # A possible person nobody has looked at: the search left out the leg through
            # its cylinder (at most its diameter, 20 m), and says whose cylinder it is.
            assert blocked, r.up.kinds()
            (b,) = blocked[-1]
            assert b["source"] == source
            assert BLOCKED_MIN_M <= b["skipped_m"] <= 20.0, b


def test_a_first_leg_that_comes_down_gets_no_passage() -> None:
    run = SurveyRun(new_id(), [], 40.0, 7.0, skill="survey_sector")
    sim = _sim_charter()

    def ctx(alt: float) -> SimpleNamespace:
        return SimpleNamespace(vehicle=SimpleNamespace(alt_rel_home_m=alt), charter=sim)

    assert run.passage(ctx(40.0)) is not None  # type: ignore[arg-type]
    assert run.passage(ctx(40.0 + GOTO_ALT_STEP_TOL_M + 0.1)) is None  # type: ignore[arg-type]
    assert SurveyRun(new_id(), [], 40.0, 7.0).passage(ctx(40.0)) is None  # type: ignore[arg-type]
    live = SimpleNamespace(vehicle=SimpleNamespace(alt_rel_home_m=40.0), charter=charter())
    assert run.passage(live) is None  # type: ignore[arg-type]


# --- executor.survey_blocked: when it is sent, and when not (review of #215) -------------


class _Io:
    """Just enough of ``ExecutorIO`` for ``SurveyRun.fly`` to upload, switch to AUTO and
    see the last point passed at once; it keeps the events and the order of the calls."""

    def __init__(self) -> None:
        self.calls: list[str] = []
        self.events: list[tuple[str, dict[str, Any]]] = []
        self.clock = SimpleNamespace(monotonic=lambda: 0.0)
        self.vehicle = SimpleNamespace(mission_seq_current=10_000)

    async def upload_mission(self, items: object) -> str:
        self.calls.append("upload")
        return "hash"

    def emit_event(self, type: str, data: dict[str, Any]) -> None:
        self.calls.append(type)
        self.events.append((type, data))

    async def set_mission_current(self, seq: int) -> None:
        self.calls.append("current")

    async def set_mode(self, mode: object) -> bool:
        self.calls.append(f"mode:{mode}")
        return True


class _IoThatCannotSend(_Io):
    """An ``ExecutorIO`` whose ``survey_blocked`` send fails (the link's queue is full,
    a bug in the encoder)."""

    def emit_event(self, type: str, data: dict[str, Any]) -> None:
        if type == "survey_blocked":
            raise RuntimeError("the send failed")
        super().emit_event(type, data)


async def _plan_and_fly(
    skill: str | None,
    centre: tuple[float, float],
    radius: float,
    source: str = "candidate",
    io: _Io | None = None,
) -> tuple[SurveyRun, _Io]:
    """A search route from home along y=0 to (100, 0) and up to (100, 100), live (no
    passage), with one cylinder; planned around it and flown."""
    pts = [LatLon(lat=la, lon=lo) for la, lo in (PLANE.latlon(100, 0), PLANE.latlon(100, 100))]
    run = SurveyRun(new_id(), pts, 40.0, 7.0, skill=skill)
    lat, lon = PLANE.latlon(0, 0)
    ctx = SimpleNamespace(
        vehicle=SimpleNamespace(lat=lat, lon=lon, alt_rel_home_m=40.0),
        charter=charter(),
        plane=PLANE,
        keepouts=(keepout(*centre, radius, source=source),),
        keepout_inside_s={},
    )
    run.around_keepouts(ctx)  # type: ignore[arg-type]
    assert run.detours()  # it does go around (a point of its own on the way)
    io = io or _Io()
    home = LatLonAlt(lat=lat, lon=lon, alt_amsl_m=0.0)
    await run.fly(SimpleNamespace(expected_duration_s=1.0), io, home)  # type: ignore[arg-type]
    return run, io


def _blocked(io: _Io) -> list[dict[str, Any]]:
    return [d for t, d in io.events if t == "survey_blocked"]


async def test_a_way_around_through_the_middle_of_a_leg_is_sent_after_auto() -> None:
    """The positive case the others are measured against: the leg crosses the cylinder,
    the way around leaves out about its diameter, and the event goes out once the route
    is up and AUTO is set (an observation after the flight action, never before it)."""
    _, io = await _plan_and_fly("survey_sector", (50.0, 0.0), 10.0)
    (event,) = _blocked(io)
    (b,) = event["blocked"]
    assert b["source"] == "candidate" and BLOCKED_MIN_M <= b["skipped_m"] <= 20.0
    assert io.calls.index("survey_blocked") > io.calls.index(f"mode:{CopterMode.AUTO}")
    assert io.calls.index("survey_blocked") > io.calls.index("upload")


async def test_a_corner_clip_under_the_minimum_is_not_a_blocked_search() -> None:
    """The cylinder only clips the corner at (100, 0): 4 m of the route inside it (2 m on
    each leg). The route goes around, but nothing is left out worth a review first."""
    run, io = await _plan_and_fly("survey_sector", (106.0, -6.0), 10.0)
    assert run.blocked == [] and _blocked(io) == []


@pytest.mark.parametrize("skill", [None, "goto", "orbit"])
async def test_a_route_that_is_not_a_search_sends_no_blocked_search(skill: str | None) -> None:
    run, io = await _plan_and_fly(skill, (50.0, 0.0), 10.0)
    assert run.blocked == [] and _blocked(io) == []


async def test_a_confirmed_find_does_not_block_the_search() -> None:
    _, io = await _plan_and_fly("survey_sector", (50.0, 0.0), 10.0, source="confirmed")
    assert _blocked(io) == []


async def test_a_failure_working_out_the_blocking_never_stops_the_way_around(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    def broken(*_: object) -> list[object]:
        raise ValueError("broken")

    monkeypatch.setattr(SurveyRun, "_blocked", broken)
    _, io = await _plan_and_fly("survey_sector", (50.0, 0.0), 10.0)
    assert _blocked(io) == []
    assert f"mode:{CopterMode.AUTO}" in io.calls


async def test_a_failed_send_of_the_blocking_never_stops_the_way_around() -> None:
    """The blocking is worked out, but sending it fails: the route still flies on in
    AUTO to its end, and nothing goes out."""
    io = _IoThatCannotSend()
    run, _ = await _plan_and_fly("survey_sector", (50.0, 0.0), 10.0, io=io)
    assert run.blocked  # there was something to send
    assert _blocked(io) == []
    assert io.calls.index(f"mode:{CopterMode.AUTO}") > io.calls.index("upload")


# --- the time inside, and the way out (review of #202) -----------------------------------


async def test_the_time_inside_an_unconfirmed_cylinder_counts_from_when_the_drone_came_in() -> None:
    """A route planned mid-passage counts the time the drone spent inside already. A
    cylinder that replaces the one the drone is in (grown, handed over) is the same
    person: no fresh budget. Confirmed cylinders are never passed: no time kept."""
    async with rig() as r:
        await r.install(_sim_charter())  # type: ignore[arg-type]
        await r.takeoff()
        here = xy(r)
        first = keepout(*here, 40.0, source="candidate")
        install(r, first)
        await r.run(3.0)
        inside = r.ex._keepout_inside_s()  # pyright: ignore[reportPrivateUsage]
        assert set(inside) == {first.keepout_id}
        assert inside[first.keepout_id] == pytest.approx(3.0, abs=0.3)
        handed = keepout(*here, 45.0, source="detection")
        install(r, handed)
        await r.run(1.0)
        inside = r.ex._keepout_inside_s()  # pyright: ignore[reportPrivateUsage]
        assert set(inside) == {handed.keepout_id}
        assert inside[handed.keepout_id] >= 3.8
        # The plan and the gate see the same time.
        ctx = r.ex._plan_context(new_id())  # pyright: ignore[reportPrivateUsage]
        assert ctx.keepout_inside_s[handed.keepout_id] >= 3.8
        gate = r.ex._gate_context(r.ex.vehicle_state())  # pyright: ignore[reportPrivateUsage]
        assert gate.keepout_inside_s[handed.keepout_id] >= 3.8
        install(r, keepout(*here, 45.0))  # confirmed
        await r.run(0.2)
        assert r.ex._keepout_inside_s() == {}  # pyright: ignore[reportPrivateUsage]
        # Out of every cylinder and back into a new one: from 0 again.
        install(r)
        await r.run(0.2)
        again = keepout(*here, 40.0, source="candidate")
        install(r, again)
        await r.run(1.0)
        inside = r.ex._keepout_inside_s()  # pyright: ignore[reportPrivateUsage]
        assert inside[again.keepout_id] == pytest.approx(1.0, abs=0.3)


async def test_the_brake_before_the_way_out_holds_where_the_drone_stops(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Review of #202: the brake is a goto ``from_stop``: its spot follows the drone until
    it stands still, never the point where the brake began (overshot at speed, then flown
    back to, maybe deeper into the cylinder)."""
    monkeypatch.setattr(executor_module, "still", lambda _v: False)
    monkeypatch.setattr(goto_module, "still", lambda _v: False)  # it moves on all the while
    async with rig() as r:
        await _search_over_a_person(r, monkeypatch)
        await _exiting(r)
        assert r.ex._exit_braking is not None  # pyright: ignore[reportPrivateUsage]
        active = r.ex.active
        assert active is not None and isinstance(active.handler, GotoHandler)
        handler = active.handler
        assert handler._stopping  # pyright: ignore[reportPrivateUsage]
        await r.run(0.3)
        assert handler.target is not None
        spot = PLANE.xy(handler.target[0], handler.target[1])
        assert math.dist(spot, xy(r)) <= 2.0  # where the drone is, not behind it


async def test_a_confirmed_cylinder_on_the_way_out_makes_it_plan_the_way_out_again(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Review of #202: the way out is an internal goto the reflex used to skip. A
    confirmed cylinder that comes over the rest of it (here over its end) stops the drone
    and the way out is planned again from there, into no confirmed cylinder."""
    async with rig() as r:
        _, person, _, start = await _search_over_a_person(r, monkeypatch, radius=40.0)
        await _exiting(r)
        plane = r.ex.plane()
        active = r.ex.active
        assert plane is not None and active is not None
        now = xy(r)
        end = remaining(active.handler, active.plan, now, plane)[-1]
        d = math.dist(now, end)
        centre = (end[0] + 5.0 * (end[0] - now[0]) / d, end[1] + 5.0 * (end[1] - now[1]) / d)
        found = keepout(*centre, 15.0)  # confirmed, over the end of the way out
        assert math.dist(now, centre) > 15.0
        install(r, person, found)
        await r.until(lambda: "executor.keepout_exit_replan" in r.up.kinds(), 10)
        await r.until(lambda: r.ex.state == S.HOLD, 120)
        assert _outside_all(r) and r.ex.hold_reason == "limit"
        assert all(math.dist(p, centre) >= found.radius_m - 0.5 for p in track_xy(r, start))


async def test_an_unconfirmed_cylinder_that_is_confirmed_on_the_way_out_is_not_crossed(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The way out may cross another person's unconfirmed cylinder; confirmed while the
    drone flies it, that one is never entered."""
    async with rig() as r:
        _, person, _, _ = await _search_over_a_person(r, monkeypatch, radius=40.0)
        await _exiting(r)
        plane = r.ex.plane()
        active = r.ex.active
        assert plane is not None and active is not None
        now = xy(r)
        end = remaining(active.handler, active.plan, now, plane)[-1]
        d = math.dist(now, end)
        u = ((end[0] - now[0]) / d, (end[1] - now[1]) / d)
        # Across the rest of the way out, ahead of the drone and clear of it.
        centre = (now[0] + 30.0 * u[0], now[1] + 30.0 * u[1])
        other = keepout(*centre, 10.0, source="candidate")
        install(r, person, other)
        await r.run(0.2)
        assert "executor.keepout_exit_replan" not in r.up.kinds()  # unconfirmed: crossed
        mark = len(r.fc.track)
        assert math.dist(xy(r), centre) > other.radius_m
        install(r, person, other.model_copy(update={"source": "confirmed"}))
        await r.until(lambda: "executor.keepout_exit_replan" in r.up.kinds(), 10)
        await r.until(lambda: r.ex.state == S.HOLD, 120)
        assert all(math.dist(p, centre) >= other.radius_m - 0.5 for p in track_xy(r, mark))
        assert _outside_all(r)


# --- after a restart: the time inside is unknown ----------------------------------------


def _saved_mid_search(path: Path, *keepouts: Keepout) -> None:
    """What a search interrupted at home saved: its rest (east through the cylinder
    around home, then north) and the cylinders it had taken (none: no install)."""
    here = LatLon(lat=PLANE.latlon(0, 0)[0], lon=PLANE.latlon(0, 0)[1])
    rp = SurveyResumePoint(
        command_id=new_id(),
        waypoints=[LatLon(lat=la, lon=lo) for la, lo in (at(-150, 0), at(150, 0), at(150, 80))],
        alt_m=40,
        speed_ms=7,
        base_index=0,
        seq_offset=0,
        next_index=1,
        interrupted_at=here,
    )
    taken = KeepoutInstall(install_seq=5, keepouts=list(keepouts)) if keepouts else None
    path.mkdir(parents=True, exist_ok=True)
    (path / "executor_state.json").write_text(
        json.dumps(
            {
                "schema": 2,
                "vehicle_id": "sim-01",
                "mission_id": None,
                "boot_id": 7,
                "state": "SURVEY",
                "hold_reason": None,
                "held_by": None,
                "expected_mode": "AUTO",
                "charter_id": None,
                "charter_version": None,
                "active_command_id": None,
                "resume_point": rp.model_dump(mode="json"),
                "keepouts": taken.model_dump(mode="json") if taken else None,
                "saved_mono_ms": 1,
                "saved_utc": "2026-11-04T10:00:00Z",
            }
        )
    )


@asynccontextmanager
async def _restarted_in_the_air(tmp_path: Path, *keepouts: Keepout) -> AsyncIterator[Rig]:
    """sd-onboard started again over home at 40 m mid-search, ``keepouts`` saved; then
    the ground's charter (sim, overflight allowed)."""
    _saved_mid_search(tmp_path, *keepouts)
    async with rig(start=False) as r:
        r.ex.store = StateStore(tmp_path, r.clock)
        r.fc.armed, r.fc.alt_rel, r.fc.mode = True, 40.0, CopterMode.AUTO
        r.fc.target = (r.fc.lat, r.fc.lon, 40.0)
        r.background.append(asyncio.ensure_future(r.ex.run(r.stop)))
        await r.until(lambda: r.ex._started, 10)  # pyright: ignore[reportPrivateUsage]
        assert r.ex.state == S.HELD and r.ex.held_by == "onboard_restart"
        await r.install(_sim_charter())  # type: ignore[arg-type]
        yield r


def _inside_s(r: Rig) -> dict[UUID, float]:
    return dict(r.ex._keepout_inside_s())  # pyright: ignore[reportPrivateUsage]


def _resume_path(r: Rig) -> list[tuple[float, float]]:
    """The search resumed from where the drone is, as the executor plans it."""
    ctx = r.ex._plan_context(new_id())  # pyright: ignore[reportPrivateUsage]
    plan = ResumeHandler().plan(ResumeParams(), ctx)
    return [PLANE.xy(p.lat, p.lon) for p in plan.path]


UNKNOWN = KEEPOUT_PASS_MAX_S  # unknown time counts as the whole budget used


async def test_after_a_restart_inside_a_cylinder_a_resume_may_only_leave_it(
    tmp_path: Path,
) -> None:
    """An onboard restart mid-passage forgets when the drone came in: its time inside is
    unknown (no fresh 12 s; the whole budget counts as used), and the search resumed from
    there may only leave the cylinder first, never go deeper. The same cylinder coming
    around the drone in a flight without a restart counts from 0."""
    c, radius = (20.0, 0.0), 40.0  # home is 20 m from its centre
    person = keepout(*c, radius, source="candidate")
    async with _restarted_in_the_air(tmp_path, person) as r:
        assert math.dist(xy(r), c) < radius
        await r.run(1.0)
        assert _inside_s(r) == {person.keepout_id: UNKNOWN}
        ctx = r.ex._plan_context(new_id())  # pyright: ignore[reportPrivateUsage]
        assert dict(ctx.keepout_inside_s) == {person.keepout_id: UNKNOWN}
        gate = r.ex._gate_context(r.ex.vehicle_state())  # pyright: ignore[reportPrivateUsage]
        assert dict(gate.keepout_inside_s) == {person.keepout_id: UNKNOWN}
        path = _resume_path(r)
        assert not leg_enters(path[0], path[1], c, radius)  # out first, no deeper
    async with rig() as r:
        await r.install(_sim_charter())  # type: ignore[arg-type]
        await r.takeoff()
        here = xy(r)
        came = keepout(here[0] + 20.0, here[1], radius, source="candidate")
        install(r, came)
        await r.run(0.5)
        assert _inside_s(r)[came.keepout_id] == pytest.approx(0.5, abs=0.2)


async def test_unknown_time_survives_a_cylinder_that_replaces_it(tmp_path: Path) -> None:
    """A cylinder that replaces one of unknown time (grown, merged, handed over to the
    ground's) is the same person: unknown too, never inherited as 0. So is one that comes
    around the drone while it is in one of unknown time. Once the caution after the
    restart is over and the drone has been outside every cylinder, one that comes around
    it counts from then."""
    person = keepout(20.0, 0.0, 40.0, source="candidate")
    async with _restarted_in_the_air(tmp_path, person) as r:
        install(r, person)  # the ground sends it again
        await r.run(0.5)
        assert _inside_s(r) == {person.keepout_id: UNKNOWN}
        grown = keepout(22.0, 0.0, 45.0, source="detection")
        install(r, grown)
        await r.run(2.0)
        assert _inside_s(r) == {grown.keepout_id: UNKNOWN}
        path = _resume_path(r)
        assert not leg_enters(path[0], path[1], (22.0, 0.0), 45.0)
        await r.run(KEEPOUT_RESTART_CAUTION_S)  # caution over: inherited all the same
        other = keepout(-10.0, 5.0, 30.0, source="candidate")
        install(r, grown, other)
        await r.run(0.5)
        assert _inside_s(r) == {grown.keepout_id: UNKNOWN, other.keepout_id: UNKNOWN}
        install(r)  # all lifted: the drone is outside every cylinder
        await r.run(0.5)
        again = keepout(20.0, 0.0, 40.0, source="candidate")
        install(r, again)
        await r.run(1.0)
        inside = _inside_s(r)
        assert set(inside) == {again.keepout_id}
        assert inside[again.keepout_id] == pytest.approx(1.0, abs=0.3)


async def test_for_a_minute_after_a_restart_a_new_cylinder_is_unknown_time(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Owner decision 2026-11-04: for KEEPOUT_RESTART_CAUTION_S after a restart in the
    air every new unconfirmed cylinder counts as unknown time, the drone's own fresh
    detections and the ground's first install too: it may only fly out of it. The window
    runs on the monotonic clock: a jump of the wall clock neither ends nor stretches it."""
    async with _restarted_in_the_air(tmp_path) as r:
        started = r.ex._restarted_at  # pyright: ignore[reportPrivateUsage]
        assert started is not None and started <= r.clock.monotonic()
        # The wall clock jumps an hour: still within the minute.
        monkeypatch.setattr(r.clock, "_utc0", r.clock.utc_now() + dt.timedelta(hours=1))
        r.ex.on_detections([detection(5.0, 0.0, KEEPOUT_SCORE_SURE)])  # its own, at once
        ((own, _),) = r.ex.local_keepouts.values()
        assert math.dist(xy(r), PLANE.xy(own.lat, own.lon)) < own.radius_m
        await r.run(0.5)
        assert _inside_s(r) == {own.keepout_id: UNKNOWN}
        path = _resume_path(r)
        assert not leg_enters(path[0], path[1], PLANE.xy(own.lat, own.lon), own.radius_m)
        ground = keepout(-5.0, 0.0, 40.0, source="candidate")  # the ground's first install
        install(r, ground)
        await r.run(0.5)
        assert _inside_s(r)[ground.keepout_id] == UNKNOWN
    async with _restarted_in_the_air(tmp_path) as r:
        started = r.ex._restarted_at  # pyright: ignore[reportPrivateUsage]
        assert started is not None
        await r.run(KEEPOUT_RESTART_CAUTION_S + 1.0 - (r.clock.monotonic() - started))
        r.ex.on_detections([detection(5.0, 0.0, KEEPOUT_SCORE_SURE)])  # at 61 s: as usual
        ((own, _),) = r.ex.local_keepouts.values()
        await r.run(0.5)
        assert _inside_s(r)[own.keepout_id] == pytest.approx(0.5, abs=0.2)


async def test_a_short_break_outside_a_cylinder_is_the_same_pass() -> None:
    """Owner decision 2026-11-04: out of an unconfirmed cylinder for less than
    KEEPOUT_PASS_GAP_S and back in, the pass goes on: its time counts from when the drone
    first came in (and is there while it is out). Out for longer: a new pass from 0."""
    async with rig() as r:
        await r.install(_sim_charter())  # type: ignore[arg-type]
        await r.takeoff()
        here = xy(r)
        person = keepout(here[0] + 20.0, here[1], 40.0, source="candidate")
        away = person.model_copy(
            update=dict(zip(("lat", "lon"), at(here[0] + 200.0, 0.0), strict=True))
        )
        install(r, person)
        await r.run(3.0)
        install(r, away)  # the ground moves it (same id): the drone is outside it now
        await r.run(KEEPOUT_PASS_GAP_S - 2.0)
        assert _inside_s(r)[person.keepout_id] == pytest.approx(KEEPOUT_PASS_GAP_S + 1.0, abs=0.3)
        install(r, person)  # back in: the same pass, its time not started again
        await r.run(1.0)
        assert _inside_s(r)[person.keepout_id] == pytest.approx(KEEPOUT_PASS_GAP_S + 2.0, abs=0.3)
        install(r, away)
        await r.run(KEEPOUT_PASS_GAP_S + 0.5)
        assert _inside_s(r) == {}  # the pass is over
        install(r, person)
        await r.run(1.0)
        assert _inside_s(r)[person.keepout_id] == pytest.approx(1.0, abs=0.3)


async def test_a_cylinder_that_replaces_one_the_drone_just_left_takes_its_pass_over() -> None:
    """A pass the drone left less than KEEPOUT_PASS_GAP_S ago goes on in a cylinder that
    replaces that one under a new id (grown, merged, handed over: the same person). Back
    in within the gap, the pass goes on, no fresh budget. A new cylinder that overlaps no
    cylinder the drone was on a pass through starts nothing."""
    async with rig() as r:
        await r.install(_sim_charter())  # type: ignore[arg-type]
        await r.takeoff()
        here = xy(r)
        person = keepout(here[0] + 20.0, here[1], 40.0, source="candidate")
        away_at = (here[0] + 200.0, here[1])
        away = person.model_copy(update=dict(zip(("lat", "lon"), at(*away_at), strict=True)))
        install(r, person)
        await r.run(3.0)
        install(r, away)  # the drone is outside it now
        await r.run(2.0)
        grown = keepout(away_at[0] + 2.0, away_at[1], 45.0, source="detection")  # a new id
        far = keepout(here[0] - 300.0, here[1], 40.0, source="candidate")
        install(r, grown, far)
        await r.run(1.0)
        inside = _inside_s(r)
        assert set(inside) == {grown.keepout_id}  # nothing for one it was never in
        assert inside[grown.keepout_id] == pytest.approx(6.0, abs=0.3)
        back = grown.model_copy(
            update=dict(zip(("lat", "lon"), at(here[0] + 20.0, here[1]), strict=True))
        )
        install(r, back, far)  # back in within the gap: the same pass goes on
        await r.run(1.0)
        assert _inside_s(r)[grown.keepout_id] == pytest.approx(7.0, abs=0.3)


async def test_after_a_restart_a_cylinder_the_drone_flies_into_is_unknown_time_too(
    tmp_path: Path,
) -> None:
    """On purpose (KEEPOUT_RESTART_CAUTION_S): for the minute after a restart in the air
    a cylinder the drone comes into from outside is unknown time too, though that time is
    measured: the drone may have been in the same cylinder before the restart and left it
    less than the gap ago. After the minute it counts from when the drone came in."""
    async with _restarted_in_the_air(tmp_path) as r:
        here = xy(r)
        person = keepout(here[0] + 200.0, here[1], 40.0, source="candidate")
        install(r, person)
        await r.run(1.0)
        assert _inside_s(r) == {}  # in force, the drone outside it
        over = person.model_copy(
            update=dict(zip(("lat", "lon"), at(here[0] + 20.0, here[1]), strict=True))
        )
        install(r, over)  # the drone is in it now
        await r.run(0.5)
        assert _inside_s(r) == {person.keepout_id: UNKNOWN}
    async with _restarted_in_the_air(tmp_path) as r:
        started = r.ex._restarted_at  # pyright: ignore[reportPrivateUsage]
        assert started is not None
        here = xy(r)
        person = keepout(here[0] + 200.0, here[1], 40.0, source="candidate")
        install(r, person)
        await r.run(KEEPOUT_RESTART_CAUTION_S + 1.0 - (r.clock.monotonic() - started))
        over = person.model_copy(
            update=dict(zip(("lat", "lon"), at(here[0] + 20.0, here[1]), strict=True))
        )
        install(r, over)
        await r.run(0.5)
        assert _inside_s(r)[person.keepout_id] == pytest.approx(0.5, abs=0.2)


def _put(k: Keepout, east: float, north: float) -> Keepout:
    """``k`` (the same id) moved so its middle is at ``(east, north)``."""
    return k.model_copy(update=dict(zip(("lat", "lon"), at(east, north), strict=True)))


async def test_a_cylinder_that_replaces_one_around_the_drone_never_counts_less_than_a_new_one() -> (
    None
):
    """The drone has been in A longer than in B (it stepped out of A for a moment and came
    into B then). B is replaced by an overlapping C around the drone: C counts at least as
    long as a new cylinder would (the earlier of B's pass and A's), never only B's time."""
    async with rig() as r:
        await r.install(_sim_charter())  # type: ignore[arg-type]
        await r.takeoff()
        e, n = xy(r)
        a = keepout(e + 20.0, n, 40.0, source="candidate")
        b = keepout(e - 20.0, n, 40.0, source="candidate")
        install(r, a)
        await r.run(3.0)
        install(r, _put(a, e + 200.0, n))  # out of A for a moment
        await r.run(0.5)
        install(r, _put(a, e + 200.0, n), b)  # B comes around the drone then
        await r.run(0.5)
        install(r, a, b)  # back in A within the gap
        await r.run(1.0)
        in_a = _inside_s(r)[a.keepout_id]
        assert in_a == pytest.approx(5.0, abs=0.3)
        c = keepout(e - 22.0, n, 45.0, source="detection")  # replaces B under a new id
        install(r, a, c)
        await r.run(0.5)
        inside = _inside_s(r)
        assert inside[c.keepout_id] >= inside[a.keepout_id] - 0.2


async def test_a_cylinder_that_replaces_one_around_the_drone_in_one_of_unknown_time_is_unknown(
    tmp_path: Path,
) -> None:
    """The same with A's time unknown (the drone was in it at a restart in the air): B,
    come in during a moment out of A after the caution, is replaced by C around the drone.
    C is unknown time (the whole budget used), as a new cylinder would be, never B's
    measured time."""
    a = keepout(20.0, 0.0, 40.0, source="candidate")
    async with _restarted_in_the_air(tmp_path, a) as r:
        install(r, a)
        started = r.ex._restarted_at  # pyright: ignore[reportPrivateUsage]
        assert started is not None
        await r.run(KEEPOUT_RESTART_CAUTION_S + 1.0 - (r.clock.monotonic() - started))
        assert _inside_s(r) == {a.keepout_id: UNKNOWN}
        e, n = xy(r)
        b = keepout(e - 20.0, n, 40.0, source="candidate")
        install(r, _put(a, e + 200.0, n))  # out of A for a moment
        await r.run(0.5)
        install(r, _put(a, e + 200.0, n), b)
        await r.run(0.5)
        install(r, a, b)
        await r.run(1.0)
        c = keepout(e - 22.0, n, 45.0, source="detection")
        install(r, a, c)
        await r.run(0.5)
        assert _inside_s(r) == {a.keepout_id: UNKNOWN, c.keepout_id: UNKNOWN}


async def test_a_cylinder_that_comes_during_a_moment_out_of_another_takes_that_pass() -> None:
    """Out of A for less than the gap is still A's pass (owner decision 2026-11-04): a new
    cylinder that comes around the drone then counts from when the drone came into A, not
    from 0."""
    async with rig() as r:
        await r.install(_sim_charter())  # type: ignore[arg-type]
        await r.takeoff()
        e, n = xy(r)
        a = keepout(e + 20.0, n, 40.0, source="candidate")
        install(r, a)
        await r.run(3.0)
        install(r, _put(a, e + 200.0, n))  # out of A for a moment
        await r.run(0.5)
        c = keepout(e - 20.0, n, 40.0, source="candidate")
        install(r, _put(a, e + 200.0, n), c)
        await r.run(0.5)
        assert _inside_s(r)[c.keepout_id] == pytest.approx(4.0, abs=0.3)


async def _times_along(seed: int) -> list[dict[int, float]]:
    """The time inside (s) of each cylinder the drone is in, by slot, at each step of a
    random track of installs (``seed``): cylinders in two slots come around the drone,
    move off it for a moment or for longer, are lifted, and are replaced by overlapping
    ones under new ids."""
    rnd = random.Random(seed)  # noqa: S311  # a test track, not a secret
    out: list[dict[int, float]] = []
    async with rig() as r:
        await r.install(_sim_charter())  # type: ignore[arg-type]
        await r.takeoff()
        e, n = xy(r)
        spots = [(e + 20.0, n), (e - 20.0, n)]
        slots: dict[int, Keepout] = {}
        where: dict[int, bool] = {}  # around the drone or not
        for _ in range(40):
            for i in range(len(spots)):  # each slot on its own: often none around the drone
                act = rnd.choice(("in", "out", "out", "replace", "lift"))
                if i not in slots:
                    if act == "in":
                        slots[i] = keepout(*spots[i], 40.0, source="candidate")
                        where[i] = True
                elif act in ("in", "out"):
                    where[i] = act == "in"
                elif act == "replace":
                    slots[i] = keepout(*spots[i], 42.0, source="detection")
                else:
                    del slots[i], where[i]
            placed = [
                k if where[j] else _put(k, spots[j][0] + 300.0, spots[j][1])
                for j, k in slots.items()
            ]
            install(r, *placed)
            await r.run(rnd.choice((0.3, 0.6, 1.5, 11.0)))
            inside = _inside_s(r)
            ids = {k.keepout_id: j for j, k in slots.items() if where[j]}
            out.append({ids[kid]: s for kid, s in inside.items() if kid in ids})
    return out


@pytest.mark.parametrize("seed", range(17))  # 11 and 16 hit the loosening before the fix
async def test_a_handover_never_gives_a_cylinder_around_the_drone_more_time(
    seed: int, monkeypatch: pytest.MonkeyPatch
) -> None:
    """For the same track of installs, every cylinder the drone is in has used at least as
    much of its budget with the handover of passes as without it (a replacing cylinder
    taking no pass over): the handover may only make the drone leave sooner, never later."""
    with_it = await _times_along(seed)
    monkeypatch.setattr(executor_module, "_handed_over", lambda *_: None)
    without = await _times_along(seed)
    for step, (a, b) in enumerate(zip(with_it, without, strict=True)):
        for slot, s in b.items():
            assert a[slot] >= s - 0.2, (step, slot, a, b)
