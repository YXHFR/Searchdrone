"""Owner decisions 2026-11-03 on the cylinders of unconfirmed persons, onboard.

A drone's own cylinder needs a second look (two stills that agree on the place) unless one
detection is very sure; a cylinder the operator rejected goes; a search in sim or an
exercise (overflight allowed) passes straight through an unconfirmed cylinder, in live
without the permit it goes around.
"""

from __future__ import annotations

import math
from types import SimpleNamespace

import pytest
from onboard_helpers import PLANE, Rig, charter, rig, square
from test_inspect import SKILLS, detection, install, keepout, xy

from sd_mav.mission import MISSION
from sd_onboard.skills.survey_sector import SurveyRun
from sd_safety.constants import (
    GOTO_ALT_STEP_TOL_M,
    KEEPOUT_PAIR_MIN_M,
    KEEPOUT_PAIR_WINDOW_S,
    KEEPOUT_PASS_MAX_S,
    KEEPOUT_SCORE_SURE,
)
from sd_schemas.base import new_id
from sd_schemas.executor import ExecutorState as S
from sd_schemas.link import KeepoutInstall

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
