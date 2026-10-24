"""Keep-out cylinders on ArduCopter SITL: the onboard reflex and ``orbit``.

A confirmed find is written as the search service writes it and reaches the onboard
executor as the ground's ``KeepoutInstall`` (``Scenario.confirm_find``).

* ``test_a_new_cylinder_reroutes_the_search``: a find is confirmed ahead of the drone
  while it flies a survey line. The rest of the survey is planned around the cylinder,
  no track sample lies inside it, and the survey ends ``done``.
* ``test_a_new_cylinder_holds_a_goto_and_stop_still_works``: a find is confirmed on a
  ``goto``'s way. The drone hovers short of the cylinder (``KEEPOUT_VIOLATION``); STOP
  then brakes and holds as always, and FLY HOME brings it home.
* ``test_own_detections_under_the_drone_during_a_search``: two of the drone's own
  person detections land either side of it mid-survey (two stills of one person). The
  drone leaves both cylinders without going deeper into either, never comes back in, and
  the survey ends ``done``.
* ``test_orbit_around_a_find``: an orbit around a confirmed find flies a ring outside
  its cylinder and hovers when done; STOP during a second orbit brakes and holds.
* ``test_a_person_seen_during_the_find_move``: the radio is cut near a confirmed find
  and the drone sees a person while it flies the link-loss find move. Ahead on its
  way: it hovers short of the new cylinder. Right below it (the drone inside the new
  cylinder, its leg heading deeper): it first leaves the cylinder straight out, then
  hovers outside it. Either way the find hover holds there, never inside a cylinder.
* ``test_an_inspection_started_inside_new_cylinders``: mid-survey the drone sees a person
  right below (two of its own cylinders around it) and the ground holds a candidate's
  cylinder there; the pilot asks for a closer look at once. The drone leaves the
  cylinders on a straight way out, stops, and flies to the point worked out from there;
  meanwhile the ground moves the candidate's cylinder (or shrinks it) under the same id.
  After the drone is out of a cylinder's shape it never comes back into it (more than a
  metre: the track's noise), and the inspection ends done or in a keep-out hover.
"""

from __future__ import annotations

import math
import sys
import time
from pathlib import Path
from typing import Any

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent))

from scenario_rig import (  # pyright: ignore[reportMissingImports]
    FIXTURE,
    Scenario,
    fresh_db,
    migrated_db,
    passwords,
    scenario,
    sim,
    survey_box_enu,
)
from test_sc_emergency import stopped_and_holding  # pyright: ignore[reportMissingImports]
from test_sc_linkloss import (  # pyright: ignore[reportMissingImports]
    back_and_land,
    find_ahead,
    north_m,
    onboard_events,
    survey_box,
    surveying,
    until_event,
)
from test_sk_inspect import (  # pyright: ignore[reportMissingImports]
    Camera,
    a_detection,
    airborne_for_inspection,
    inspecting,
    person,
)

from sd_onboard.executor.link_loss import INTERNAL_FIND_GOTO
from sd_safety.constants import FIND_MOVE_DEADLINE_S, LINK_LOSS_DETECT_S, PERSON_KEEPOUT_M
from sd_safety.targets import Plane, distance_m
from sd_schemas.base import new_id
from sd_schemas.keepout import Keepout
from sd_schemas.link import KeepoutInstall

pytestmark = [pytest.mark.sitl, pytest.mark.slow]
__all__ = ["fresh_db", "migrated_db", "passwords", "sim"]

SIGMA_M = 3.0
RADIUS_M = PERSON_KEEPOUT_M + SIGMA_M
CLEAR_M = 15.0  # how far outside the cylinder the drone is when the find comes
ORBIT_M = 50.0  # the ring's radius: more than the cylinder plus the orbit's 10 m
RING_SLACK_M = 10.0  # the flown ring's distance from its waypoint polygon and overshoot
SKILLS = ["orbit"]
FIND_LOITER_BY_S = LINK_LOSS_DETECT_S + FIND_MOVE_DEADLINE_S + 0.5  # LOITER by then
FIND_RADIUS_M = PERSON_KEEPOUT_M + 5.0  # ``find_ahead``'s find (its sigma is 5 m)
NOISE_M = 1.0  # the flight controller's wobble and the track's sampling
# Far above the ground's own installs in this test: a later one of those is stale.
INSTALL_SEQ = 1_000_000


def _inside(sc: Scenario, since: float, centre: tuple[float, float]) -> list[Any]:
    return [s for s in sc.track if s.t >= since and distance_m(s.lat, s.lon, *centre) < RADIUS_M]


async def _land(sc: Scenario) -> None:
    assert (await sc.press("return_home")).status_code == 202
    await sc.until(lambda: not sc.tel.armed, 240, "landed")
    assert sc.from_home_m() <= 3.0
    await sc.check_invariants()


async def _executor(sc: Scenario, state: str) -> bool:
    return (await sc.onboard_state())["executor_state"] == state


@pytest.mark.budget_s(480)
async def test_a_new_cylinder_reroutes_the_search(
    sim: Any, migrated_db: Any, tmp_path: Path
) -> None:
    async with scenario(sim, migrated_db, tmp_path) as sc:
        await sc.airborne()
        res = await sc.operator_order("survey_sector", {"polygon": survey_box(sc)})
        assert res["gate"]["decision"] == "ALLOW", res
        survey = await sc.wait_command("survey_sector", "running", 60)
        await sc.until(lambda: _executor(sc, "SURVEY"), 120, "in the box")
        await surveying(sc)

        # A person in the middle of the box: the lines through it lie ahead.
        corners = survey_box_enu()
        centre = sc.at(sum(e for e, _ in corners) / 4, sum(n for _, n in corners) / 4)
        assert distance_m(sc.tel.lat, sc.tel.lon, *centre) >= RADIUS_M + CLEAR_M
        since = time.monotonic()
        await sc.confirm_find(*centre, sigma95_m=SIGMA_M)
        await until_event(sc, "keepout_reroute")

        await sc.wait_command("survey_sector", "done", 300)
        after = await sc.command(survey["id"])
        assert after["status"] == "done", after
        inside = _inside(sc, since, centre)
        assert not inside, inside[:3]
        await _land(sc)


@pytest.mark.budget_s(480)
async def test_own_detections_under_the_drone_during_a_search(
    sim: Any, migrated_db: Any, tmp_path: Path
) -> None:
    async with scenario(sim, migrated_db, tmp_path) as sc:
        await sc.airborne()
        res = await sc.operator_order("survey_sector", {"polygon": survey_box(sc)})
        assert res["gate"]["decision"] == "ALLOW", res
        survey = await sc.wait_command("survey_sector", "running", 60)
        await sc.until(lambda: _executor(sc, "SURVEY"), 120, "in the box")
        await surveying(sc)

        # Two people right below, one just ahead along the line and one behind: the leg heads
        # deeper into the first, and they lie far enough apart not to merge into one cylinder.
        plane = Plane(*sc.r.home)
        now = sc.track[-1]
        before = next(s for s in reversed(sc.track) if s.t <= now.t - 1.0)
        (e, n), (e0, n0) = plane.xy(now.lat, now.lon), plane.xy(before.lat, before.lon)
        step = math.dist((e, n), (e0, n0))
        de, dn = (e - e0) / step, (n - n0) / step
        centres = [
            plane.latlon(e + 6.0 * de, n + 6.0 * dn),
            plane.latlon(e - 13.0 * de, n - 13.0 * dn),
        ]
        since = time.monotonic()
        sc.r.onboard.ex.on_detections([person(c) for c in centres])
        await until_event(sc, "keepout_reroute")

        await sc.wait_command("survey_sector", "done", 300)
        after = await sc.command(survey["id"])
        assert after["status"] == "done", after
        assert not await onboard_events(sc, "keepout_reroute_refused")
        assert not await onboard_events(sc, "keepout_hold")
        # Out of both and never back in.
        track = [s for s in sc.track if s.t >= since]
        inside = [any(distance_m(s.lat, s.lon, *c) < RADIUS_M for c in centres) for s in track]
        assert not inside[-1]
        first_out = inside.index(False)
        assert not any(inside[first_out:]), "back into a cylinder"
        await _land(sc)


@pytest.mark.budget_s(420)
async def test_a_new_cylinder_holds_a_goto_and_stop_still_works(
    sim: Any, migrated_db: Any, tmp_path: Path
) -> None:
    async with scenario(sim, migrated_db, tmp_path) as sc:
        await sc.airborne()
        p1 = next(p for p in FIXTURE["pins"] if p["id"] == "P1")
        target = sc.at(float(p1["east_m"]), float(p1["north_m"]))
        res = await sc.operator_order("goto", {"lat": target[0], "lon": target[1], "alt_m": 40})
        assert res["gate"]["decision"] == "ALLOW", res
        goto = await sc.wait_command("goto", "running", 30)
        await sc.until(lambda: sc.from_home_m() >= 30.0, 60, "on the way")

        # A person on the way, well ahead of the drone (the goto's end lies in the cylinder).
        ahead = sc.at(0.0, 170.0)
        since = time.monotonic()
        await sc.confirm_find(*ahead, sigma95_m=SIGMA_M)
        await until_event(sc, "keepout_hold", 60.0)
        await sc.until(lambda: _executor(sc, "HOLD"), 10, "the drone hovers")
        live = await sc.onboard_state()
        assert live["hold_reason"] == "limit", live
        await sc.until(lambda: sc.tel.groundspeed_ms < 0.5, 15, "still")
        after = await sc.command(goto["id"])
        assert after["status"] in ("superseded", "failed"), after
        inside = _inside(sc, since, ahead)
        assert not inside, inside[:3]

        t0 = time.monotonic()
        assert (await sc.press("stop")).status_code == 202
        await stopped_and_holding(sc, t0)
        assert (await sc.press("resume")).status_code == 202
        await _land(sc)


@pytest.mark.budget_s(480)
async def test_orbit_around_a_find(sim: Any, migrated_db: Any, tmp_path: Path) -> None:
    async with scenario(sim, migrated_db, tmp_path) as sc:
        charter = FIXTURE["charters"]["charter_test"]
        await sc.airborne(allowed_skills=[*charter["allowed_skills"], *SKILLS])
        centre = sc.at(0.0, 100.0)
        await sc.confirm_find(*centre, sigma95_m=SIGMA_M)
        ring = {"lat": centre[0], "lon": centre[1], "radius_m": ORBIT_M, "alt_m": 40}

        since = time.monotonic()
        res = await sc.operator_order("orbit", ring)
        assert res["gate"]["decision"] == "ALLOW", res
        await sc.wait_command("orbit", "running", 30)
        await sc.until(
            lambda: abs(distance_m(sc.tel.lat, sc.tel.lon, *centre) - ORBIT_M) <= RING_SLACK_M,
            90,
            "on the ring",
        )
        on_ring = time.monotonic()
        await sc.wait_command("orbit", "done", 240)
        await sc.until(lambda: _executor(sc, "HOLD"), 5, "hovers after the orbit")
        assert not _inside(sc, since, centre)
        flown = [
            distance_m(s.lat, s.lon, *centre)
            for s in sc.track
            if s.t >= on_ring and s.mode == "AUTO"
        ]
        assert flown, "no ring flown"
        assert all(ORBIT_M - RING_SLACK_M <= d <= ORBIT_M + RING_SLACK_M for d in flown), (
            min(flown),
            max(flown),
        )

        # STOP during an orbit.
        res = await sc.operator_order("orbit", ring)
        assert res["gate"]["decision"] == "ALLOW", res
        await sc.wait_command("orbit", "running", 30)
        await sc.until(lambda: sc.tel.groundspeed_ms > 2.0, 30, "orbiting")
        t0 = time.monotonic()
        assert (await sc.press("stop")).status_code == 202
        await stopped_and_holding(sc, t0)
        assert (await sc.press("resume")).status_code == 202
        await _land(sc)


@pytest.mark.parametrize("seen", ["ahead", "below"])
@pytest.mark.budget_s(420)
async def test_a_person_seen_during_the_find_move(
    sim: Any, migrated_db: Any, tmp_path: Path, seen: str
) -> None:
    async with scenario(sim, migrated_db, tmp_path, proxied_link=True) as sc:
        await sc.airborne()
        ex = sc.r.onboard.ex
        flat, flon, _, _ = await find_ahead(sc)
        sc.radio(mavlink=True, sdlink=False)
        down = time.monotonic()
        try:
            await sc.until(
                lambda: ex.active is not None and ex.active.skill == INTERNAL_FIND_GOTO,
                LINK_LOSS_DETECT_S + 3.0,
                "the find move",
            )
            plane = Plane(*sc.r.home)
            st = sc.sim.harness.state
            e, n = plane.xy(st.lat, st.lon)
            # Ahead: on the way to the hover point, a little past it. Below: ahead of the
            # drone, which is 8 m inside the new cylinder with its leg heading deeper. A
            # short way out, so it ends well before the find move's deadline.
            centre = plane.latlon(e, n + (RADIUS_M - 8.0 if seen == "below" else 45.0))
            since = time.monotonic()
            ex.on_detections([person(centre)])
            ((k, _),) = ex.local_keepouts.values()
            await sc.until(
                lambda: (
                    sc.fc_mode() == "LOITER"
                    and str(ex.state) == "HOLD"
                    and ex.hold_reason == "find_hold"
                ),
                FIND_LOITER_BY_S,
                "find hover",
            )
            assert time.monotonic() - down <= FIND_LOITER_BY_S
            await sc.until(lambda: st.ground_speed_ms < 0.5, 15, "still")
            await sc.hold_for(10.0, lambda: sc.fc_mode() == "LOITER", "the find hover holds")
            track = [s for s in sc.track if s.t >= since and s.armed]
            dists = [distance_m(s.lat, s.lon, k.lat, k.lon) for s in track]
            if seen == "ahead":
                assert min(dists) >= k.radius_m, min(dists)
            else:
                assert dists[0] < k.radius_m  # it came with the drone inside
                out = next(i for i, d in enumerate(dists) if d >= k.radius_m)
                assert all(d >= k.radius_m for d in dists[out:]), "back into the cylinder"
                # Out, not deeper: no closer than braking from the find move's speed.
                assert min(dists) >= dists[0] - 8.0, (dists[0], min(dists))
            assert all(distance_m(s.lat, s.lon, flat, flon) >= FIND_RADIUS_M for s in track)
            assert north_m(sc) < 150.0
        finally:
            await back_and_land(sc)
        kinds = {"ahead": "find_move_keepout", "below": "find_move_keepout_exit"}
        await until_event(sc, kinds[seen])
        await _land(sc)


def _shapes_entered(
    shapes: list[tuple[float, float, float, float, float]], track: list[Any]
) -> list[str]:
    """(lat, lon, radius, since, until) per shape of a cylinder: the samples that came
    back more than ``NOISE_M`` into one after being out of it."""
    broken: list[str] = []
    for lat, lon, radius, since, until in shapes:
        out = False
        for s in track:
            if not since <= s.t <= until:
                continue
            d = distance_m(lat, lon, s.lat, s.lon) - radius
            out = out or d >= 0.0
            if out and d < -NOISE_M:
                broken.append(f"{-d:.1f} m in, {s.t - since:.0f} s after the shape came")
                break
    return broken


@pytest.mark.parametrize("change", ["moves", "shrinks"])
@pytest.mark.budget_s(480)
async def test_an_inspection_started_inside_new_cylinders(
    sim: Any, migrated_db: Any, tmp_path: Path, change: str
) -> None:
    async with scenario(sim, migrated_db, tmp_path) as sc:
        await airborne_for_inspection(sc)
        res = await sc.operator_order("survey_sector", {"polygon": survey_box(sc)})
        assert res["gate"]["decision"] == "ALLOW", res
        await sc.wait_command("survey_sector", "running", 60)
        await sc.until(lambda: _executor(sc, "SURVEY"), 120, "in the box")
        await surveying(sc)
        ex = sc.r.onboard.ex

        # A person just ahead, right below: two of the drone's own cylinders around it
        # (two stills of one person) and the candidate's cylinder from the ground.
        plane = Plane(*sc.r.home)
        now = sc.track[-1]
        before = next(s for s in reversed(sc.track) if s.t <= now.t - 1.0)
        (e, n), (e0, n0) = plane.xy(now.lat, now.lon), plane.xy(before.lat, before.lon)
        step = math.dist((e, n), (e0, n0))
        de, dn = (e - e0) / step, (n - n0) / step
        target = plane.latlon(e + 6.0 * de, n + 6.0 * dn)
        det = await a_detection(sc, target)
        ex.on_detections([person(target), person(plane.latlon(e - 13.0 * de, n - 13.0 * dn))])
        cand = Keepout(
            keepout_id=new_id(),
            lat=target[0],
            lon=target[1],
            radius_m=RADIUS_M,
            source="candidate",
            source_id=new_id(),
        )
        ex.on_keepouts(KeepoutInstall(install_seq=INSTALL_SEQ, keepouts=[cand]), new_id())
        Camera(sc)
        since = time.monotonic()
        order = await inspecting(sc, det)

        # Once the drone is out of the cylinders, the ground moves (or shrinks) the
        # candidate's cylinder under the same id.
        def out() -> bool:
            st = sc.sim.harness.state
            return all(
                distance_m(k.lat, k.lon, st.lat, st.lon) >= k.radius_m
                for k in ex.keepouts_in_force()
            )

        await sc.until(out, 60, "out of the cylinders")
        st = sc.sim.harness.state
        if change == "moves":  # 12 m towards where the drone is
            ce, cn = plane.xy(cand.lat, cand.lon)
            pe, pn = plane.xy(st.lat, st.lon)
            k = 12.0 / math.dist((ce, cn), (pe, pn))
            lat, lon = plane.latlon(ce + k * (pe - ce), cn + k * (pn - cn))
            later = cand.model_copy(update={"lat": lat, "lon": lon})
        else:
            later = cand.model_copy(update={"radius_m": PERSON_KEEPOUT_M - 10.0})
        changed = time.monotonic()
        ex.on_keepouts(KeepoutInstall(install_seq=INSTALL_SEQ + 1, keepouts=[later]), new_id())

        await sc.until(
            lambda: _status(sc, order["id"], "done", "superseded", "failed"), 240, "ended"
        )
        after = await sc.command(order["id"])
        assert after["status"] in ("done", "superseded"), after
        if after["status"] == "superseded":  # the keep-out hover (KEEPOUT_VIOLATION)
            assert await onboard_events(sc, "keepout_hold"), after
        own = [(k.lat, k.lon, k.radius_m, since, math.inf) for k, _ in ex.local_keepouts.values()]
        shapes = [
            *own,
            (cand.lat, cand.lon, cand.radius_m, since, changed),
            (later.lat, later.lon, later.radius_m, changed, math.inf),
        ]
        track = [s for s in sc.track if s.t >= since and s.armed]
        broken = _shapes_entered(shapes, track)
        assert not broken, broken
        await _land(sc)


async def _status(sc: Scenario, command_id: str, *want: str) -> bool:
    return (await sc.command(command_id))["status"] in want
