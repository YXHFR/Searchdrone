"""Keep-out cylinders on ArduCopter SITL: the onboard reflex and ``orbit``.

A confirmed find is written as the search service writes it and reaches the onboard
executor as the ground's ``KeepoutInstall`` (``Scenario.confirm_find``).

* ``test_a_new_cylinder_reroutes_the_search``: a find is confirmed ahead of the drone
  while it flies a survey line. The rest of the survey is planned around the cylinder,
  no track sample lies inside it, and the survey ends ``done``.
* ``test_overlapping_cylinders_are_passed_as_one_group``: two finds are confirmed close
  together in the middle of the box, so their cylinders overlap. The rest of the survey
  goes around both as one group: no hold, no refused reroute, no track sample inside
  either, and the survey ends ``done``.
* ``test_a_way_around_keeps_the_margin_from_a_cylinder_beside_it``: the case #96 fixed.
  On a survey's only line a find is confirmed beside the line, and then one on it whose
  short way around would pass inside the first one's margin without entering it. The way
  around keeps the margin from both: the track never comes closer than the margin to the
  one beside the line (less the track's noise), never inside either, and the survey ends
  ``done`` with no hold and no refused reroute.
* ``test_a_person_seen_close_ahead_stops_the_search_first``: the drone sees a person
  ahead on a long survey line, closer than it can turn away from at its speed but farther
  than braking takes it. It stops in LOITER first, plans the rest again from where it
  stopped, never comes inside the cylinder, and the survey ends ``done`` (or in the
  keep-out hover, when no way around is left from there).
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
* ``test_a_person_below_during_an_inspection_is_left_on_a_replan``: on the leg to an
  inspection point the drone sees a person right below, the leg heading deeper. The
  reflex plans the leg again (``executor.keepout_replan``): out of the cylinder, never
  deeper than braking takes it and never back in; the order ends done, failed (no safe
  way to the point) or in a keep-out hover.
"""

from __future__ import annotations

import itertools
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

from sd_onboard.executor.keepout_reflex import stop_m
from sd_onboard.executor.link_loss import INTERNAL_FIND_GOTO
from sd_safety.constants import FIND_MOVE_DEADLINE_S, LINK_LOSS_DETECT_S, PERSON_KEEPOUT_M
from sd_safety.keepout import KEEPOUT_ARC_MARGIN_M, reroute
from sd_safety.targets import Plane, distance_m, point_segment_distance, polygon_from_geojson
from sd_schemas.base import new_id
from sd_schemas.keepout import INSPECT_SPEED_MS, Keepout
from sd_schemas.link import KeepoutInstall
from sd_sim.search_invariants import stop_distance_m

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
# A survey of a box narrower than two lines' spacing has one line: east-west, this far
# north of home, from this far west to as far east.
LINE_N = 150.0
LINE_E = 130.0


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
async def test_overlapping_cylinders_are_passed_as_one_group(
    sim: Any, migrated_db: Any, tmp_path: Path
) -> None:
    async with scenario(sim, migrated_db, tmp_path) as sc:
        await sc.airborne()
        res = await sc.operator_order("survey_sector", {"polygon": survey_box(sc)})
        assert res["gate"]["decision"] == "ALLOW", res
        survey = await sc.wait_command("survey_sector", "running", 60)
        await sc.until(lambda: _executor(sc, "SURVEY"), 120, "in the box")
        await surveying(sc)

        # Two people 30 m apart in the middle of the box: their cylinders overlap, so a way
        # around one alone would run into the other.
        corners = survey_box_enu()
        ce, cn = sum(e for e, _ in corners) / 4, sum(n for _, n in corners) / 4
        centres = [sc.at(ce - 15.0, cn), sc.at(ce + 15.0, cn)]
        assert distance_m(*centres[0], *centres[1]) < 2 * RADIUS_M
        for c in centres:
            assert distance_m(sc.tel.lat, sc.tel.lon, *c) >= RADIUS_M + CLEAR_M
        since = time.monotonic()
        for c in centres:
            await sc.confirm_find(*c, sigma95_m=SIGMA_M)
        await until_event(sc, "keepout_reroute")

        await sc.wait_command("survey_sector", "done", 300)
        after = await sc.command(survey["id"])
        assert after["status"] == "done", after
        assert not await onboard_events(sc, "keepout_reroute_refused")
        assert not await onboard_events(sc, "keepout_hold")
        for c in centres:
            inside = _inside(sc, since, c)
            assert not inside, inside[:3]
        await _land(sc)


def _one_line(sc: Scenario) -> dict[str, Any]:
    corners = [(-LINE_E, LINE_N - 10.0), (LINE_E, LINE_N - 10.0), (LINE_E, LINE_N + 10.0)]
    ring = [
        [lon, lat] for lat, lon in (sc.at(e, n) for e, n in [*corners, (-LINE_E, LINE_N + 10.0)])
    ]
    return {"type": "Polygon", "coordinates": [[*ring, ring[0]]]}


@pytest.mark.budget_s(480)
async def test_a_way_around_keeps_the_margin_from_a_cylinder_beside_it(
    sim: Any, migrated_db: Any, tmp_path: Path
) -> None:
    async with scenario(sim, migrated_db, tmp_path) as sc:
        await sc.airborne()
        line = {"polygon": _one_line(sc), "heading_deg": 90.0, "spacing_m": 30.0}
        res = await sc.operator_order("survey_sector", line)
        assert res["gate"]["decision"] == "ALLOW", res
        survey = await sc.wait_command("survey_sector", "running", 60)
        ex = sc.r.onboard.ex
        plane, charter = ex.plane(), ex.charter
        assert plane is not None and charter is not None
        st = sc.sim.harness.state

        def at_its_start() -> bool:
            e, n = plane.xy(st.lat, st.lon)
            return abs(n - LINE_N) < 3.0 and abs(e) > 90.0

        await sc.until(at_its_start, 120, "at the start of the line")
        here = plane.xy(st.lat, st.lon)
        way = 1.0 if here[0] < 0.0 else -1.0  # towards the other end
        # A, the person on the line, a little north of it: the short way around is south.
        # B, south of the line, where that way would pass inside B's margin without
        # entering B. B overlaps neither A nor the line.
        a = (40.0 * way, LINE_N + 5.0)
        b = (a[0], a[1] - 73.0)
        area = polygon_from_geojson(charter.area, plane)
        alone = reroute([here, (LINE_E * way, LINE_N)], [(a, RADIUS_M)], area=area)
        assert alone is not None
        past_b = min(point_segment_distance(b, p, q) for p, q in itertools.pairwise(alone))
        assert RADIUS_M <= past_b < RADIUS_M + KEEPOUT_ARC_MARGIN_M - NOISE_M, past_b
        assert math.dist(a, b) > 2 * RADIUS_M and LINE_N - b[1] > RADIUS_M

        since = time.monotonic()
        cb, ca = plane.latlon(*b), plane.latlon(*a)
        await sc.confirm_find(*cb, sigma95_m=SIGMA_M)
        assert distance_m(sc.tel.lat, sc.tel.lon, *ca) >= RADIUS_M + CLEAR_M
        await sc.confirm_find(*ca, sigma95_m=SIGMA_M)
        await until_event(sc, "keepout_reroute")

        await sc.wait_command("survey_sector", "done", 300)
        after = await sc.command(survey["id"])
        assert after["status"] == "done", after
        assert not await onboard_events(sc, "keepout_reroute_refused")
        assert not await onboard_events(sc, "keepout_hold")
        for c in (ca, cb):
            inside = _inside(sc, since, c)
            assert not inside, inside[:3]
        track = [s for s in sc.track if s.t >= since and s.armed]
        nearest = min(distance_m(s.lat, s.lon, *cb) for s in track)
        assert nearest >= RADIUS_M + KEEPOUT_ARC_MARGIN_M - NOISE_M, nearest
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


@pytest.mark.budget_s(480)
async def test_a_person_seen_close_ahead_stops_the_search_first(
    sim: Any, migrated_db: Any, tmp_path: Path
) -> None:
    async with scenario(sim, migrated_db, tmp_path) as sc:
        await sc.airborne()
        res = await sc.operator_order("survey_sector", {"polygon": survey_box(sc)})
        assert res["gate"]["decision"] == "ALLOW", res
        survey = await sc.wait_command("survey_sector", "running", 60)
        await sc.until(lambda: _executor(sc, "SURVEY"), 120, "in the box")
        await surveying(sc)
        ex = sc.r.onboard.ex
        plane = Plane(*sc.r.home)
        line: list[tuple[float, float, float, float, float]] = []

        def on_a_long_line() -> bool:
            """Straight along a survey line at speed, with room on it for the cylinder:
            where the drone is, which way it flies and how fast."""
            now = sc.track[-1]
            before = next((s for s in reversed(sc.track) if s.t <= now.t - 1.0), None)
            if before is None or now.groundspeed_ms < 4.0:
                return False
            (e, n), (e0, n0) = plane.xy(now.lat, now.lon), plane.xy(before.lat, before.lon)
            step = math.dist((e, n), (e0, n0))
            if step < 0.9 * now.groundspeed_ms * (now.t - before.t):
                return False  # still turning or speeding up
            de, dn = (e - e0) / step, (n - n0) / step
            run = getattr(ex.active.handler, "run", None) if ex.active is not None else None
            room = stop_m(now.groundspeed_ms) + 2 * RADIUS_M + 20.0
            for w in [] if run is None else run.waypoints:
                we, wn = plane.xy(w.lat, w.lon)
                d = math.dist((e, n), (we, wn))
                if d > room and (de * (we - e) + dn * (wn - n)) > 0.999 * d:
                    line[:] = [(e, n, de, dn, now.groundspeed_ms)]
                    return True
            return False

        await sc.until(on_a_long_line, 120, "on a long survey line")
        ((e, n, de, dn, speed),) = line
        # The edge between where braking alone stops the drone and the reflex's stop
        # distance: too close to turn away, far enough to stop outside.
        edge = (stop_distance_m(speed) + stop_m(speed)) / 2
        assert stop_distance_m(speed) + 3.0 < edge < stop_m(speed), (speed, edge)
        centre = plane.latlon(e + (edge + RADIUS_M) * de, n + (edge + RADIUS_M) * dn)
        since = time.monotonic()
        ex.on_detections([person(centre)])
        ((k, _),) = ex.local_keepouts.values()

        reroutes = await until_event(sc, "keepout_reroute")
        assert reroutes[-1]["payload"].get("stop_first") is True, reroutes[-1]
        await until_event(sc, "keepout_reroute_stopped", 30.0)
        await sc.until(lambda: _status(sc, survey["id"], "done", "failed"), 300, "the survey ended")
        after = await sc.command(survey["id"])
        if after["status"] == "failed":  # no way around from where it stopped
            assert await onboard_events(sc, "keepout_hold"), after
        else:
            assert after["status"] == "done", after
        track = [s for s in sc.track if s.t >= since and s.armed]
        nearest = min(distance_m(s.lat, s.lon, k.lat, k.lon) for s in track)
        assert nearest >= k.radius_m - NOISE_M, (nearest, k.radius_m)
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
            here_e, here_n = plane.xy(st.lat, st.lon)
            k = 12.0 / math.dist((ce, cn), (here_e, here_n))
            lat, lon = plane.latlon(ce + k * (here_e - ce), cn + k * (here_n - cn))
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


@pytest.mark.budget_s(420)
async def test_a_person_below_during_an_inspection_is_left_on_a_replan(
    sim: Any, migrated_db: Any, tmp_path: Path
) -> None:
    async with scenario(sim, migrated_db, tmp_path) as sc:
        await airborne_for_inspection(sc)
        p1 = next(p for p in FIXTURE["pins"] if p["id"] == "P1")
        det = await a_detection(sc, sc.at(float(p1["east_m"]), float(p1["north_m"])))
        Camera(sc)
        order = await inspecting(sc, det)
        ex = sc.r.onboard.ex
        st = sc.sim.harness.state
        await sc.until(
            lambda: north_m(sc) >= 50.0 and st.ground_speed_ms > 3.0, 90, "on the leg north"
        )

        # A person 15 m ahead, right below the leg: the drone is inside the new cylinder
        # and its leg heads deeper.
        plane = Plane(*sc.r.home)
        e, n = plane.xy(st.lat, st.lon)
        centre = plane.latlon(e, n + 15.0)
        since = time.monotonic()
        ex.on_detections([person(centre)])
        k = min(
            (k for k, _ in ex.local_keepouts.values()),
            key=lambda k: distance_m(k.lat, k.lon, *centre),
        )
        await until_event(sc, "keepout_replan")

        await sc.until(
            lambda: _status(sc, order["id"], "done", "failed", "superseded"), 240, "ended"
        )
        after = await sc.command(order["id"])
        if after["status"] == "superseded":  # the keep-out hover (KEEPOUT_VIOLATION)
            assert await onboard_events(sc, "keepout_hold"), after
        track = [s for s in sc.track if s.t >= since and s.armed]
        dists = [distance_m(s.lat, s.lon, k.lat, k.lon) for s in track]
        assert dists[0] < k.radius_m  # it came with the drone inside
        assert min(dists) >= dists[0] - stop_distance_m(INSPECT_SPEED_MS), (dists[0], min(dists))
        broken = _shapes_entered([(k.lat, k.lon, k.radius_m, since, math.inf)], track)
        assert not broken, broken
        await _land(sc)


async def _status(sc: Scenario, command_id: str, *want: str) -> bool:
    return (await sc.command(command_id))["status"] in want
