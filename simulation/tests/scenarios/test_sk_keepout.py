"""Keep-out cylinders on ArduCopter SITL: the onboard reflex and ``orbit``.

A confirmed find is written as the search service writes it and reaches the onboard
executor as the ground's ``KeepoutInstall`` (``Scenario.confirm_find``).

* ``test_a_new_cylinder_reroutes_the_search``: a find is confirmed ahead of the drone
  while it flies a survey line. The rest of the survey is planned around the cylinder,
  no track sample lies inside it, and the survey ends ``done``.
* ``test_a_new_cylinder_holds_a_goto_and_stop_still_works``: a find is confirmed on a
  ``goto``'s way. The drone hovers short of the cylinder (``KEEPOUT_VIOLATION``); STOP
  then brakes and holds as always, and FLY HOME brings it home.
* ``test_orbit_around_a_find``: an orbit around a confirmed find flies a ring outside
  its cylinder and hovers when done; STOP during a second orbit brakes and holds.
"""

from __future__ import annotations

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
    survey_box,
    surveying,
    until_event,
)

from sd_safety.constants import PERSON_KEEPOUT_M
from sd_safety.targets import distance_m

pytestmark = [pytest.mark.sitl, pytest.mark.slow]
__all__ = ["fresh_db", "migrated_db", "passwords", "sim"]

SIGMA_M = 3.0
RADIUS_M = PERSON_KEEPOUT_M + SIGMA_M
CLEAR_M = 15.0  # how far outside the cylinder the drone is when the find comes
ORBIT_M = 50.0  # the ring's radius: more than the cylinder plus the orbit's 10 m
RING_SLACK_M = 10.0  # the flown ring's distance from its waypoint polygon and overshoot
SKILLS = ["orbit"]


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
        assert await _executor(sc, "HOLD")
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
