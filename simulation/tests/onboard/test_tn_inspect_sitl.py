"""The closer look's scenarios on the onboard rig (spec "Titta närmare" §16.4, §17).

Each scenario's world, map errors and laser are in ``simulation/scenarios/TNxx_*.yaml``
(``sd_sim.inspect_world``) and flies with its three fixed seeds (the person's place, the
candidate's error, the detections' noise and the map's random errors move with the seed).
``inspect_rig.TnLook`` flies the look with the executor's own ``InspectHandler`` against
ArduCopter SITL (builtin physics) and checks acceptance 7 (clearance 5 m, at most 1.5 m
under the floor in force) and 20 (every step a still per camera, with the order's id and
its step) in every run, and writes the §16.3 numbers to the run's ``search_report.md``.

* TN01: meadow, exercise, no laser: down to 30 m by itself, "Gå lägre?", the pilot's yes,
  down to the map floor without the laser (about 21 m), never below 30 m before the yes.
* TN02: 24 m spruce, no laser: stops at the trees plus 15 m ("golvet nått").
* TN03: spruce with a GRF-250: the pilot's yes takes it to the trees plus 5 m, never
  nearer than 12 m to the surface the laser sees.
* TN04: meadow with a TF02-Pro: no confirmation over dark ground, a confirmation over the
  light meadow.
* TN05: live without the permit: the side look, never nearer than the cylinder's radius.
* TN06: SD-link lost on the lowest step: straight up to the start height, then the usual
  rule's hold at 23 s and home at 53 s.
* TN07: the laser lost below the trees plus 15 m: the climb to that floor within 1 s.
* TN08: a hole in the map under the person: counted as 30 m forest.
* TN09: a power line 20 m away: no descent below the start height within 30 m of it.
* TN13: a tree 8 m taller than the map: clear without the laser, a higher floor with it.
* TN14: the battery's reserve during the look: home.
* TN19: another person 25 m away: its own cylinder, none new around the target.
"""

from __future__ import annotations

import asyncio
import dataclasses
import math
import sys
import time
from pathlib import Path
from typing import Any

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent))

from inspect_rig import TnLook, tn_stack  # pyright: ignore[reportMissingImports]
from test_onboard_sitl import flight  # pyright: ignore[reportMissingImports]

from sd_safety.constants import (
    LINK_LOSS_DETECT_S,
    PERSON_KEEPOUT_M,
    SDLINK_GRACE_S,
    SDLINK_HOLD_S,
)
from sd_schemas.executor import ExecutorState as S
from sd_schemas.inspect import AUTO_FLOOR_ABOVE_GROUND_M, CLIMB_MIN_MS, match_radius_m
from sd_schemas.rangefinder import LASER_GAP_S
from sd_sim import inspect_world, paths

pytestmark = [
    pytest.mark.sitl,
    pytest.mark.slow,
    pytest.mark.inspect,
    pytest.mark.skipif(not paths.arducopter_bin().is_file(), reason="ArduCopter SITL is not built"),
]

TOL_M = 1.5  # GOTO_ALT_STEP_TOL_M: how near a height counts as reached
HOME_MODES = frozenset({"RTL", "SMART_RTL"})


def seeds(scenario: str) -> list[int]:
    return list(inspect_world.load(scenario, 1).seeds)


def world(scenario: str, seed: int, **changes: Any) -> inspect_world.InspectWorld:
    w = inspect_world.load(scenario, seed)
    return dataclasses.replace(w, **changes) if changes else w


def no_floor_m(w: inspect_world.InspectWorld, tree_m: float = 0.0) -> float:
    """The map's floor without the laser over ``tree_m`` of trees (spec §6.3): the trees
    within 10 m, plus 2 m, plus 0.5 m a year since the scan (at most 5), plus 15 m."""
    year = w.map_errors.scan_year
    growth = 0.0 if year is None else min(0.5 * (time.gmtime().tm_year - year), 5.0)
    return tree_m + 2.0 + growth + 15.0


def run(coro: Any) -> None:
    asyncio.run(coro)


# --- TN01: meadow, exercise, no laser ---------------------------------------------------------


@pytest.mark.budget_s(330)
@pytest.mark.parametrize("seed", seeds("TN01"))
def test_tn01_meadow_down_to_30_m_then_the_pilots_yes_to_the_map_floor(seed: int) -> None:
    w = world("TN01", seed)

    async def flown() -> None:
        with tn_stack(w) as sim:
            async with flight(sim) as f:
                look = TnLook(f, w)
                await look.start()
                await look.order()
                offer = await look.wait_event("inspect.lower_offer", 180)
                # By itself never below 30 m: the question comes at the automatic floor.
                assert abs(look.alt() - AUTO_FLOOR_ABOVE_GROUND_M) <= TOL_M, offer
                assert min(a for *_, a in look.track(look.t_order)) >= 30 - TOL_M
                assert offer["lowest_above_ground_m"] == pytest.approx(no_floor_m(w), abs=0.5)
                await look.yes()
                done = await look.waiting(120)
                assert done["end"] in ("clear", "floor"), done
                m = look.report()
                # Acceptance 8: the lowest height 20.9 ± 1.5 m over the meadow.
                assert m.lowest_above_ground_m == pytest.approx(no_floor_m(w), abs=TOL_M), m
                assert 20.9 - TOL_M <= m.lowest_above_ground_m <= 20.9 + TOL_M, m
                look.check_steps()
                assert not look.of("inspect.aborted")

    run(flown())


def truth(f: Any) -> Any:
    """What the drone can hit: the world, never the map."""
    return f.sim.stack.obstacle_scene()


def trees_at(f: Any, xy: tuple[float, float], radius_m: float = 10.0) -> float:
    """The tallest crown within ``radius_m`` of ``xy``, above the ground there."""
    scene = truth(f)
    return scene.top(*xy, radius_m, lines=False) - scene.ground.at1(*xy)


def phases(look: TnLook) -> set[str]:
    return {p.phase for _, p in look.progress()}


def reasons(look: TnLook) -> set[str | None]:
    return {p.floor_reason for _, p in look.progress()}


# --- TN02 / TN03: 24 m spruce, without the laser and with a GRF-250 ---------------------------


@pytest.mark.budget_s(330)
@pytest.mark.parametrize("seed", seeds("TN02"))
def test_tn02_spruce_without_the_laser_stops_at_the_trees_plus_15_m(seed: int) -> None:
    w = world("TN02", seed)

    async def flown() -> None:
        with tn_stack(w) as sim:
            async with flight(sim) as f:
                look = TnLook(f, w)
                await look.start()
                await look.order()
                done = await look.flown(answer=None)
                assert done is not None and done["end"] == "floor", done
                assert not look.of("inspect.lower_offer")  # above 30 m: nothing to ask
                m = look.report()
                tree_m = trees_at(f, w.estimate_enu)
                # Acceptance 9: the trees (the world's, not the map's) + 15 m.
                assert m.lowest_above_ground_m >= tree_m + 15.0 - TOL_M, (m, tree_m)
                assert m.lowest_above_ground_m <= no_floor_m(w, tree_m) + 2 * TOL_M, (m, tree_m)
                look.check_steps()

    run(flown())


@pytest.mark.budget_s(330)
@pytest.mark.parametrize("seed", seeds("TN03"))
def test_tn03_spruce_with_the_grf250_to_the_trees_plus_5_m_never_within_12_m(seed: int) -> None:
    w = world("TN03", seed)

    async def flown() -> None:
        with tn_stack(w) as sim:
            async with flight(sim) as f:
                look = TnLook(f, w)
                await look.start()
                await look.order()
                done = await look.flown("yes")
                assert done is not None and done["end"] in ("floor", "clear"), done
                assert any(p.laser_confirmed for _, p in look.progress()), "the laser"
                m = look.report()
                scene = truth(f)
                tree_m = trees_at(f, w.estimate_enu)
                # Acceptance 10: never nearer the surface the laser sees than 12 m ...
                track = look.track(look.t_order)
                near = min(z - scene.top(e, n, 0.0, lines=False) for _, e, n, z in track)
                assert near >= 12.0 - TOL_M, (near, m)
                # ... and the map's trees + 5 m with the laser confirmed.
                with_laser = no_floor_m(w, tree_m) - 15.0 + 5.0
                assert m.lowest_above_ground_m >= with_laser - TOL_M, (m, tree_m)
                look.check_steps()

    run(flown())


# --- TN04: the TF02-Pro over dark and light ground ----------------------------------------------


@pytest.mark.budget_s(330)
@pytest.mark.parametrize("light", [False, True], ids=["dark", "light"])
@pytest.mark.parametrize("seed", seeds("TN04"))
def test_tn04_the_tf02_confirms_over_the_light_meadow_only(seed: int, light: bool) -> None:
    w = world("TN04", seed) if light else world("TN04", seed, light=())

    async def flown() -> None:
        with tn_stack(w) as sim:
            async with flight(sim) as f:
                look = TnLook(f, w)
                await look.start()
                await look.order()
                done = await look.flown("yes")
                assert done is not None, look.of("inspect.aborted")
                m = look.report()
                confirmed = any(p.laser_confirmed for _, p in look.progress())
                if light:
                    # 40 m over light ground: confirmed, the floor the map's + 5 m.
                    assert confirmed
                    floor = max(no_floor_m(w) - 15.0 + 5.0, w.alt_min_agl_m)
                    assert m.lowest_above_ground_m == pytest.approx(floor, abs=TOL_M), m
                else:
                    # 13.5 m over dark ground: never from 21 m, the floor without it.
                    assert not confirmed
                    assert m.lowest_above_ground_m == pytest.approx(no_floor_m(w), abs=TOL_M), m
                look.check_steps()

    run(flown())


# --- TN05: live without the permit ------------------------------------------------------------


@pytest.mark.budget_s(360)
@pytest.mark.parametrize("seed", seeds("TN05"))
def test_tn05_live_without_the_permit_never_nearer_than_the_cylinder(seed: int) -> None:
    w = world("TN05", seed)
    radius = PERSON_KEEPOUT_M + w.sigma95_m

    async def flown() -> None:
        with tn_stack(w) as sim:
            async with flight(sim) as f:
                look = TnLook(f, w)
                await look.start()
                await look.order(refused_ok=True)
                if look.ack.stage == "running":
                    done = await look.flown(answer=None)
                    if done is not None:
                        assert done["end"] == "side_look", done
                        assert done["still_ids"], done
                        # The side look's stills: the main camera only (the tele cannot
                        # see 25 degrees off straight down).
                        assert {s.camera for s in look.shots} == {"main"}
                    else:
                        (aborted,) = look.of("inspect.aborted")
                        assert aborted["reason"] == "failed", aborted
                else:
                    assert "ONB_INSPECT_GEOMETRY" in look.ack.reason_codes, look.ack
                assert not look.of("inspect.lower_offer")
                track = look.track(look.t_order)
                sideways = min(math.dist((e, n), w.estimate_enu) for _, e, n, _ in track)
                assert sideways >= radius - 1.0, (sideways, radius)
                m = look.report()
                # Acceptance 11 (the candidate is where the person is in TN05).
                assert m.min_target_horizontal_m >= radius - 1.0, m

    run(flown())


# --- TN06: the SD-link lost on the lowest step ------------------------------------------------


@pytest.mark.budget_s(420)
@pytest.mark.parametrize("seed", seeds("TN06"))
def test_tn06_link_lost_on_the_lowest_step_climbs_then_the_usual_rule(seed: int) -> None:
    w = world("TN06", seed)
    lowest = no_floor_m(w)
    hold_at = LINK_LOSS_DETECT_S + SDLINK_GRACE_S
    rtl_at = hold_at + SDLINK_HOLD_S

    async def flown() -> None:
        with tn_stack(w) as sim:
            async with flight(sim) as f:
                look = TnLook(f, w)
                await look.start()
                await look.order()
                await look.flown("yes", until=lambda: look.alt() <= lowest + TOL_M)
                assert look.t_yes is not None and not look.of("inspect.done")
                f.proxy.broken = True
                cut = time.monotonic()
                try:
                    states: list[tuple[float, S]] = []
                    last = f.ex.state
                    while time.monotonic() - cut < rtl_at + 4.0:
                        await asyncio.sleep(0.1)
                        if f.ex.state != last:
                            last = f.ex.state
                            states.append((time.monotonic() - cut, last))
                finally:
                    f.proxy.broken = False
                (aborted,) = look.of("inspect.aborted")
                assert aborted["reason"] == "link_loss", aborted
                climb = look.first("executor.inspect_climb")
                assert climb is not None, look.events[-5:]
                t_climb, data = climb
                # Acceptance 12: up within 1 s of the loss being seen (4 s of silence) ...
                assert t_climb - cut <= LINK_LOSS_DETECT_S + 1.0, t_climb - cut
                assert data["alt_m"] == pytest.approx(w.start_alt_m, abs=0.5), data
                rising = [(t, e, n, z) for t, e, n, z in look.track(t_climb)]
                top = next((t for t, *_, z in rising if z >= w.start_alt_m - TOL_M), None)
                assert top is not None, rising[-3:]
                start_z = rising[0][3]
                assert top - t_climb <= (w.start_alt_m - start_z) / CLIMB_MIN_MS + 3.0
                # ... straight up: no more than 3 m sideways on the way.
                e0, n0 = rising[0][1], rising[0][2]
                drift = max(math.dist((e, n), (e0, n0)) for t, e, n, _ in rising if t <= top)
                assert drift <= 3.0, drift
                # Then the usual rule: hovering at 23 s, home at 53 s.
                at_hold = [s for t, s in states if t <= hold_at + 2.0]
                assert at_hold and at_hold[-1] == S.HOLD, states
                rtl = next((t for t, s in states if s == S.RTL), None)
                assert rtl is not None and abs(rtl - rtl_at) <= 2.0, states
                look.report(until=top)

    run(flown())


# --- TN07: the laser lost below the map's floor without it ------------------------------------


@pytest.mark.budget_s(360)
@pytest.mark.parametrize("seed", seeds("TN07"))
def test_tn07_laser_lost_below_the_trees_plus_15_m_climbs_to_that_floor(seed: int) -> None:
    w = world("TN07", seed)
    floor = no_floor_m(w)

    async def flown() -> None:
        with tn_stack(w) as sim:
            async with flight(sim) as f:
                look = TnLook(f, w)
                await look.start()
                await look.order()
                await look.flown("yes", until=lambda: look.alt() <= floor - 3.0)
                assert any(p.laser_confirmed for _, p in look.progress()), "the laser"
                await asyncio.to_thread(sim.stack.set_rangefinder_fault, "dropout")
                cut = time.monotonic()
                try:
                    lost = await look.wait_event("inspect.laser_lost", 10)
                    t_lost = look.first("inspect.laser_lost")
                    assert t_lost is not None
                    # Acceptance 13: the climb within 1 s of the laser's last reading (the
                    # reader calls it gone after LASER_GAP_S without one).
                    assert t_lost[0] - cut <= LASER_GAP_S + 1.0, (t_lost[0] - cut, lost)
                    done = await look.flown(answer=None, limit_s=60)
                finally:
                    await asyncio.to_thread(sim.stack.set_rangefinder_fault, None)
                assert done is not None and done["end"] == "laser_lost", done
                assert look.alt() >= floor - TOL_M, look.alt()
                m = look.report()
                assert m.lowest_above_ground_m < floor - 2.0, m  # it was below, with the laser
                look.check_steps()

    run(flown())


# --- TN08: a hole in the map under the person -------------------------------------------------


@pytest.mark.budget_s(330)
@pytest.mark.parametrize("seed", seeds("TN08"))
def test_tn08_a_hole_in_the_map_counts_as_30_m_forest(seed: int) -> None:
    w = world("TN08", seed)

    async def flown() -> None:
        with tn_stack(w) as sim:
            async with flight(sim) as f:
                look = TnLook(f, w)
                await look.start()
                await look.order()
                done = await look.flown(answer=None)
                assert done is not None and done["end"] == "floor", done
                assert "unknown" in reasons(look), reasons(look)
                assert not look.of("inspect.lower_offer")
                m = look.report()
                assert m.lowest_above_ground_m >= no_floor_m(w, 30.0) - TOL_M, m
                look.check_steps()

    run(flown())


# --- TN09: a power line 20 m from the person --------------------------------------------------


def _from_line_m(xy: tuple[float, float], a: tuple[float, float], b: tuple[float, float]) -> float:
    ax, ay = a
    bx, by = b
    dx, dy = bx - ax, by - ay
    k = max(0.0, min(1.0, ((xy[0] - ax) * dx + (xy[1] - ay) * dy) / (dx * dx + dy * dy)))
    return math.dist(xy, (ax + k * dx, ay + k * dy))


@pytest.mark.budget_s(300)
@pytest.mark.parametrize("seed", seeds("TN09"))
def test_tn09_no_descent_within_30_m_of_a_power_line(seed: int) -> None:
    w = world("TN09", seed)
    (line,) = w.lines

    async def flown() -> None:
        with tn_stack(w) as sim:
            async with flight(sim) as f:
                look = TnLook(f, w)
                await look.start()
                await look.order()
                done = await look.flown(answer=None)
                assert done is not None and done["end"] == "floor", done
                assert "powerline" in reasons(look), reasons(look)
                assert not look.of("inspect.lower_offer")
                # Acceptance 14: never below the start height within 30 m of the line.
                a, b = (line.a[0], line.a[1]), (line.b[0], line.b[1])
                near = [
                    z
                    for _, e, n, z in look.track(look.t_order)
                    if _from_line_m((e, n), a, b) <= 30.0
                ]
                assert near and min(near) >= w.start_alt_m - TOL_M, min(near)
                look.report()
                look.check_steps()

    run(flown())


# --- TN13: a tree 8 m taller than the map -----------------------------------------------------


@pytest.mark.budget_s(330)
@pytest.mark.parametrize("laser", ["none", "grf250"])
@pytest.mark.parametrize("seed", seeds("TN13"))
def test_tn13_a_tree_the_map_has_8_m_short(seed: int, laser: str) -> None:
    w = world("TN13", seed, laser=laser)

    async def flown() -> None:
        with tn_stack(w) as sim:
            async with flight(sim) as f:
                look = TnLook(f, w)
                await look.start()
                await look.order()
                done = await look.flown("yes")
                assert done is not None, look.of("inspect.aborted")
                # Acceptance 18: the margins hold the 8 m (``report``: 5 m clearance) ...
                look.report()
                if laser == "grf250":
                    # ... and with the laser the floor rises once it sees the crown.
                    map_tree = trees_at(f, w.estimate_enu) - 8.0
                    with_laser = no_floor_m(w, map_tree) - 15.0 + 5.0
                    seen = [
                        p.floor_lowest_above_ground_m
                        for t, p in look.progress()
                        if p.laser_confirmed and p.floor_lowest_above_ground_m is not None
                    ]
                    assert seen and max(seen) >= with_laser + 2.0, (seen, with_laser)
                look.check_steps()

    run(flown())


# --- TN14: the battery's reserve during the look ----------------------------------------------


@pytest.mark.budget_s(420)
@pytest.mark.parametrize("seed", seeds("TN14"))
def test_tn14_the_battery_reserve_during_the_look_flies_home(seed: int) -> None:
    w = world("TN14", seed)

    async def flown() -> None:
        with tn_stack(w) as sim:
            async with flight(sim) as f:
                look = TnLook(f, w)
                await look.start(battery_rtl_pct=25.0)
                await look.order()
                await look.flown("yes", until=lambda: look.t_yes is not None)
                await asyncio.to_thread(sim.harness.battery_set_remaining, 26.0)
                await look.flown("yes", limit_s=200, until=lambda: f.ex.state == S.RTL)
                assert f.ex.state == S.RTL, (f.ex.state, look.events[-5:])
                await f.until(lambda: bool(look.of("inspect.aborted")), 5, "the look aborted")
                (aborted,) = look.of("inspect.aborted")
                assert aborted["reason"] == "battery", aborted
                assert not look.of("inspect.done")
                look.report(until=look.first("inspect.aborted")[0])  # type: ignore[index]

    run(flown())


# --- TN19: another person 25 m from the target ------------------------------------------------


@pytest.mark.budget_s(360)
@pytest.mark.parametrize("seed", seeds("TN19"))
def test_tn19_another_person_gets_a_cylinder_the_target_none(seed: int) -> None:
    w = world("TN19", seed)
    (other,) = w.others

    async def flown() -> None:
        with tn_stack(w) as sim:
            async with flight(sim) as f:
                look = TnLook(f, w)
                await look.start()
                await look.order()
                # "Nej, stanna här" (the owner's decision): the look ends where it hovers.
                done = await look.flown("no")
                assert done is not None, look.of("inspect.aborted")
                if look.of("inspect.lower_offer"):
                    assert done["end"] == "lower_declined", done
                    t_no = look.first("inspect.lower_offer")[0]  # type: ignore[index]
                    after = [z for _, _, _, z in look.track(t_no)]
                    assert min(after) >= AUTO_FLOOR_ABOVE_GROUND_M - TOL_M, min(after)
                # Acceptance 19: the other person's cylinder, and none new near the target.
                radius = match_radius_m(w.sigma95_m)
                centres = [f.plane.xy(k.lat, k.lon) for k, _ in f.ex.local_keepouts.values()]
                assert any(math.dist(c, other.enu) <= 5.0 for c in centres), centres
                assert all(math.dist(c, w.estimate_enu) > radius for c in centres), centres
                look.report()
                look.check_steps()

    run(flown())
