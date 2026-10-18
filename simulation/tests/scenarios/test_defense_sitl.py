"""Defence in depth: what the flight controller does by itself, past our software.

* The FC fence against orders that never went through the gate: a test script sends
  MAVLink straight to the flight controller (the harness, on purpose) with a target 130 m
  above home, and one 520 m away. The fence acts: the height never goes above
  ``FENCE_ALT_MAX`` + 5 m, the distance from home never above ``FENCE_RADIUS`` + 10 m;
  the vehicle lands at home.
* GPS lost during the search: the flight controller's own EKF failsafe (``FS_EKF_ACTION``,
  LAND) acts; the onboard computer sends no GUIDED targets afterwards; the mission goes
  to ``FAILSAFE``.
"""

from __future__ import annotations

import asyncio
import time
from pathlib import Path
from typing import Any

import pytest
from rig_faults import (  # pyright: ignore[reportMissingImports]
    HARNESS_GOTO,
    brain_started,
    ground_vehicle_events,
    harness_goto,
)
from scenario_rig import AI_RATE_S, Scenario, scenario  # pyright: ignore[reportMissingImports]

from sd_mav import CopterMode
from sd_safety.targets import distance_m
from sd_sim import expect
from sd_sim.invariants import Sent

pytestmark = [pytest.mark.sitl, pytest.mark.slow]

ABOVE_FENCE_M = 5.0
PAST_RADIUS_M = 10.0
HIGH_TARGET_M = 130.0
FAR_TARGET_M = 520.0
CONTROL_EAST_M = 40.0
CONTROL_SLACK_M = 5.0
CONTROL_S = 20.0
PUSH_S = 40.0  # the harness sends its target this long (or until the fence acts)
EKF_FS_S = 30.0  # GPS gone -> the EKF failsafe's mode
GPS_AFTER_S = 20.0  # into the search
LAND_GPS_S = 120.0  # LAND without GPS from 40 m
MISSION_S = 10.0  # the failsafe mode -> the mission's FAILSAFE
# ArduCopter's FS_EKF_ACTION: 1 = LAND, 2 = ALT_HOLD, 3 = LAND even in STABILIZE.
EKF_ACTION_MODE = {1: CopterMode.LAND, 2: CopterMode.ALT_HOLD, 3: CopterMode.LAND}


@pytest.mark.parametrize(
    "variant",
    [
        pytest.param("high", marks=pytest.mark.budget_s(240), id="high"),
        pytest.param("far", marks=pytest.mark.budget_s(240), id="far"),
    ],
)
@pytest.mark.script("SC16")
async def test_fc_fence_against_direct_mavlink(
    sim: Any, migrated_db: Any, tmp_path: Path, variant: str
) -> None:
    async with scenario(
        f"test_fc_fence_direct[{variant}]", sim, migrated_db, tmp_path, "SC16"
    ) as sc:
        await _direct(sc, variant)


async def _direct(sc: Scenario, variant: str) -> None:
    fc_texts = ground_vehicle_events(sc)
    await sc.arrange_airborne()
    alt_max = expect.value("FENCE_ALT_MAX")
    radius = expect.value("FENCE_RADIUS")
    chain = {int(m) for m in expect.fs_chain("FENCE_ACTION")}
    home = sc.r.home
    # Control: a target inside the fence on the same path is flown, so the path works.
    near_lat, near_lon = sc.at(CONTROL_EAST_M, 0.0)
    await harness_goto(sc, near_lat, near_lon, sc.tel.alt_rel_home_m, CONTROL_S)
    moved = sc.from_home_m()
    sc.criterion(
        "DEF.0",
        moved >= CONTROL_EAST_M - CONTROL_SLACK_M,
        "harnessens mål innanför staketet flygs (vägen fungerar)",
        measured=round(moved, 1),
        limit=CONTROL_EAST_M - CONTROL_SLACK_M,
    )
    first = len(sc.tl.samples)
    texts = len(fc_texts)
    if variant == "high":
        lat, lon, alt = sc.tel.lat, sc.tel.lon, HIGH_TARGET_M
    else:
        (lat, lon), alt = sc.at(FAR_TARGET_M, 0.0), sc.tel.alt_rel_home_m
    # The harness keeps sending the target; the flight controller either refuses a target
    # outside its fence or breaches and takes the fence action.
    await harness_goto(sc, lat, lon, alt, PUSH_S)
    samples = sc.tl.samples[first:]
    top = max(s.alt_m for s in samples)
    far = max(distance_m(home[0], home[1], s.lat, s.lon) for s in samples)
    said = [x[2] for x in fc_texts[texts:]]
    acted = any(s.mode in chain for s in samples)
    sc.criterion(
        "DEF.1",
        top <= alt_max + ABOVE_FENCE_M and far <= radius + PAST_RADIUS_M,
        "FC-staketet håller emot MAVLink förbi grinden",
        height_m=round(top, 1),
        height_limit_m=alt_max + ABOVE_FENCE_M,
        distance_m=round(far, 1),
        distance_limit_m=radius + PAST_RADIUS_M,
        fence_action=acted,
        fc_texts=said[-4:],
    )
    started = await sc.press("return_home")
    await sc.until(lambda: not sc.tel.armed, expect.L_LAND_S, "landad")
    sc.criterion(
        "DEF.2",
        sc.from_home_m() <= expect.L_LAND_M,
        "landar hemma (L-LAND)",
        measured=round(sc.from_home_m(), 1),
        limit=expect.L_LAND_M,
        land_s=round(time.monotonic() - started, 1),
    )
    exempt = ("INV1",) if variant == "high" else ("INV4",)
    await sc.finish(exempt, harness_extra=HARNESS_GOTO)


@pytest.mark.budget_s(200)
@pytest.mark.script("SC08")
async def test_gps_loss_in_search(sim: Any, migrated_db: Any, tmp_path: Path) -> None:
    async with scenario("test_gps_loss[search]", sim, migrated_db, tmp_path, "SC08") as sc:
        await _gps_loss(sc)


async def _gps_loss(sc: Scenario) -> None:
    await sc.arrange_airborne()
    await asyncio.sleep(AI_RATE_S)
    await sc.say("Sök av rutan.")
    await brain_started(sc, "survey_sector")
    await asyncio.sleep(GPS_AFTER_S)
    lost = time.monotonic()
    sent = len(sc.tl.sent)
    sc.tl.sent.append(Sent(lost, "harness", "PARAM_SET", param_id="SIM_GPS1_ENABLE"))
    await asyncio.to_thread(sc.sim.harness.set_sim_param, "SIM_GPS1_ENABLE", 0)
    ekf_mode = int(EKF_ACTION_MODE[int(expect.value("FS_EKF_ACTION"))])
    took = await sc.until(lambda: sc.mode() == ekf_mode, EKF_FS_S, "EKF-failsafens läge")

    async def failsafe() -> bool:
        return await sc.r.status(sc.mission_id) == "FAILSAFE"

    await sc.until(failsafe, MISSION_S, "uppdraget FAILSAFE")
    await sc.until(lambda: not sc.tel.armed, LAND_GPS_S, "landad utan GPS")
    guided = [
        m
        for m in sc.tl.sent[sent:]
        if m.sender == "onboard"
        and (m.type.startswith("SET_POSITION_TARGET") or m.mode == CopterMode.GUIDED)
    ]
    sc.criterion(
        "GPS.1",
        not guided,
        "FS_EKF_ACTION tar över och uppdraget blir FAILSAFE; inga GUIDED-mål från ombord",
        failsafe_s=round(took, 1),
        mode=ekf_mode,
        guided=len(guided),
    )
    await sc.finish()
