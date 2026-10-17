"""Failsafes against ArduCopter SITL: battery, the whole radio, SD-link, the FC fence.

* SC06a: the battery reaches the charter's reserve (25 % with ``charter_batt25``): the
  onboard rule flies home within 3 s, the AI's orders are refused, it lands within 3 m.
* SC06c: the flight controller's own battery failsafe with no onboard computer and no
  ground link: SmartRTL (or RTL) within 5 s of the low level.
* SC07a/b: the whole radio down for 10 s: nothing changes and no AI order gets through;
  for 90 s: the flight controller's GCS failsafe flies home after FS_GCS_TIMEOUT and
  lands before the radio is back, and the mission does not resume on its own.
* SC14b: only SD-link down for 30 s: the onboard computer holds after 23 s, the ground
  refuses orders as LINK_DOWN, and after the link is back the drone keeps holding.
* SC16: the harness flies out of the charter past the gate on purpose: the flight
  controller's fence takes over (SmartRTL/RTL) and the drone lands.
"""

from __future__ import annotations

import asyncio
import sys
import time
from pathlib import Path
from typing import Any

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent))

from scenario_rig import (  # pyright: ignore[reportMissingImports]
    VID,
    Scenario,
    fresh_db,
    migrated_db,
    passwords,
    pin,
    scenario,
    sim,
)

from sd_safety.constants import LINK_LOSS_DETECT_S, SDLINK_GRACE_S
from sd_safety.targets import Plane, distance_m, polygon_from_geojson
from sd_sim.params import merge_layers
from sd_sim.sitl import SAFETY_PARAM_FILE

pytestmark = [pytest.mark.sitl, pytest.mark.slow]
__all__ = ["fresh_db", "migrated_db", "passwords", "sim"]

PARAMS = Path(__file__).resolve().parents[2] / "sitl/params"
SAFETY = merge_layers([PARAMS / SAFETY_PARAM_FILE])
HOME_MODES = ("SMART_RTL", "RTL")
SKILL_RTL_S = 3.0
FS_SLACK_S = 2.0


async def to_p1(sc: Scenario, speed_ms: float = 6.0) -> None:
    lat, lon = pin("P1")
    res = await sc.order("goto", {"lat": lat, "lon": lon, "alt_m": 40, "speed_ms": speed_ms})
    assert res.get("decision") == "ALLOW", res


@pytest.mark.budget_s(480)
async def test_sc06a_battery_reserve_flies_home(sim: Any, migrated_db: Any, tmp_path: Path) -> None:
    async with scenario(sim, migrated_db, tmp_path) as sc:
        await sc.airborne("charter_batt25")
        await to_p1(sc, 4.0)
        await sc.until(lambda: sc.from_home_m() > 30.0, 60, "on the way")
        await sc.harness(sim.harness.battery_set_remaining, 26)
        await sc.until(
            lambda: sc.tel.battery_pct is not None and sc.tel.battery_pct <= 25.0,
            180,
            "battery at the reserve",
        )
        t0 = time.monotonic()
        await sc.until(lambda: sc.mode() in HOME_MODES, SKILL_RTL_S, "flying home")
        assert time.monotonic() - t0 <= SKILL_RTL_S
        assert sc.mode() == "SMART_RTL"
        await asyncio.sleep(3.1)
        res = await sc.order("goto", {"lat": pin("P1")[0], "lon": pin("P1")[1], "alt_m": 40})
        assert res.get("decision") == "REJECT", res
        assert {"FAILSAFE_ACTIVE", "BATTERY_RESERVE"} & set(res["reason_codes"]), res
        line = await sc.say("Fortsätt.")
        assert line["handled_by"] is None
        await sc.until(lambda: not sc.tel.armed, 180, "landed")
        assert sc.from_home_m() <= 3.0
        assert sc.tel.battery_pct is not None and sc.tel.battery_pct >= 8.0
        await sc.check_invariants()


@pytest.mark.budget_s(420)
async def test_sc06c_fc_battery_failsafe_alone(sim: Any, migrated_db: Any, tmp_path: Path) -> None:
    async with scenario(sim, migrated_db, tmp_path) as sc:
        await sc.airborne()
        await to_p1(sc, 4.0)
        await sc.until(lambda: sc.from_home_m() > 30.0, 60, "on the way")
        await sc.kill_onboard()
        sc.radio(mavlink=False)
        try:
            # Just above the FC's 15 % (BATT_LOW_MAH): the pack crosses it as it drains.
            await sc.harness(sim.harness.battery_set_remaining, 16)
            low = 100.0 * SAFETY["BATT_LOW_MAH"] / SAFETY["BATT_CAPACITY"]
            await sc.until(
                lambda: 0 <= sim.harness.state.battery_pct <= round(low),
                120,
                "the pack at the FC's low level",
            )
            t0 = time.monotonic()
            await sc.until(lambda: sc.fc_mode() in HOME_MODES, 5.0, "the FC's battery failsafe")
            assert time.monotonic() - t0 <= 5.0
            await sc.until(lambda: not sim.harness.state.armed, 180, "landed")
        finally:
            sc.radio(mavlink=True)
        s = sim.harness.state
        assert distance_m(sc.r.home[0], sc.r.home[1], s.lat, s.lon) <= 3.0
        await sc.check_invariants()


@pytest.mark.budget_s(300)
async def test_sc07a_radio_down_10s(sim: Any, migrated_db: Any, tmp_path: Path) -> None:
    async with scenario(sim, migrated_db, tmp_path, proxied_link=True) as sc:
        await sc.airborne()
        await to_p1(sc)
        await sc.wait_command("goto", "done", 90)
        await asyncio.sleep(3.1)
        before = (sim.harness.state.lat, sim.harness.state.lon)
        mode = sc.fc_mode()
        sc.radio(mavlink=False, sdlink=False)
        down = time.monotonic()
        tried: list[str] = []
        try:
            lat, lon = pin("OUT")
            while time.monotonic() - down < 10.0:
                for skill, params in (
                    ("goto", {"lat": lat, "lon": lon, "alt_m": 40}),
                    ("hold", {}),
                ):
                    res = await sc.order(skill, params)
                    tried.append(str(res.get("command_id")))
                    await asyncio.sleep(1.5)
                assert sc.fc_mode() == mode
        finally:
            sc.radio(mavlink=True, sdlink=True)
        s = sim.harness.state
        assert distance_m(*before, s.lat, s.lon) <= 3.0
        assert sc.fc_mode() == mode
        accepted = [
            c
            for c in await sc.commands()
            if str(c["id"]) in tried and c["status"] != "rejected_by_gate"
        ]
        assert accepted == [], accepted
        await sc.until(lambda: sc.r.link.up(VID), 10, "SD-link back")
        assert (await sc.press("return_home")).status_code == 202
        await sc.until(lambda: not sc.tel.armed, 180, "landed")
        await sc.check_invariants()


@pytest.mark.budget_s(360)
async def test_sc07b_radio_down_90s(sim: Any, migrated_db: Any, tmp_path: Path) -> None:
    async with scenario(sim, migrated_db, tmp_path, proxied_link=True) as sc:
        await sc.airborne()
        await to_p1(sc)
        await sc.wait_command("goto", "done", 90)
        sc.radio(mavlink=False, sdlink=False)
        down = time.monotonic()
        try:
            limit = SAFETY["FS_GCS_TIMEOUT"] + FS_SLACK_S + 1.5  # last 250 heartbeat <= 1 s
            await sc.until(lambda: sc.fc_mode() in HOME_MODES, limit, "the GCS failsafe")
            took = time.monotonic() - down
            assert took >= SAFETY["FS_GCS_TIMEOUT"] - 1.5, took
            # L-LAND from the mode change; the radio stays down until it has landed.
            await sc.until(lambda: not sim.harness.state.armed, 180, "landed")
            s = sim.harness.state
            assert distance_m(sc.r.home[0], sc.r.home[1], s.lat, s.lon) <= 3.0
            await asyncio.sleep(max(0.0, 90.0 - (time.monotonic() - down)))
        finally:
            sc.radio(mavlink=True, sdlink=True)
        await sc.until(lambda: sc.r.core.vehicle.telemetry.fc_connected, 10, "MAVLink back")
        await sc.until(lambda: sc.r.link.up(VID), 15, "SD-link back")
        await asyncio.sleep(5.0)
        assert not sc.tel.armed  # it does not take off again on its own
        assert not [c for c in await sc.commands() if c["created_at"] and c["status"] == "running"]
        await sc.check_invariants()


@pytest.mark.budget_s(300)
async def test_sc14b_sdlink_down_30s(sim: Any, migrated_db: Any, tmp_path: Path) -> None:
    async with scenario(sim, migrated_db, tmp_path, proxied_link=True) as sc:
        await sc.airborne()
        await to_p1(sc, 3.0)
        await sc.wait_command("goto", "running", 10)
        sc.radio(sdlink=False)
        down = time.monotonic()
        hold_at = LINK_LOSS_DETECT_S + SDLINK_GRACE_S
        try:
            await sc.until(lambda: not sc.r.link.up(VID), LINK_LOSS_DETECT_S + 8.0, "link lost")
            res = await sc.order("hold", {})
            assert res.get("decision") == "REJECT" and "LINK_DOWN" in res["reason_codes"], res
            await sc.until(
                lambda: sc.fc_mode() in ("LOITER", "BRAKE"), hold_at + 2.0, "onboard hold"
            )
            took = time.monotonic() - down
            assert hold_at - 1.5 <= took <= hold_at + 1.5, took
            await asyncio.sleep(max(0.0, 30.0 - (time.monotonic() - down)))
        finally:
            sc.radio(sdlink=True)
        await sc.until(lambda: sc.r.link.up(VID), 15, "SD-link back")
        here = (sim.harness.state.lat, sim.harness.state.lon)
        await sc.hold_for(
            15.0,
            lambda: distance_m(*here, sim.harness.state.lat, sim.harness.state.lon) <= 3.0,
            "keeps holding after the link is back",
        )
        live = await sc.onboard_state()
        assert live["executor_state"] == "HOLD" and live["hold_reason"] == "link_loss", live
        assert (await sc.press("return_home")).status_code == 202
        await sc.until(lambda: not sc.tel.armed, 180, "landed")
        await sc.check_invariants()


@pytest.mark.budget_s(360)
async def test_sc16_fc_fence_is_the_last_line(sim: Any, migrated_db: Any, tmp_path: Path) -> None:
    async with scenario(sim, migrated_db, tmp_path) as sc:
        await sc.airborne()
        charter = (await sc.r.mission(sc.mid))["charter"]["body"]
        plane = Plane(*sc.r.home)
        area = polygon_from_geojson(charter["area"], plane)
        fence_margin = charter["geofence_buffer_m"] + 10.0  # the FC polygon: area + buffer + 10
        stop = {"done": False}
        flight = asyncio.ensure_future(
            sc.harness(sim.harness.guided_velocity, 0.0, 8.0, 0.0, 60.0, lambda: stop["done"])
        )
        try:
            await sc.until(lambda: sc.fc_mode() in (*HOME_MODES, "LAND"), 60, "the FC fence")
        finally:
            stop["done"] = True
            await flight
        await sc.until(lambda: not sim.harness.state.armed, 240, "landed")
        worst = max(
            area.boundary_distance(plane.xy(s.lat, s.lon))
            for s in sc.track
            if s.armed and not area.contains(plane.xy(s.lat, s.lon))
        )
        assert worst <= fence_margin + 20.0, worst
        live = await sc.onboard_state()
        assert live["executor_state"] in ("FAILSAFE", "HELD", "LANDED", "PREFLIGHT", "IDLE"), live
        await sc.check_invariants(exempt=("INV1", "INV3", "INV4"))


@pytest.mark.budget_s(300)
async def test_sc16b_fc_fence_height(sim: Any, migrated_db: Any, tmp_path: Path) -> None:
    """The harness climbs straight up past the gate: the FC's height fence takes over and
    the drone never goes more than 5 m above FENCE_ALT_MAX."""
    async with scenario(sim, migrated_db, tmp_path) as sc:
        await sc.airborne()
        stop = {"done": False}
        flight = asyncio.ensure_future(
            sc.harness(sim.harness.guided_velocity, 0.0, 0.0, -3.0, 60.0, lambda: stop["done"])
        )
        try:
            await sc.until(lambda: sc.fc_mode() in (*HOME_MODES, "LAND"), 60, "the FC fence")
        finally:
            stop["done"] = True
            await flight
        await sc.until(lambda: not sim.harness.state.armed, 240, "landed")
        top = max(s.alt_m for s in sc.track)
        assert top <= SAFETY["FENCE_ALT_MAX"] + 5.0, top
        await sc.check_invariants(exempt=("INV1", "INV3"))


@pytest.mark.budget_s(300)
async def test_gps_lost_during_flight(sim: Any, migrated_db: Any, tmp_path: Path) -> None:
    """GPS lost on the way to P1: the flight controller's own EKF failsafe
    (FS_EKF_ACTION) takes over, the onboard computer sends no more targets and the
    mission shows the failsafe."""
    async with scenario(sim, migrated_db, tmp_path) as sc:
        await sc.airborne()
        await to_p1(sc, 4.0)
        await sc.until(lambda: sc.from_home_m() > 30.0, 60, "on the way")
        await sc.harness(sim.harness.set_sim_param, "SIM_GPS1_ENABLE", 0)
        try:
            want = {1.0: "LAND", 2.0: "ALT_HOLD", 3.0: "LAND"}[SAFETY["FS_EKF_ACTION"]]
            await sc.until(lambda: sc.fc_mode() == want, 60, "the EKF failsafe")
            await sc.r.wait_status(sc.mid, "FAILSAFE", 10)
            live = await sc.onboard_state()
            assert live["executor_state"] in ("FAILSAFE", "HELD"), live
            await asyncio.sleep(3.1)
            lat, lon = pin("P1")
            res = await sc.order("goto", {"lat": lat, "lon": lon, "alt_m": 40})
            assert res.get("decision") == "REJECT", res
            await sc.until(lambda: not sim.harness.state.armed, 180, "landed")
        finally:
            await sc.harness(sim.harness.set_sim_param, "SIM_GPS1_ENABLE", 1)
        await sc.check_invariants(exempt=("INV2", "INV4"))
