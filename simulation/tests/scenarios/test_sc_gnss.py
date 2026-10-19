"""The GPS against ArduCopter SITL: lost, back, spoofed, and a strong wind that is neither.

The drone flies 200 m out (P1) in a light wind, then the simulator's GPS is changed:

* GP1/GP2: the GPS off (``SIM_GPS1_ENABLE 0``): the flight controller dead-reckons and
  flies home (``FS_DR_ENABLE 2``) within 12 s, the onboard computer hands off (FAILSAFE);
  after ``FS_DR_TIMEOUT`` the EKF failsafe lands. With the GPS back after 15 s RTL goes
  on and lands at home.
* GP3/GP4: a spoofed position, a 100 m jump (S1) or a slow climb of the GPS height
  (S2): L-SPOOF switches the EKF off GPS (source set 2) and the flight controller flies
  home on dead reckoning. The simulator's true position (``SIMSTATE``) is the judge.
* GP5: a 15 m jump for 5 s (a glitch, below S1): nothing happens.
* GP6: a whole flight in an 8 m/s wind: no GPS alarm.
* GP7: the GPS off while hovering beside a find with the whole radio down: the flight
  controller's failsafe wins and the hover ends as ``fc_failsafe``.
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
    Scenario,
    fresh_db,
    migrated_db,
    passwords,
    pin,
    scenario,
    sim,
    sim_stack,
)
from test_sc_linkloss import (  # pyright: ignore[reportMissingImports]
    _find_hover,
    back_and_land,
    fc_texts,
    onboard_events,
)

from sd_safety.constants import FS_DR_TIMEOUT_S, SPOOF_ALT_BASELINE_S
from sd_safety.targets import distance_m

pytestmark = [pytest.mark.sitl, pytest.mark.slow]
__all__ = ["fresh_db", "migrated_db", "passwords", "sim", "sim_stack"]

HOME_MODES = ("SMART_RTL", "RTL")
DR_RTL_BY_S = 12.0
SPOOF_BY_S = 4.0
RAMP_BY_S = 30.0
JUMP_DEG = 0.0009  # ~100 m north
GLITCH_DEG = 0.000135  # ~15 m north
M_PER_DEG = 111_111.0
GPS_CODES = ("GPS_WEAK", "GPS_LOST", "GPS_SPOOF", "GPS_GLITCH", "DEAD_RECKONING")


async def out_to_p1(sc: Scenario, wind_ms: float = 4.0) -> None:
    """Airborne, a light wind, 200 m out at P1 and holding there."""
    await sc.airborne()
    await sc.harness(sc.sim.harness.set_sim_param, "SIM_WIND_SPD", wind_ms)
    lat, lon = pin("P1")
    res = await sc.order("goto", {"lat": lat, "lon": lon, "alt_m": 40, "speed_ms": 8})
    assert res.get("decision") == "ALLOW", res
    await sc.wait_command("goto", "done", 90)
    # the spoofing guard's takeoff height difference is long settled
    assert time.monotonic() - sc.track[0].t > SPOOF_ALT_BASELINE_S


def true_from_home_m(sc: Scenario) -> float:
    """Where the simulated drone really is (SIMSTATE), not what the EKF believes."""
    s = sc.sim.harness.state
    assert s.true_lat is not None and s.true_lon is not None
    return distance_m(sc.r.home[0], sc.r.home[1], s.true_lat, s.true_lon)


def alert_codes(sc: Scenario) -> list[str]:
    return [str(a.get("code")) for a in sc.published("alert")]


async def gps(sc: Scenario, *, on: bool) -> None:
    await sc.harness(sc.sim.harness.set_sim_param, "SIM_GPS1_ENABLE", 1 if on else 0)


async def glitch(sc: Scenario, north_deg: float = 0.0, up_m: float = 0.0) -> None:
    h = sc.sim.harness
    await sc.harness(h.set_sim_param, "SIM_GPS1_GLTCH_X", north_deg)
    await sc.harness(h.set_sim_param, "SIM_GPS1_GLTCH_Z", up_m)


async def executor_state(sc: Scenario) -> str:
    return str((await sc.onboard_state())["executor_state"])


async def until_spoof(sc: Scenario, within_s: float, what: str) -> None:
    deadline = time.monotonic() + within_s
    while not await onboard_events(sc, "gnss_spoof_suspected"):
        assert time.monotonic() < deadline, what
        await asyncio.sleep(0.25)


async def until_state(sc: Scenario, want: str, within_s: float) -> None:
    deadline = time.monotonic() + within_s
    while (state := await executor_state(sc)) != want:
        assert time.monotonic() < deadline, f"executor {state}, not {want}"
        await asyncio.sleep(0.25)


# --- GPS lost -----------------------------------------------------------------------------


@pytest.mark.budget_s(330)
async def test_gp1_gps_lost(sim: Any, migrated_db: Any, tmp_path: Path) -> None:
    async with scenario(sim, migrated_db, tmp_path) as sc:
        await out_to_p1(sc)
        await gps(sc, on=False)
        off = time.monotonic()
        try:
            await sc.until(lambda: sc.fc_mode() in HOME_MODES, DR_RTL_BY_S, "dead reckoning RTL")
            assert fc_texts(sc, "Dead Reckoning started", off), sim.harness.state.texts[-10:]
            await until_state(sc, "FAILSAFE", 5.0)
            await asyncio.sleep(3.1)  # the ground's gate sees the new snapshot
            lat, lon = pin("P1")
            res = await sc.order("goto", {"lat": lat, "lon": lon, "alt_m": 40})
            assert res.get("decision") == "REJECT", res
            # FS_DR_TIMEOUT later the EKF failsafe (FS_EKF_ACTION 1) lands where it is
            await sc.until(lambda: sc.fc_mode() == "LAND", FS_DR_TIMEOUT_S + 15, "the EKF LAND")
            assert time.monotonic() - off >= FS_DR_TIMEOUT_S - 2.0
            await sc.until(lambda: not sim.harness.state.armed, 180, "landed")
            assert true_from_home_m(sc) <= 100.0, true_from_home_m(sc)
        finally:
            await gps(sc, on=True)
        codes = alert_codes(sc)
        assert "GPS_LOST" in codes and "DEAD_RECKONING" in codes, codes
        await sc.check_invariants(exempt=("INV2", "INV4"))


@pytest.mark.budget_s(300)
async def test_gp2_gps_back(sim: Any, migrated_db: Any, tmp_path: Path) -> None:
    async with scenario(sim, migrated_db, tmp_path) as sc:
        await out_to_p1(sc)
        await gps(sc, on=False)
        try:
            await sc.until(lambda: sc.fc_mode() in HOME_MODES, DR_RTL_BY_S, "dead reckoning RTL")
            await asyncio.sleep(max(0.0, 15.0 - DR_RTL_BY_S))
        finally:
            await gps(sc, on=True)
        await sc.until(lambda: not sim.harness.state.armed, 240, "landed")
        assert true_from_home_m(sc) <= 3.0, true_from_home_m(sc)
        assert "GPS_OK" in alert_codes(sc), alert_codes(sc)
        await sc.check_invariants(exempt=("INV2",))


# --- spoofing -----------------------------------------------------------------------------


async def spoofed_home(sc: Scenario, since: float, start_m: float) -> None:
    """After L-SPOOF: FAILSAFE, the EKF on source set 2, dead reckoning home."""
    events = await onboard_events(sc, "gnss_spoof_suspected")
    assert events and events[-1]["payload"]["source_set_ack"] == "ACCEPTED", events
    await until_state(sc, "FAILSAFE", 3.0)
    await sc.until(lambda: sc.fc_mode() in HOME_MODES, DR_RTL_BY_S, "home on dead reckoning")
    await asyncio.sleep(20.0)
    assert true_from_home_m(sc) < start_m - 40.0, (start_m, true_from_home_m(sc))
    assert "GPS_SPOOF" in alert_codes(sc), alert_codes(sc)
    assert time.monotonic() - since <= 60.0


@pytest.mark.budget_s(330)
async def test_gp3_spoof_jump(sim: Any, migrated_db: Any, tmp_path: Path) -> None:
    async with scenario(sim, migrated_db, tmp_path) as sc:
        await out_to_p1(sc)
        start = true_from_home_m(sc)
        await glitch(sc, north_deg=JUMP_DEG)
        jumped = time.monotonic()
        try:
            await until_spoof(sc, SPOOF_BY_S, "L-SPOOF (S1)")
            await spoofed_home(sc, jumped, start)
            await sc.until(lambda: not sim.harness.state.armed, 240, "landed")
        finally:
            await glitch(sc)
        found = (await onboard_events(sc, "gnss_spoof_suspected"))[-1]["payload"]
        assert found["kind"] == "jump", found
        await sc.check_invariants(exempt=("INV2", "INV4"))


@pytest.mark.budget_s(360)
async def test_gp4_spoof_slow_climb(sim: Any, migrated_db: Any, tmp_path: Path) -> None:
    """The spoofer drags the position north and the height up at 1 m/s: the EKF follows
    the position, but the GPS height leaves the barometer's (S2)."""
    async with scenario(sim, migrated_db, tmp_path) as sc:
        await out_to_p1(sc)
        start_m = true_from_home_m(sc)
        began = time.monotonic()
        try:
            while not await onboard_events(sc, "gnss_spoof_suspected"):
                t = time.monotonic() - began
                assert t <= RAMP_BY_S + 1.5, "no L-SPOOF from the slow climb"
                await glitch(sc, north_deg=t / M_PER_DEG, up_m=t)
                await asyncio.sleep(0.5)
            await spoofed_home(sc, began, start_m)
            await sc.until(lambda: not sim.harness.state.armed, 240, "landed")
        finally:
            await glitch(sc)
        found = (await onboard_events(sc, "gnss_spoof_suspected"))[-1]["payload"]
        assert found["kind"] == "altitude", found
        await sc.check_invariants(exempt=("INV2", "INV4"))


@pytest.mark.budget_s(270)
async def test_gp5_glitch_is_not_spoofing(sim: Any, migrated_db: Any, tmp_path: Path) -> None:
    async with scenario(sim, migrated_db, tmp_path) as sc:
        await out_to_p1(sc)
        try:
            await glitch(sc, north_deg=GLITCH_DEG)
            await asyncio.sleep(5.0)
        finally:
            await glitch(sc)
        await sc.hold_for(10.0, lambda: sc.fc_mode() not in HOME_MODES, "no failsafe")
        assert await executor_state(sc) == "HOLD"
        assert not await onboard_events(sc, "gnss_spoof_suspected")
        assert "GPS_SPOOF" not in alert_codes(sc), alert_codes(sc)
        assert (await sc.press("return_home")).status_code == 202
        await sc.until(lambda: not sim.harness.state.armed, 240, "landed")
        await sc.check_invariants()


@pytest.mark.budget_s(300)
async def test_gp6_strong_wind_no_alarm(sim: Any, migrated_db: Any, tmp_path: Path) -> None:
    async with scenario(sim, migrated_db, tmp_path) as sc:
        await out_to_p1(sc, wind_ms=8.0)
        await sc.hold_for(20.0, lambda: sc.fc_mode() not in HOME_MODES, "holds in the wind")
        assert (await sc.press("return_home")).status_code == 202
        await sc.until(lambda: not sim.harness.state.armed, 240, "landed")
        assert not [c for c in alert_codes(sc) if c in GPS_CODES], alert_codes(sc)
        assert not await onboard_events(sc, "gnss_spoof_suspected")
        assert not await onboard_events(sc, "gnss_degraded")
        await sc.check_invariants()


# --- the GPS and a find -------------------------------------------------------------------


@pytest.mark.budget_s(360)
async def test_gp7_gps_lost_in_find_hover(sim: Any, migrated_db: Any, tmp_path: Path) -> None:
    async with scenario(sim, migrated_db, tmp_path, proxied_link=True) as sc:
        await sc.airborne()
        await sc.harness(sim.harness.set_sim_param, "SIM_WIND_SPD", 4.0)
        try:
            await _find_hover(sc, mavlink=False, until_s=60.0)
            await gps(sc, on=False)
            await sc.until(lambda: sc.fc_mode() in HOME_MODES, DR_RTL_BY_S, "dead reckoning RTL")
            await sc.until(lambda: not sim.harness.state.armed, 240, "landed")
        finally:
            await gps(sc, on=True)
            await back_and_land(sc)
        await sc.until(
            lambda: bool(sc.published("executor.find_hold_ended")), 15, "the queued events"
        )
        ended = await onboard_events(sc, "find_hold_ended")
        assert ended and ended[-1]["payload"]["reason"] == "fc_failsafe", ended
        assert await onboard_events(sc, "fc_failsafe")
        await sc.check_invariants(exempt=("INV2", "INV4"))
