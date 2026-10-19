"""Link loss against ArduCopter SITL: the usual rule, the find hover and L-GCS.

The radio between the ground and the drone is cut with ``sc.radio`` (SD-link through the
proxy, the ground's MAVLink through linksim); the test rig's own channel to the flight
controller (serial0) stays up, so the flight mode and the texts are always seen. Times
count from the cut; the onboard computer counts from its last inbound message, which is
at most a second earlier.

* LL1/LL2/LL3/LL12: no find nearby. SD-link only: hold at 23 s, home at 53 s. The whole
  radio, moving: the flight controller's GCS failsafe at 20 s. The whole radio in HOLD
  (LOITER) or only MAVLink: L-GCS flies home at the flight controller's text.
* LL4-LL11: a confirmed find within 100 m: hover beside it from at the latest 15 s, home
  at 123 s; the battery, the RC switch and the link coming back win; the Lua script
  flies home at 150 s when the onboard computer died.
"""

from __future__ import annotations

import asyncio
import json
import math
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
    sim_stack,
)

from sd_safety.constants import (
    FC_LOITER_CAP_S,
    FIND_HOVER_EDGE_MARGIN_M,
    FIND_HOVER_S,
    FIND_MOVE_DEADLINE_S,
    LINK_LOSS_DETECT_S,
    PERSON_KEEPOUT_M,
    SDLINK_GRACE_S,
    SDLINK_HOLD_S,
)
from sd_safety.targets import Plane, distance_m
from sd_sim.params import merge_layers
from sd_sim.sitl import SAFETY_PARAM_FILE

pytestmark = [pytest.mark.sitl, pytest.mark.slow]
__all__ = ["fresh_db", "migrated_db", "passwords", "sim", "sim_stack"]

PARAMS = Path(__file__).resolve().parents[2] / "sitl/params"
SAFETY = merge_layers([PARAMS / SAFETY_PARAM_FILE])
HOME_MODES = ("SMART_RTL", "RTL")
SLACK_S = 1.5
FS_SLACK_S = 2.0
HOLD_AT = LINK_LOSS_DETECT_S + SDLINK_GRACE_S  # 23 s
HOME_AT = HOLD_AT + SDLINK_HOLD_S  # 53 s
FIND_LOITER_BY = LINK_LOSS_DETECT_S + FIND_MOVE_DEADLINE_S  # 15 s
FIND_HOME_AT = LINK_LOSS_DETECT_S + FIND_HOVER_S  # 123 s
SIGMA_M = 5.0
RADIUS_M = PERSON_KEEPOUT_M + SIGMA_M
CONTINUING = "Continuing Pilot Control"


# --- helpers ------------------------------------------------------------------------------


async def heading_north(sc: Scenario, speed_ms: float = 4.0) -> None:
    """Toward P1 (200 m north of home), slowly enough to cut the link where wanted."""
    lat, lon = pin("P1")
    res = await sc.order("goto", {"lat": lat, "lon": lon, "alt_m": 40, "speed_ms": speed_ms})
    assert res.get("decision") == "ALLOW", res
    await sc.wait_command("goto", "running", 15)


def north_m(sc: Scenario) -> float:
    return Plane(*sc.r.home).xy(sc.tel.lat, sc.tel.lon)[1]


async def find_ahead(sc: Scenario, find_north_m: float = 150.0, cut_at_m: float = 90.0) -> Any:
    """On the way north a person's find is confirmed ahead; once the drone is
    ``find_north_m - cut_at_m`` from it, the function returns (the caller cuts the radio).
    Returns (find lat, lon, the hover point lat, lon)."""
    await heading_north(sc)
    flat, flon = sc.at(0.0, find_north_m)
    await sc.confirm_find(flat, flon, SIGMA_M)
    await sc.until(lambda: north_m(sc) >= cut_at_m, 90, "near the find")
    hlat, hlon = sc.at(0.0, find_north_m - RADIUS_M - FIND_HOVER_EDGE_MARGIN_M)
    return flat, flon, hlat, hlon


def closest_to(sc: Scenario, lat: float, lon: float, since: float) -> float:
    """The closest the drone came to (lat, lon) since ``since`` (the recorded track)."""
    return min(
        (distance_m(lat, lon, s.lat, s.lon) for s in sc.track if s.t >= since and s.armed),
        default=math.inf,
    )


def bearing_deg(lat1: float, lon1: float, lat2: float, lon2: float) -> float:
    plane = Plane(lat1, lon1)
    x, y = plane.xy(lat2, lon2)
    return math.degrees(math.atan2(x, y)) % 360.0


def angle_off(a: float, b: float) -> float:
    return abs((a - b + 180.0) % 360.0 - 180.0)


def fc_texts(sc: Scenario, needle: str, since: float) -> list[float]:
    return [t for t, text in sc.sim.harness.state.texts if t >= since and needle in text]


async def onboard_events(sc: Scenario, kind: str) -> list[dict[str, Any]]:
    return [e for e in await sc.events() if e["type"] == f"executor.{kind}"]


async def back_and_land(sc: Scenario) -> None:
    sc.radio(mavlink=True, sdlink=True)
    await sc.until(lambda: sc.r.link.up(VID), 20, "SD-link back")


# --- the usual rule -----------------------------------------------------------------------


@pytest.mark.budget_s(330)
async def test_ll1_sdlink_lost_in_survey(sim: Any, migrated_db: Any, tmp_path: Path) -> None:
    async with scenario(sim, migrated_db, tmp_path, proxied_link=True) as sc:
        await sc.airborne()
        res = await sc.order("survey_sector", {"sector_id": "A"})
        assert res.get("decision") == "ALLOW", res
        await sc.wait_command("survey_sector", "running", 60)
        sc.radio(sdlink=False)
        down = time.monotonic()
        try:
            await sc.until(lambda: sc.fc_mode() == "LOITER", HOLD_AT + SLACK_S + 2, "hold")
            assert HOLD_AT - SLACK_S <= time.monotonic() - down <= HOLD_AT + SLACK_S
            await sc.until(lambda: sc.fc_mode() in HOME_MODES, HOME_AT, "home")
            assert HOME_AT - SLACK_S <= time.monotonic() - down <= HOME_AT + SLACK_S
            assert sc.fc_mode() == "SMART_RTL"
            await sc.until(lambda: not sim.harness.state.armed, 240, "landed")
        finally:
            sc.radio(sdlink=True)
        assert sc.from_home_m() <= 3.0
        await sc.check_invariants()


@pytest.mark.budget_s(300)
async def test_ll2_all_lost_in_survey(sim: Any, migrated_db: Any, tmp_path: Path) -> None:
    async with scenario(sim, migrated_db, tmp_path, proxied_link=True) as sc:
        await sc.airborne()
        res = await sc.order("survey_sector", {"sector_id": "A"})
        assert res.get("decision") == "ALLOW", res
        await sc.wait_command("survey_sector", "running", 60)
        sc.radio(mavlink=False, sdlink=False)
        down = time.monotonic()
        try:
            limit = SAFETY["FS_GCS_TIMEOUT"] + FS_SLACK_S + SLACK_S
            await sc.until(lambda: sc.fc_mode() in HOME_MODES, limit, "the GCS failsafe")
            assert time.monotonic() - down >= SAFETY["FS_GCS_TIMEOUT"] - SLACK_S
            await sc.until(lambda: not sim.harness.state.armed, 240, "landed")
        finally:
            sc.radio(mavlink=True, sdlink=True)
        assert sc.from_home_m() <= 3.0
        await sc.check_invariants()


async def _gcs_lost_in_hold(sc: Scenario, *, sdlink: bool) -> None:
    """Hovering in HOLD (LOITER); the ground's MAVLink goes (and SD-link unless
    ``sdlink``): the flight controller keeps LOITER (FS_OPTIONS bit 4) and says so, and
    L-GCS flies home at once (case C, case D)."""
    await sc.airborne()
    res = await sc.order("hold", {})
    assert res.get("decision") == "ALLOW", res
    await sc.until(lambda: sc.fc_mode() == "LOITER", 15, "holding")
    sc.radio(mavlink=False, sdlink=sdlink)
    down = time.monotonic()
    try:
        limit = SAFETY["FS_GCS_TIMEOUT"] + SLACK_S
        await sc.until(lambda: sc.fc_mode() in HOME_MODES, limit + 1.0, "L-GCS home")
        took = time.monotonic() - down
        assert took <= limit, took
        assert took >= SAFETY["FS_GCS_TIMEOUT"] - SLACK_S, took
        assert fc_texts(sc, CONTINUING, down), sc.sim.harness.state.texts[-10:]
        await sc.until(lambda: not sc.sim.harness.state.armed, 240, "landed")
    finally:
        sc.radio(mavlink=True, sdlink=True)
    await sc.until(lambda: sc.r.link.up(VID), 20, "SD-link back")
    assert await onboard_events(sc, "gcs_lost_rtl")
    assert sc.from_home_m() <= 3.0
    await sc.check_invariants()


@pytest.mark.budget_s(300)
async def test_ll3_all_lost_in_hold(sim: Any, migrated_db: Any, tmp_path: Path) -> None:
    async with scenario(sim, migrated_db, tmp_path, proxied_link=True) as sc:
        await _gcs_lost_in_hold(sc, sdlink=False)


@pytest.mark.budget_s(300)
async def test_ll12_mavlink_only_lost(sim: Any, migrated_db: Any, tmp_path: Path) -> None:
    async with scenario(sim, migrated_db, tmp_path, proxied_link=True) as sc:
        await _gcs_lost_in_hold(sc, sdlink=True)


# --- the find hover -----------------------------------------------------------------------


async def _find_hover(
    sc: Scenario, *, mavlink: bool, until_s: float = FIND_HOME_AT
) -> tuple[float, tuple[float, float, float, float]]:
    """A find ahead, the radio cut: LOITER by 15 s, at the hover point facing the find,
    never inside its cylinder, and still there at ``until_s``. Returns the cut time and
    the points."""
    points = await find_ahead(sc)
    flat, flon, hlat, hlon = points
    sc.radio(mavlink=mavlink, sdlink=False)
    down = time.monotonic()
    await sc.until(lambda: sc.fc_mode() == "LOITER", FIND_LOITER_BY + 0.5, "find hover")
    assert time.monotonic() - down <= FIND_LOITER_BY + 0.5
    await sc.until(
        lambda: distance_m(hlat, hlon, sc.tel.lat, sc.tel.lon) <= 3.0, 20, "at the hover point"
    )
    heading = sc.sim.harness.state.heading_deg
    assert angle_off(heading, bearing_deg(sc.tel.lat, sc.tel.lon, flat, flon)) <= 15.0, heading
    await sc.hold_for(
        max(0.0, until_s - SLACK_S - (time.monotonic() - down)),
        lambda: sc.fc_mode() == "LOITER",
        "hovers beside the find",
    )
    assert closest_to(sc, flat, flon, down) >= RADIUS_M
    return down, points


@pytest.mark.budget_s(420)
async def test_ll4_find_sdlink_lost(sim: Any, migrated_db: Any, tmp_path: Path) -> None:
    async with scenario(sim, migrated_db, tmp_path, proxied_link=True) as sc:
        await sc.airborne()
        try:
            down, (flat, flon, hlat, hlon) = await _find_hover(sc, mavlink=True)
            await sc.until(lambda: sc.fc_mode() in HOME_MODES, 2 * SLACK_S + 1, "home")
            took = time.monotonic() - down
            assert FIND_HOME_AT - SLACK_S <= took <= FIND_HOME_AT + SLACK_S, took
            await sc.until(lambda: not sim.harness.state.armed, 240, "landed")
            assert closest_to(sc, flat, flon, down) >= RADIUS_M
        finally:
            await back_and_land(sc)
        assert sc.from_home_m() <= 3.0
        await sc.until(
            lambda: _has(sc, "executor.find_hold_ended"), 15, "the queued events after the link"
        )
        started = await onboard_events(sc, "find_hold_started")
        ended = await onboard_events(sc, "find_hold_ended")
        assert started and ended, (started, ended)
        assert ended[-1]["payload"]["reason"] == "timeout"
        keepout = started[-1]["payload"]["keepout_id"]
        lines = [
            json.loads(line)
            for line in sc.r.onboard.ex.store.find_holds_path.read_text().splitlines()
        ]
        done = [x for x in lines if x.get("record") == "ended"]
        assert done and done[-1]["keepout_id"] == keepout
        assert distance_m(hlat, hlon, done[-1]["hover_lat"], done[-1]["hover_lon"]) <= 1.0
        await sc.check_invariants()


def _has(sc: Scenario, type_: str) -> bool:
    return bool(sc.published(type_))


@pytest.mark.budget_s(420)
async def test_ll5_find_all_lost(sim: Any, migrated_db: Any, tmp_path: Path) -> None:
    async with scenario(sim, migrated_db, tmp_path, proxied_link=True) as sc:
        await sc.airborne()
        try:
            down, _ = await _find_hover(sc, mavlink=False)
            assert fc_texts(sc, CONTINUING, down), sim.harness.state.texts[-10:]
            await sc.until(lambda: sc.fc_mode() in HOME_MODES, 2 * SLACK_S + 1, "home")
            took = time.monotonic() - down
            assert FIND_HOME_AT - SLACK_S <= took <= FIND_HOME_AT + SLACK_S, took
            await sc.until(lambda: not sim.harness.state.armed, 240, "landed")
        finally:
            await back_and_land(sc)
        assert sc.from_home_m() <= 3.0
        await sc.check_invariants()


@pytest.mark.budget_s(360)
async def test_ll6_find_battery_wins(sim: Any, migrated_db: Any, tmp_path: Path) -> None:
    async with scenario(sim, migrated_db, tmp_path, proxied_link=True) as sc:
        await sc.airborne("charter_batt25")
        try:
            down, _ = await _find_hover(sc, mavlink=True, until_s=40.0)
            await sc.harness(sim.harness.battery_set_remaining, 26)
            await sc.until(lambda: sc.fc_mode() in HOME_MODES, 60, "the battery rule")
            assert time.monotonic() - down < FIND_HOME_AT - SLACK_S
            await sc.until(lambda: not sim.harness.state.armed, 240, "landed")
        finally:
            await back_and_land(sc)
        await sc.until(lambda: _has(sc, "executor.find_hold_ended"), 15, "queued events")
        ended = await onboard_events(sc, "find_hold_ended")
        assert ended and ended[-1]["payload"]["reason"] == "battery", ended
        await sc.check_invariants()


@pytest.mark.budget_s(360)
async def test_ll7_find_rc_switch(sim: Any, migrated_db: Any, tmp_path: Path) -> None:
    """a: the pilot flicks the switch to RTL during the hover: the flight controller
    flies home at once, the onboard computer lets go (HELD) and sends nothing more."""
    async with scenario(sim, migrated_db, tmp_path, proxied_link=True) as sc:
        await sc.airborne()
        try:
            down, _ = await _find_hover(sc, mavlink=False, until_s=40.0)
            await sc.rc_switch("RTL")
            await sc.until(lambda: sc.fc_mode() == "RTL", 3, "RTL from the switch")
            ex = sc.r.onboard.ex
            await sc.until(lambda: str(ex.state) == "HELD", 5, "the onboard computer lets go")
            await sc.hold_for(10.0, lambda: sc.fc_mode() == "RTL", "RTL stays")
            assert time.monotonic() - down < FIND_HOME_AT
            await sc.until(lambda: not sim.harness.state.armed, 240, "landed")
        finally:
            await back_and_land(sc)
        await sc.check_invariants()


@pytest.mark.budget_s(420)
async def test_ll7b_find_rc_switch_loiter(sim: Any, migrated_db: Any, tmp_path: Path) -> None:
    """b: the pilot puts the switch on LOITER: the pilot steers; neither the onboard
    computer nor the Lua script flies home, also past 150 s."""
    async with scenario(sim, migrated_db, tmp_path, proxied_link=True) as sc:
        await sc.airborne()
        try:
            down, _ = await _find_hover(sc, mavlink=False, until_s=40.0)
            await sc.rc_switch("LOITER")
            await sc.until(lambda: sc.fc_mode() == "LOITER", 5, "LOITER from the switch")
            await asyncio.sleep(1.0)  # through RTL's position: LOITER again
            await sc.hold_for(
                max(0.0, 170.0 - (time.monotonic() - down)),
                lambda: sc.fc_mode() == "LOITER",
                "the pilot's LOITER stays",
            )
            assert not fc_texts(sc, "SD failsafe", down)
        finally:
            await back_and_land(sc)
        assert (await sc.press("return_home")).status_code == 202
        await sc.until(lambda: not sim.harness.state.armed, 240, "landed")
        await sc.check_invariants(exempt=("INV2",))


@pytest.mark.budget_s(330)
async def test_ll8_find_too_far(sim: Any, migrated_db: Any, tmp_path: Path) -> None:
    async with scenario(sim, migrated_db, tmp_path, proxied_link=True) as sc:
        await sc.airborne()
        await heading_north(sc, 3.0)
        flat, flon = sc.at(140.0, 250.0)  # ~300 m from home, ~250 m from the cut
        await sc.confirm_find(flat, flon, SIGMA_M)
        await sc.until(lambda: north_m(sc) >= 30.0, 60, "on the way")
        sc.radio(sdlink=False)
        down = time.monotonic()
        try:
            await sc.until(lambda: sc.fc_mode() == "LOITER", HOLD_AT + SLACK_S + 2, "hold")
            assert HOLD_AT - SLACK_S <= time.monotonic() - down <= HOLD_AT + SLACK_S
            await sc.until(lambda: sc.fc_mode() in HOME_MODES, HOME_AT, "home")
            assert HOME_AT - SLACK_S <= time.monotonic() - down <= HOME_AT + SLACK_S
            await sc.until(lambda: not sim.harness.state.armed, 240, "landed")
        finally:
            await back_and_land(sc)
        assert not await onboard_events(sc, "find_hold_started")
        await sc.check_invariants()


@pytest.mark.budget_s(330)
async def test_ll9_find_link_back(sim: Any, migrated_db: Any, tmp_path: Path) -> None:
    async with scenario(sim, migrated_db, tmp_path, proxied_link=True) as sc:
        await sc.airborne()
        try:
            down, _ = await _find_hover(sc, mavlink=True, until_s=60.0)
        finally:
            sc.radio(sdlink=True)
        await sc.until(lambda: sc.r.link.up(VID), 20, "SD-link back")
        await sc.hold_for(
            max(0.0, 140.0 - (time.monotonic() - down)),
            lambda: sc.fc_mode() == "LOITER",
            "still in HOLD after 123 s",
        )
        live = await sc.onboard_state()
        assert (live["executor_state"], live["hold_reason"]) == ("HOLD", "find_hold"), live
        assert [a for a in sc.published("alert") if a.get("code") == "LINK_BACK_HOLD"]
        assert (await sc.press("return_home")).status_code == 202
        await sc.until(lambda: sc.fc_mode() in HOME_MODES, 5, "FLY HOME works")
        await sc.until(lambda: not sim.harness.state.armed, 240, "landed")
        ended = await onboard_events(sc, "find_hold_ended")
        assert ended and ended[-1]["payload"]["reason"] == "link_restored", ended
        await sc.check_invariants()


@pytest.mark.budget_s(420)
async def test_ll10_find_onboard_dies(sim: Any, migrated_db: Any, tmp_path: Path) -> None:
    async with scenario(sim, migrated_db, tmp_path, proxied_link=True) as sc:
        await sc.airborne()
        try:
            down, _ = await _find_hover(sc, mavlink=False, until_s=30.0)
            await sc.kill_onboard()
            cap_at = FC_LOITER_CAP_S  # from the last ground heartbeat, the cut
            await sc.hold_for(
                max(0.0, cap_at - 2.0 - (time.monotonic() - down)),
                lambda: sc.fc_mode() == "LOITER",
                "LOITER until the Lua cap",
            )
            await sc.until(lambda: sc.fc_mode() in HOME_MODES, 6.0, "the Lua cap")
            took = time.monotonic() - down
            assert cap_at - 2.0 <= took <= cap_at + 2.0, took
            assert fc_texts(sc, "SD failsafe", down), sim.harness.state.texts[-10:]
            await sc.until(lambda: not sim.harness.state.armed, 240, "landed")
        finally:
            sc.radio(mavlink=True, sdlink=True)
        assert sc.from_home_m() <= 3.0
        await sc.check_invariants(exempt=("INV2",))


@pytest.mark.budget_s(330)
async def test_ll11_find_move_deadline(sim: Any, migrated_db: Any, tmp_path: Path) -> None:
    """A find 100 m away (the hover point some 60 m away) against a strong head wind:
    LOITER by 15.5 s whether the point was reached or not, and no GCS failsafe at 20 s
    with the whole radio down."""
    async with scenario(sim, migrated_db, tmp_path, proxied_link=True) as sc:
        await sc.airborne()
        await heading_north(sc, 3.0)
        flat, flon = sc.at(0.0, 145.0)
        await sc.confirm_find(flat, flon, SIGMA_M)
        await sc.until(lambda: north_m(sc) >= 47.0, 60, "100 m from the find")
        await sc.harness(sim.harness.set_sim_param, "SIM_WIND_DIR", 0)  # from the north
        await sc.harness(sim.harness.set_sim_param, "SIM_WIND_SPD", 8)
        sc.radio(mavlink=False, sdlink=False)
        down = time.monotonic()
        try:
            await sc.until(lambda: sc.fc_mode() == "LOITER", FIND_LOITER_BY + 0.5, "deadline")
            await sc.hold_for(
                max(0.0, SAFETY["FS_GCS_TIMEOUT"] + 5.0 - (time.monotonic() - down)),
                lambda: sc.fc_mode() == "LOITER",
                "no GCS failsafe",
            )
            assert closest_to(sc, flat, flon, down) >= RADIUS_M
        finally:
            sc.radio(mavlink=True, sdlink=True)
            await sc.harness(sim.harness.set_sim_param, "SIM_WIND_SPD", 0)
        await sc.until(lambda: sc.r.link.up(VID), 20, "SD-link back")
        assert (await sc.press("return_home")).status_code == 202
        await sc.until(lambda: not sim.harness.state.armed, 240, "landed")
        await sc.check_invariants()
