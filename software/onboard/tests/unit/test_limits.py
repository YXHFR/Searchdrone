"""The limit monitors, one by one, and in the air on the fake."""

from __future__ import annotations

import pytest
from onboard_helpers import HOME_ALT, Rig, at, charter, rig

from sd_mav.modes import CopterMode
from sd_onboard.executor.limits import LimitInput, LimitMonitor, PilotTracker
from sd_safety.constants import LINK_LOSS_DETECT_S, SDLINK_GRACE_S, SDLINK_HOLD_S
from sd_schemas.base import LatLon
from sd_schemas.executor import ExecutorState as S
from sd_schemas.link import CommanderStatus, PilotPosition
from sd_schemas.vehicle import VehicleState

C = charter(alt_max_agl_m=100, max_speed_ms=12, battery_rtl_pct=20, ai_silent_timeout_s=10)
PILOT = C.pilot_position


def vs(**kw: object) -> VehicleState:
    lat, lon = at(0, 0)
    data: dict[str, object] = {
        "vehicle_id": "sim-01",
        "mono_ms": 0,
        "fc_connected": True,
        "gcs_mavlink_seen_mono_ms": None,
        "armed": True,
        "fc_mode": "GUIDED",
        "fc_system_status": 4,
        "landed_state": "in_air",
        "lat": lat,
        "lon": lon,
        "alt_rel_home_m": 40.0,
        "alt_amsl_m": HOME_ALT + 40,
        "groundspeed_ms": 5.0,
        "climb_ms": 0.0,
        "heading_deg": 0.0,
        "battery_pct": 80.0,
        "battery_charge_state": 1,
        "battery_voltage_v": 16.0,
        "gps_fix_type": 3,
        "gps_satellites": 12,
        "ekf_ok": True,
        "home": None,
        "mission_seq_current": None,
        "pilot_position": PILOT,
        "pilot_position_source": "charter",
    }
    data.update(kw)
    return VehicleState.model_validate(data)


def li(now_s: float, v: VehicleState, **kw: object) -> LimitInput:
    data: dict[str, object] = {
        "now_ms": round(now_s * 1000),
        "state": S.TRANSIT,
        "vehicle": v,
        "charter": C,
        "pilot": PILOT,
        "active_skill": "goto",
        "inbound_order_active": False,
        "link_last_inbound_ms": round(now_s * 1000),
        "link_up": True,
        "commander_connected": True,
        "commander_status_ms": round(now_s * 1000),
    }
    data.update(kw)
    return LimitInput(**data)  # type: ignore[arg-type]


def run(
    m: LimitMonitor, seconds: float, v: VehicleState, start: float = 0.0, **kw: object
) -> list[str]:
    fired: list[str] = []
    t = start
    while t <= start + seconds + 1e-9:
        fired += [f"{a.limit}:{a.kind}" for a in m.check(li(t, v, **kw))]
        t = round(t + 0.1, 3)
    return fired


def test_limit_monitor_battery() -> None:
    m = LimitMonitor()
    assert run(m, 5, vs(battery_pct=20.0)) == ["L-BATT:rtl"]
    assert m.battery_rtl_active


def test_limit_monitor_range_hard() -> None:
    lat, lon = at(560, 0)
    assert run(LimitMonitor(), 1, vs(lat=lat, lon=lon)) == ["L-RANGE2:rtl"]


def test_limit_monitor_area_needs_2s() -> None:
    lat, lon = at(230, 0)  # 30 m outside the area, 10 m beyond its 20 m buffer
    m = LimitMonitor()
    assert run(m, 1.8, vs(lat=lat, lon=lon)) == []
    assert run(m, 0.5, vs(lat=lat, lon=lon), start=1.9) == ["L-AREA:rtl"]
    lat, lon = at(215, 0)  # inside the buffer
    assert run(LimitMonitor(), 5, vs(lat=lat, lon=lon)) == []


def test_limit_monitor_nav() -> None:
    m = LimitMonitor()
    assert run(m, 3, vs(gps_fix_type=2)) == ["L-GPS:hold"]
    assert m.nav_degraded(3000)
    run(m, 3, vs(), start=3.1)
    assert m.nav_degraded(6200)  # good for less than 5 s
    assert not m.nav_degraded(11_200)


def test_limit_battery_unknown_rtl() -> None:
    m = LimitMonitor()
    assert run(m, 4.8, vs(battery_pct=None)) == []
    assert run(m, 0.5, vs(battery_pct=None), start=4.9) == ["L-BATT0:hold"]
    assert run(m, 29.0, vs(battery_pct=None), start=5.5) == []
    assert run(m, 1.0, vs(battery_pct=None), start=34.6) == ["L-BATT0:rtl"]


def test_limit_alt_single_trigger() -> None:
    m = LimitMonitor()
    assert run(m, 5, vs(alt_rel_home_m=103.0)) == ["L-ALT:alt"]
    assert run(m, 5, vs(alt_rel_home_m=103.0), start=5.1, active_skill="internal_alt_goto") == []


def test_limit_monitor_speed() -> None:
    m = LimitMonitor()
    assert run(m, 2.8, vs(groundspeed_ms=14.5)) == []
    assert run(m, 0.5, vs(groundspeed_ms=14.5), start=2.9) == ["L-SPEED:hold"]


def test_limit_edge_events_once() -> None:
    m = LimitMonitor()
    bad = vs(gps_fix_type=1, groundspeed_ms=15.0, battery_pct=None)
    fired = run(m, 20, bad)
    assert sorted(fired) == ["L-BATT0:hold", "L-GPS:hold", "L-SPEED:hold"]


def test_limit_range_allows_inbound_goto() -> None:
    """Pilot 300 m south of home: 210 m north of home is 510 m from the pilot."""
    pilot = LatLon(lat=at(0, -300)[0], lon=at(0, -300)[1])
    m = LimitMonitor()

    def at_north(n: float) -> VehicleState:
        lat, lon = at(0, n)
        return vs(lat=lat, lon=lon, pilot_position=pilot)

    assert [a.limit for a in m.check(li(0, at_north(210), pilot=pilot))] == ["L-RANGE"]
    assert m.range_active
    fired = []
    for i in range(50):  # flying back in at 2 m/s
        fired += [
            a.limit
            for a in m.check(
                li(0.1 + 0.1 * i, at_north(210 - 0.2 * i), pilot=pilot, inbound_order_active=True)
            )
        ]
    assert fired == []
    assert m.range_active  # inside again, but not yet for LIMIT_CLEAR_S
    for i in range(40):  # and out again
        fired += [
            a.limit
            for a in m.check(
                li(5.1 + 0.1 * i, at_north(200 + 0.3 * i), pilot=pilot, inbound_order_active=True)
            )
        ]
    assert fired == ["L-RANGE"]


def test_first_rtl_wins() -> None:
    lat, lon = at(560, 0)
    m = LimitMonitor()
    fired = [a.limit for a in m.check(li(0, vs(lat=lat, lon=lon, battery_pct=10.0)))]
    assert fired == ["L-BATT"]  # RANGE2 after it is dropped in the same tick


def test_ai_silent_modes() -> None:
    for mode, kind in (
        ("hold", "hold"),
        ("continue_plan", "after_step"),
        ("return_home", "after_step"),
    ):
        c = C.model_copy(update={"on_ai_silent": mode})
        m = LimitMonitor()
        fired = []
        for i in range(120):
            fired += [a.kind for a in m.check(li(i * 0.1, vs(), charter=c, commander_status_ms=0))]
        assert fired == [kind], mode


def test_ai_silent_suspended_during_link_loss() -> None:
    c = C.model_copy(update={"on_ai_silent": "hold"})
    m = LimitMonitor()
    fired = []
    for i in range(150):  # the link is down for 15 s, the AI silent too
        fired += [
            a.limit
            for a in m.check(
                li(
                    i * 0.1,
                    vs(),
                    charter=c,
                    link_up=False,
                    link_last_inbound_ms=0,
                    commander_status_ms=0,
                )
            )
        ]
    assert "L-AI" not in fired
    for i in range(95):  # the link is back: the timer starts again
        fired += [
            a.limit for a in m.check(li(15 + i * 0.1, vs(), charter=c, commander_status_ms=0))
        ]
    assert "L-AI" not in fired
    for i in range(10):
        fired += [
            a.limit for a in m.check(li(24.6 + i * 0.1, vs(), charter=c, commander_status_ms=0))
        ]
    assert fired.count("L-AI") == 1


def test_pilot_position_stale() -> None:
    p = PilotTracker()
    lat, lon = at(10, 0)
    assert p.update(lat, lon, 0)
    pos, stale = p.effective(C, 4000)
    assert pos == LatLon(lat=lat, lon=lon) and not stale
    pos, stale = p.effective(C, 5100)
    assert pos == C.pilot_position and stale
    assert p.effective(C, 6000) == (C.pilot_position, False)  # reported once


def test_pilot_position_jump_ignored() -> None:
    p = PilotTracker()
    a = at(0, 0)
    b = at(0, 100)
    assert p.update(*a, 0)
    assert not p.update(*b, 1000)  # 100 m/s
    assert p.update(*b, 3000)  # 33 m/s


# --- in the air ----------------------------------------------------------------------------


async def _hovering(r: Rig, **charter_kw: object) -> None:
    await r.install(charter(**charter_kw))
    await r.takeoff(40)


async def test_battery_rtl_in_flight() -> None:
    async with rig() as r:
        await _hovering(r)
        r.fc.battery_pct = 15
        await r.run(2)
        assert r.ex.state == S.RTL
        assert r.fc.mode in (CopterMode.SMART_RTL, CopterMode.RTL)
        lat, lon = at(20, 0)
        cmd = await r.send(
            r.order("goto", {"lat": lat, "lon": lon, "alt_m": 40}, issuer="brain:ground")
        )
        assert r.up.codes(cmd.command_id) == ("ONB_RTL_ACTIVE",)
        await r.until(lambda: r.ex.state == S.LANDED, 200, step=0.05)


async def test_altitude_overshoot_goes_back_down() -> None:
    async with rig() as r:
        await _hovering(r, alt_max_agl_m=50, default_alt_agl_m=40)
        r.fc.alt_rel = 55.0  # a gust pushed it up while it hovers in LOITER
        await r.until(lambda: r.ex.state == S.TRANSIT, 20)
        assert r.ex.active is not None and r.ex.active.skill == "internal_alt_goto"
        await r.until(lambda: r.ex.state == S.HOLD, 60)
        assert r.ex.hold_reason == "limit"
        assert abs(r.fc.alt_rel - 45) < 2
        assert r.up.kinds().count("executor.limit_altitude") == 1


async def test_altitude_overshoot_with_the_ai_silent_ends_in_the_limit_hover() -> None:
    """The way back down ends in the altitude limit's hover (``limit``) even when the AI
    has gone quiet meanwhile (finish the step, then hover), as the keep-out climb does."""
    async with rig() as r:
        await _hovering(r, alt_max_agl_m=50, default_alt_agl_m=40)
        r.fc.alt_rel = 55.0
        await r.until(lambda: r.ex.state == S.TRANSIT, 20)

        def held_with_the_ai_silent() -> bool:  # the rig's AI keeps talking: silence it
            r.ex.ai_silent_after_step = True
            return r.ex.state == S.HOLD

        await r.until(held_with_the_ai_silent, 60)
        assert r.ex.hold_reason == "limit"


@pytest.mark.parametrize("seconds", [10, 60])
async def test_sdlink_loss(seconds: int) -> None:
    async with rig() as r:
        await _hovering(r)
        lat, lon = at(150, 100)
        await r.send(r.order("goto", {"lat": lat, "lon": lon, "alt_m": 40, "speed_ms": 2}))
        cut = r.clock.monotonic()
        r.link = False
        changes: list[tuple[float, S]] = []
        last = r.ex.state
        end = cut + seconds
        while r.clock.monotonic() < end:
            await r.run(0.1)
            if r.ex.state != last:
                last = r.ex.state
                changes.append((r.clock.monotonic() - cut, last))
        r.link = True
        await r.run(2)
        hold_at = LINK_LOSS_DETECT_S + SDLINK_GRACE_S
        rtl_at = hold_at + SDLINK_HOLD_S
        if seconds == 10:
            assert changes == []  # nothing visible happens for a short dropout
            assert r.ex.state == S.TRANSIT
        else:
            states = [s for _, s in changes]
            assert states[:2] == [S.HOLD, S.RTL], changes
            assert abs(changes[0][0] - hold_at) <= 1 and abs(changes[1][0] - rtl_at) <= 1
            assert r.ex.hold_reason is None and r.ex.state == S.RTL
            await r.until(lambda: r.ex.state == S.LANDED, 300, step=0.05)
            assert r.fc.distance_from_home_m() <= 3


async def test_link_back_before_home_keeps_hold() -> None:
    async with rig() as r:
        await _hovering(r)
        r.link = False
        await r.run(LINK_LOSS_DETECT_S + SDLINK_GRACE_S + 2)
        assert r.ex.state == S.HOLD and r.ex.hold_reason == "link_loss"
        r.link = True
        await r.run(SDLINK_HOLD_S + 5)
        assert r.ex.state == S.HOLD and r.ex.hold_reason == "link_loss"


async def test_ai_silent_hold() -> None:
    async with rig() as r:
        await _hovering(r, on_ai_silent="hold", ai_silent_timeout_s=5)
        lat, lon = at(150, 0)
        await r.send(r.order("goto", {"lat": lat, "lon": lon, "alt_m": 40, "speed_ms": 3}))
        r.commander = False
        r.ex.on_commander_status(CommanderStatus(connected=False, state="down"))
        await r.run(1)
        assert r.ex.state == S.HOLD and r.ex.hold_reason == "ai_silent"


async def _ai_quiet_during_goto(r: Rig) -> None:
    lat, lon = at(60, 0)
    goto = await r.send(r.order("goto", {"lat": lat, "lon": lon, "alt_m": 40, "speed_ms": 5}))
    r.commander = False
    r.ex.on_commander_status(CommanderStatus(connected=False, state="down"))
    await r.run(2)
    assert r.ex.state == S.TRANSIT  # the step goes on
    await r.until(lambda: r.ex.state == S.HOLD, 60)
    assert r.up.stages(goto.command_id)[-1] == "done"
    assert r.ex.hold_reason == "ai_silent"


async def test_ai_silent_continue_plan() -> None:
    async with rig() as r:
        await _hovering(r, on_ai_silent="continue_plan", ai_silent_timeout_s=5)
        await _ai_quiet_during_goto(r)


async def test_ai_silent_return_home() -> None:
    """The AI going quiet never sends the vehicle home: the step finishes, then it holds."""
    async with rig() as r:
        await _hovering(r, on_ai_silent="return_home", ai_silent_timeout_s=5)
        await _ai_quiet_during_goto(r)
        await r.run(30)
        assert r.ex.state == S.HOLD
        assert r.fc.mode == CopterMode.LOITER


async def test_pilot_position_from_the_ground_is_used() -> None:
    async with rig() as r:
        await _hovering(r)
        lat, lon = at(30, 0)
        r.ex.on_pilot_position(PilotPosition(lat=lat, lon=lon, source="phone", accuracy_m=3))
        assert r.ex.vehicle_state().pilot_position == LatLon(lat=lat, lon=lon)
        await r.run(6)
        assert r.ex.vehicle_state().pilot_position == r.charter.pilot_position  # type: ignore[union-attr]
        assert r.up.kinds().count("executor.pilot_position_stale") == 1


async def test_ai_back_clears_the_silent_label() -> None:
    async with rig() as r:
        await _hovering(r, on_ai_silent="continue_plan", ai_silent_timeout_s=5)
        await _ai_quiet_during_goto(r)
        r.commander = True
        await r.run(5)
        lat, lon = at(0, 30)
        goto = await r.send(r.order("goto", {"lat": lat, "lon": lon, "alt_m": 40, "speed_ms": 5}))
        await r.until(lambda: r.up.stages(goto.command_id)[-1] == "done", 60)
        assert r.ex.hold_reason == "skill_done"
