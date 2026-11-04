"""The companion watchdog (04 T15c, T35; 03 §8.3) and the battery guard (04 T15b)."""

from __future__ import annotations

from stop_helpers import PILOT, VID, stack

from sd_ground.mav.companion_watchdog import CompanionWatchdog, WatchInput
from sd_ground.services.vehicle_actions import OperatorPresence
from sd_mav.modes import HOME_MODES, STOP_MODES, CopterMode


def _inp(t_s: float, mode: CopterMode | None, **kw: object) -> WatchInput:
    base: dict[str, object] = {
        "now_ms": int(t_s * 1000),
        "onboard_seen_ms": 0,
        "fc_mode": mode,
        "in_flight": True,
        "operator_override": False,
        "battery_latch": False,
    }
    base.update(kw)
    return WatchInput(**base)  # type: ignore[arg-type]


def test_watchdog_phases_pure() -> None:
    w = CompanionWatchdog(0)
    assert w.tick(_inp(2.9, CopterMode.GUIDED)) is None
    assert w.tick(_inp(3.0, CopterMode.GUIDED)) == "watchdog_stop"
    assert w.phase == "stopped"
    for t in (3.1, 10.0, 20.0, 60.0, 600.0):  # its own BRAKE: it hovers, never home
        for mode in (CopterMode.BRAKE, CopterMode.LOITER):
            assert w.tick(_inp(t, mode)) is None, (t, mode)
    assert w.phase == "stopped" and w.alarm_active
    # The operator's FLY HOME (or the battery failsafe) takes it home: the watchdog lets go.
    assert w.tick(_inp(601.0, CopterMode.SMART_RTL)) is None
    # The heartbeat comes back: idle again.
    assert w.tick(_inp(602.0, CopterMode.SMART_RTL, onboard_seen_ms=601_500)) is None
    assert w.phase == "idle" and not w.alarm_active


def test_watchdog_modes() -> None:
    for mode in (
        CopterMode.BRAKE,
        CopterMode.LOITER,
        CopterMode.RTL,
        CopterMode.SMART_RTL,
        CopterMode.LAND,
        CopterMode.ALT_HOLD,
        CopterMode.STABILIZE,
    ):
        w = CompanionWatchdog(0)
        for t in (5.0, 20.0, 25.0, 600.0):  # LOITER (an order ended) included: it hovers
            assert w.tick(_inp(t, mode)) is None, (t, mode)
        assert w.alarm_active and w.phase == "idle"
    # GUIDED seen late (the ground restarted): still only STOP's steps, never home.
    w = CompanionWatchdog(0)
    assert w.tick(_inp(25.0, CopterMode.GUIDED)) == "watchdog_stop"
    assert w.tick(_inp(60.0, CopterMode.BRAKE)) is None


def test_watchdog_stops_an_auto_flight_and_never_flies_it_home() -> None:
    """Option A: nobody steers the search mission any more, so the drone stops at 3 s,
    and that stop stands; only the operator, the battery or the radio failsafe ends it."""
    w = CompanionWatchdog(0)
    assert w.tick(_inp(2.9, CopterMode.AUTO)) is None
    assert w.tick(_inp(3.0, CopterMode.AUTO)) == "watchdog_stop"
    assert w.phase == "stopped"
    for t in (5.0, 20.0, 60.0, 600.0):
        for mode in (CopterMode.BRAKE, CopterMode.LOITER):
            assert w.tick(_inp(t, mode)) is None, (t, mode)
    assert w.phase == "stopped" and w.alarm_active
    # Seen late (the ground restarted): an AUTO flight still only stops.
    w = CompanionWatchdog(0)
    assert w.tick(_inp(25.0, CopterMode.AUTO)) == "watchdog_stop"
    assert w.tick(_inp(60.0, CopterMode.BRAKE)) is None
    # The heartbeat comes back and goes again in GUIDED: a new stop, and again no flight home.
    assert w.tick(_inp(61.0, CopterMode.BRAKE, onboard_seen_ms=60_500)) is None
    assert w.phase == "idle"
    assert w.tick(_inp(64.0, CopterMode.GUIDED, onboard_seen_ms=60_500)) == "watchdog_stop"
    assert w.tick(_inp(81.0, CopterMode.BRAKE, onboard_seen_ms=60_500)) is None


def test_watchdog_respects_the_operator_and_the_ground() -> None:
    w = CompanionWatchdog(0)
    assert w.tick(_inp(5.0, CopterMode.GUIDED, operator_override=True)) is None
    assert w.alarm_active  # ONBOARD_SILENT is still shown
    assert w.tick(_inp(25.0, CopterMode.BRAKE, operator_override=True)) is None
    w = CompanionWatchdog(0)
    assert w.tick(_inp(5.0, CopterMode.GUIDED, battery_latch=True)) is None
    w = CompanionWatchdog(0)
    assert w.tick(_inp(5.0, CopterMode.GUIDED, in_flight=False)) is None
    assert not w.alarm_active
    w = CompanionWatchdog(0)  # the pilot steers by hand: only the alarm
    assert w.tick(_inp(5.0, CopterMode.GUIDED, manual=True)) is None
    assert w.tick(_inp(25.0, CopterMode.LOITER, manual=True)) is None
    assert w.alarm_active


async def test_watchdog_leaves_the_pilots_manual_control_alone() -> None:
    async with stack() as s:
        s.fly()
        await s.run(1.0)
        s.actions.takeover(VID, client_action_id="t1", by=PILOT)
        await s.run(1.0)
        assert s.fc.mode == CopterMode.LOITER
        s.onboard_alive[0] = False
        await s.run(25.0)
        assert s.fc.mode == CopterMode.LOITER and s.latches.get(VID) is None
        assert "alarm.ONBOARD_SILENT" in s.g.types()


async def test_battery_guard_also_after_the_operators_stop() -> None:
    """Low battery flies home also after STOP (the battery latch then replaces it)."""
    async with stack(rtl_pct=20.0) as s:
        s.fly()
        await s.run(1.0)
        s.stop()
        await s.run(1.0)
        s.fc.battery_pct = 19.0
        await s.run(1.5)
        assert s.fc.mode in HOME_MODES
        assert s.latches.get(VID).action == "battery_rtl"  # type: ignore[union-attr]


async def test_watchdog_stops_a_silent_guided_flight_and_never_flies_it_home() -> None:
    async with stack() as s:
        s.fly()
        await s.run(1.0)
        s.onboard_alive[0] = False  # the onboard computer falls silent
        await s.run(3.4)  # its last heartbeat came up to 1 s earlier
        assert s.fc.mode in STOP_MODES
        latch = s.latches.get(VID)
        assert latch is not None and latch.action == "watchdog_stop"
        assert "vehicle.onboard_silent" in s.g.types()
        assert "alarm.ONBOARD_SILENT" in s.g.types()
        assert "alarm.MISSION_FALLBACK" not in s.g.types()  # no search mission was flying
        await s.run(60.0)  # well past the old 20 s flight home
        assert s.fc.mode in STOP_MODES
        assert s.latches.get(VID).action == "watchdog_stop"  # type: ignore[union-attr]
        assert "vehicle.onboard_silent_rtl" not in s.g.types()


async def test_watchdog_stops_a_silent_auto_flight_and_it_stays_stopped() -> None:
    """The search mission flies in AUTO and the onboard computer falls silent: STOP's
    steps at 3 s, and no flight home however long the silence lasts."""
    async with stack() as s:
        s.fly(mode=CopterMode.AUTO)
        await s.run(1.0)
        s.onboard_alive[0] = False
        await s.run(3.4)
        assert s.fc.mode in STOP_MODES
        latch = s.latches.get(VID)
        assert latch is not None and latch.action == "watchdog_stop"
        assert "alarm.ONBOARD_SILENT" in s.g.types()
        assert "alarm.MISSION_FALLBACK" in s.g.types()  # the search stopped mid-way
        await s.run(60.0)
        assert s.fc.mode in STOP_MODES
        assert s.latches.get(VID).action == "watchdog_stop"  # type: ignore[union-attr]
        assert "vehicle.onboard_silent_rtl" not in s.g.types()


async def test_battery_guard_flies_a_stopped_silent_auto_flight_home() -> None:
    """After the watchdog's stop of an AUTO flight the battery still brings it home."""
    async with stack(rtl_pct=20.0) as s:
        s.fly(mode=CopterMode.AUTO)
        await s.run(1.0)
        s.onboard_alive[0] = False
        await s.run(3.4)
        assert s.fc.mode in STOP_MODES
        s.fc.battery_pct = 19.0
        await s.run(1.5)
        assert s.fc.mode in HOME_MODES
        assert s.latches.get(VID).action == "battery_rtl"  # type: ignore[union-attr]


async def test_watchdog_does_nothing_while_the_operators_stop_stands() -> None:
    async with stack() as s:
        s.fly()
        await s.run(1.0)
        s.stop()
        await s.run(1.0)
        s.onboard_alive[0] = False
        await s.run(30.0)
        assert s.fc.mode == CopterMode.BRAKE
        assert s.latches.get(VID).action == "stop"  # type: ignore[union-attr]
        assert "alarm.ONBOARD_SILENT" in s.g.types()


async def test_watchdog_in_rtl_or_on_the_ground_does_nothing() -> None:
    async with stack() as s:
        s.fly(mode=CopterMode.RTL)
        s.fc.target = (s.fc.lat + 0.01, s.fc.lon, 20.0)
        await s.run(1.0)
        s.onboard_alive[0] = False
        await s.run(25.0)
        assert s.latches.get(VID) is None
    async with stack() as s:
        s.onboard_alive[0] = False
        await s.run(25.0)
        assert s.latches.get(VID) is None


async def test_battery_guard_flies_home_within_one_second() -> None:
    async with stack(rtl_pct=20.0) as s:
        s.fly()
        await s.run(1.0)
        s.fc.battery_pct = 19.0
        t0 = s.clock.monotonic()
        while s.fc.mode not in HOME_MODES and s.clock.monotonic() - t0 < 2.0:
            await s.run(0.05)
        assert s.fc.mode in HOME_MODES
        assert s.clock.monotonic() - t0 <= 1.5  # crossing seen at 2 Hz, then within 1 s
        assert s.latches.get(VID).action == "battery_rtl"  # type: ignore[union-attr]
        assert "vehicle.battery_rtl" in s.g.types()
        try:
            s.resume()
        except Exception as exc:
            assert getattr(exc, "code", "") == "BATTERY_RTL_ACTIVE"
        else:
            raise AssertionError("Resume must be refused during a battery flight home")


async def test_stop_during_battery_rtl_then_resume_flies_home_again() -> None:
    async with stack(rtl_pct=20.0) as s:
        s.fly()
        await s.run(1.0)
        s.fc.battery_pct = 19.0
        await s.run(1.5)
        assert s.fc.mode in HOME_MODES
        s.stop()  # the operator sees the way home is dangerous
        await s.run(2.0)
        latch = s.latches.get(VID)
        assert latch is not None and (latch.action, latch.underlying) == ("stop", "battery_rtl")
        assert s.fc.mode == CopterMode.BRAKE  # the guard does not fight the STOP
        s.resume()
        await s.run(1.5)
        assert s.fc.mode in HOME_MODES  # still low: home again within 1 s
        assert s.latches.get(VID).action == "battery_rtl"  # type: ignore[union-attr]


async def test_operator_absent_holds_once() -> None:
    async with stack() as s:
        s.fly()
        await s.run(1.0)
        s.g.operator_seen_mono = s.clock.monotonic()
        presence = OperatorPresence(s.actions, mission_active=lambda _v: True)
        assert presence.absent_s == 30.0
        presence.tick(s.g)
        await s.run(29.9)
        presence.tick(s.g)
        assert s.latches.get(VID) is None
        await s.run(0.2)
        presence.tick(s.g)
        await s.run(1.0)
        assert s.latches.get(VID).action == "watchdog_stop"  # type: ignore[union-attr]
        assert s.fc.mode == CopterMode.BRAKE
        assert "vehicle.operator_absent" in s.g.types()
        presence.tick(s.g)
        assert s.g.types().count("vehicle.operator_absent") == 1


async def test_a_pilot_back_within_30_s_of_a_restart_keeps_the_flight() -> None:
    async with stack() as s:
        s.fly()
        await s.run(1.0)
        presence = OperatorPresence(s.actions, mission_active=lambda _v: True)
        presence.tick(s.g)  # the restarted ground station has seen no pilot yet
        await s.run(5.0)
        s.g.operator_seen_mono = s.clock.monotonic()  # the panel reconnects
        await s.run(29.0)
        presence.tick(s.g)
        assert s.latches.get(VID) is None
        assert "vehicle.operator_absent" not in s.g.types()


async def test_operator_absent_is_always_a_hold() -> None:
    """Owner decision: 30 s without a pilot always gives the alarm and a hold."""
    async with stack() as s:
        s.fly()
        await s.run(1.0)
        presence = OperatorPresence(s.actions, mission_active=lambda _v: True)
        # Nobody seen by this process at all (it restarted mid-flight): 30 s from its
        # first tick, then the hold, as for a pilot who went away.
        presence.tick(s.g)
        await s.run(29.9)
        presence.tick(s.g)
        assert s.latches.get(VID) is None
        await s.run(0.2)
        presence.tick(s.g)
        await s.run(1.5)
        assert s.fc.mode == CopterMode.BRAKE
        assert s.latches.get(VID).action == "watchdog_stop"  # type: ignore[union-attr]
    async with stack() as s:  # no active mission: nothing at all
        presence = OperatorPresence(s.actions, mission_active=lambda _v: False)
        presence.tick(s.g)
        assert "vehicle.operator_absent" not in s.g.types()


def test_operator_absence_cannot_be_configured_away() -> None:
    from pydantic import ValidationError

    from sd_ground.config import GateConfig

    assert GateConfig().operator_absent_s == 30.0
    for bad in (
        {"operator_absent_action": "off"},
        {"operator_absent_action": "return_home"},
        {"operator_absent_s": 300},
        {"operator_absent_s": 1e9},
    ):
        try:
            GateConfig.model_validate(bad)
        except ValidationError:
            continue
        raise AssertionError(f"accepted {bad}")


def test_vehicle_events_are_throttled_per_kind() -> None:
    from sd_ground.mav.thread import vehicle_events

    now = [0.0]
    seen: list[tuple[str, dict[str, object]]] = []
    on = vehicle_events(lambda t, _s, p: seen.append((t, p)), now=lambda: now[0])
    for _ in range(100):
        on("mav.inbound_command_ignored", {"command": 1})
    on("mav.statustext", {"severity": 6, "text": "info only"})  # below warning: dropped
    on("mav.statustext", {"severity": 4, "text": "warning"})
    now[0] = 1.5
    on("mav.inbound_command_ignored", {"command": 1})
    assert [t for t, _ in seen] == [
        "mav.inbound_command_ignored",
        "fc.statustext",
        "mav.inbound_command_ignored",
    ]
    assert seen[-1][1]["skipped"] == 99


def test_failsafe_texts_are_kept_at_any_level() -> None:
    """ArduCopter sends a fence breach as a notice; it must still reach the events."""
    from sd_ground.mav.thread import vehicle_events

    now = [0.0]
    seen: list[tuple[str, dict[str, object]]] = []
    on = vehicle_events(lambda t, _s, p: seen.append((t, p)), now=lambda: now[0])
    on("mav.statustext", {"severity": 5, "text": "Polygon fence breached"})
    now[0] = 2.0
    on("mav.statustext", {"severity": 6, "text": "Throttle armed"})  # not a failsafe
    assert [(t, p["text"]) for t, p in seen] == [("fc.statustext", "Polygon fence breached")]
