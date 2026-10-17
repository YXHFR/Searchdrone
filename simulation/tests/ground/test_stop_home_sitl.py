"""STOP, FLY HOME and the companion watchdog against ArduCopter SITL (builtin physics).

The ground station's emergency path runs in this process: ``GroundMav`` on the ground's
MAVLink channel (250/190, through linksim), with the real ``Latches`` and
``VehicleActions``. The onboard executor runs in-process too, as in the onboard SITL
tests (mavtap, a test ground station on SD-link). The notices go to the onboard side over
that SD-link session. ``sd-onboard`` dying is played by stopping the executor and closing
its MAVLink channel (no more heartbeats from 1/191).

* STOP with the onboard computer gone: BRAKE in the HEARTBEAT within 1 s, standing still
  within 5 s; FLY HOME likewise, landing within 3 m of home;
* STOP armed on the ground disarms; BRAKE holds for 60 s without RC;
* only the ground's MAVLink cut: STOP still arrives through the onboard fallback;
* the companion watchdog: BRAKE within 4 s and FLY HOME within 21 s when the onboard
  computer dies in GUIDED, nothing while the operator's STOP stands;
* the ground station dies: the flight controller's GCS failsafe flies home.
"""

from __future__ import annotations

import asyncio
import contextlib
import json
import sys
import time
import urllib.request
from collections.abc import AsyncIterator
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import pytest
from pymavlink.dialects.v20 import ardupilotmega as mavlink

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "onboard"))

from test_onboard_sitl import (  # pyright: ignore[reportMissingImports]
    PARM,
    Flight,
    Sim,
    _before_executor,
    _commander,
    _install,
    _read_acks,
    _takeoff,
    _vehicle,
    _wait_home,
    sim,
)

from sd_ground.auth import Principal
from sd_ground.liveness import LoopLiveness
from sd_ground.mav.thread import GroundMav, HeartbeatGate, ThreadSafeMav
from sd_ground.services.latches import Latches
from sd_ground.services.vehicle_actions import VehicleActions
from sd_link.session import LinkSession
from sd_link_testing import FakeGround, LossyProxy
from sd_mav import CopterMode, Role, Vehicle
from sd_mav.modes import HOME_MODES, STOP_MODES
from sd_obs.clock import MonotonicClock
from sd_onboard.executor.executor import Executor, ExecutorOptions
from sd_onboard.io.params import effective_params
from sd_onboard.link_client import OnboardLink
from sd_safety.targets import Plane, distance_m
from sd_schemas.base import SdModel, new_id
from sd_schemas.executor import ExecutorState as S
from sd_schemas.link import ClearanceConfirmation
from sd_sim.util import LOOPBACK_OPENER

pytestmark = [pytest.mark.sitl, pytest.mark.slow]
__all__ = ["sim"]  # the fixture, imported from the onboard SITL rig

VID = "sim-01"
PILOT = Principal(role="pilot", issuer_ref="operator:pilot", auth_mode="local")


@dataclass
class SessionSink:
    """The notices go to the onboard side over the test ground's SD-link session."""

    session: LinkSession | None = None
    proxy: LossyProxy | None = None

    def up(self, vehicle_id: str) -> bool:
        return self.session is not None and not (self.proxy is not None and self.proxy.broken)

    async def send(self, vehicle_id: str, message: SdModel) -> bool:
        if self.session is None:
            return False
        await self.session.send(message)
        return True


@dataclass
class Emergency:
    f: Flight
    core: GroundMav
    actions: VehicleActions
    sink: SessionSink
    pump: asyncio.Task[None]
    executor_tasks: list[asyncio.Future[Any]] = field(default_factory=list)
    onboard_dead: bool = False

    @property
    def fc(self) -> Any:
        return self.core.vehicle.telemetry

    def mode(self) -> str:
        return self.fc.fc_mode

    async def kill_onboard(self) -> None:
        """``sd-onboard`` gone: no executor, no heartbeat from 1/191."""
        for task in self.executor_tasks:
            task.cancel()
        await asyncio.gather(*self.executor_tasks, return_exceptions=True)
        await self.f.onboard.close()
        self.onboard_dead = True

    def stop(self) -> Any:
        return self.actions.dispatch(
            "stop", VID, client_action_id=str(new_id()), via="button", by=PILOT
        )

    def home(self) -> Any:
        return self.actions.dispatch(
            "return_home", VID, client_action_id=str(new_id()), via="button", by=PILOT
        )

    async def wait(self, cond: Any, limit_s: float, what: str) -> float:
        start = time.monotonic()
        while not cond():
            if time.monotonic() - start > limit_s:
                t = self.fc
                raise AssertionError(
                    f"{what}: not within {limit_s} s (mode {self.mode()}, "
                    f"{t.groundspeed_ms:.1f} m/s, {t.alt_rel_home_m:.1f} m, "
                    f"{self.from_home_m():.0f} m from home, executor {self.f.ex.state})"
                )
            await asyncio.sleep(0.02)
        return time.monotonic() - start

    def from_home_m(self) -> float:
        home = self.fc.home
        assert home is not None
        return distance_m(home.lat, home.lon, self.fc.lat, self.fc.lon)


class _Pump:
    def __init__(self) -> None:
        from sd_ground.bus import EventBus

        self.clock = MonotonicClock()
        self.bus = EventBus()
        self.events: list[str] = []
        self.operator_seen_mono: float | None = None

    def add_event(self, type_: str, **_: Any) -> None:
        self.events.append(type_)


@contextlib.asynccontextmanager
async def emergency(
    sim: Sim,
    *,
    rtl_pct: list[float] | None = None,
    liveness: LoopLiveness | None = None,
    **options: Any,
) -> AsyncIterator[Emergency]:
    clock = MonotonicClock()
    ports = sim.ports
    holder: list[Executor] = []
    onboard = _vehicle(
        f"udpin:127.0.0.1:{ports['onboard_mavlink']}",
        Role.ONBOARD,
        on_event=lambda k, d: holder[0].on_vehicle_event(k, d) if holder else None,
        heartbeat_allowed=lambda: bool(holder) and holder[0].loop_alive(),
        send_gate=lambda o: holder[0].send_gate(o) if holder else _before_executor(o),
    )
    sent: list[Any] = []
    write = onboard._transport.write_message  # record what 1/191 writes

    def recording(msg: Any) -> None:
        sent.append(msg)
        write(msg)

    onboard._transport.write_message = recording  # type: ignore[method-assign]
    ground: Vehicle = _vehicle(
        f"udpin:127.0.0.1:{ports['ground_mavlink']}",
        Role.GROUND,
        heartbeat_allowed=None if liveness is None else HeartbeatGate(liveness),
    )
    level = rtl_pct if rtl_pct is not None else [15.0]
    latches = Latches(clock)
    sink = SessionSink()
    actions = VehicleActions(clock, latches, vehicle_ids=[VID], notices=sink)
    core = GroundMav(
        ground,
        clock,
        vehicle_id=VID,
        latches=latches,
        on_report=actions.on_report,
        on_event=actions.on_event,
        system_action=actions.system,
        rtl_pct=lambda: level[0],
    )
    actions.mav = ThreadSafeMav(core, asyncio.get_running_loop())
    await core.start()
    await onboard.start()
    assert await ground.wait_connected(20) and await onboard.wait_connected(20)
    await _wait_home(ground)
    home = ground.telemetry.home
    assert home is not None
    g = _Pump()
    async with FakeGround(clock=clock) as fake, LossyProxy(fake.url) as proxy:
        link = OnboardLink(url=proxy.url, token=fake.token, vehicle_id=VID, clock=clock)
        ex = Executor(
            onboard,
            clock,
            link,
            ExecutorOptions(vehicle_id=VID, expected_params=effective_params([PARM]), **options),
        )
        holder.append(ex)
        link.executor = ex
        f = Flight(ex, onboard, ground, fake, proxy, Plane(home.lat, home.lon), sim, sent=sent)
        stop = asyncio.Event()
        e = Emergency(f, core, actions, sink, asyncio.ensure_future(actions.pump(g)))
        e.executor_tasks += [
            asyncio.ensure_future(ex.run(stop)),
            asyncio.ensure_future(link.run(stop)),
        ]
        try:
            f.session = await asyncio.wait_for(fake.sessions.get(), 30)
            sink.session, sink.proxy = f.session, proxy
            f.tasks += [
                asyncio.ensure_future(_read_acks(f, f.session)),
                asyncio.ensure_future(_commander(f)),
            ]
            yield e
        finally:
            stop.set()
            for task in [*f.tasks, *e.executor_tasks, e.pump]:
                task.cancel()
            await asyncio.gather(*f.tasks, *e.executor_tasks, e.pump, return_exceptions=True)
            await ex.close()
            if not e.onboard_dead:
                await onboard.close()
            await core.close()


async def _airborne_goto(e: Emergency, east: float, speed: float, alt: float = 25.0) -> None:
    f = e.f
    await _install(f)
    await _takeoff(f, alt)
    await f.until(lambda: f.ex.state == S.HOLD, 60, "takeoff done")
    lat, lon = f.at(east, 0)
    await f.order("goto", {"lat": lat, "lon": lon, "alt_m": alt, "speed_ms": speed})
    await f.until(lambda: f.ex.state == S.TRANSIT, 10, "transit")


def _set_ground_link(sim: Sim, up: bool) -> None:
    body = json.dumps({"state": "up" if up else "down"}).encode()
    req = urllib.request.Request(
        f"http://127.0.0.1:{sim.ports['linksim_control']}/flows/mavlink",
        data=body,
        method="POST",
        headers={
            "Authorization": f"Bearer {sim.stack.linksim_token}",
            "Content-Type": "application/json",
        },
    )
    with LOOPBACK_OPENER.open(req, timeout=2) as resp:
        assert resp.status in (200, 204)


# --- STOP and FLY HOME past the onboard computer -----------------------------------------


@pytest.mark.budget_s(240)
def test_stop_bypasses_onboard(sim: Sim) -> None:
    """``sd-onboard`` frozen during a goto at full speed: BRAKE <= 1.0 s, < 0.5 m/s in 5 s.

    The goto asks for 8 m/s; ArduCopter 4.7.1 flies GUIDED at ``WP_SPD`` (5 m/s) here, so
    the test waits for cruise speed rather than for 8 m/s."""

    async def run() -> None:
        async with emergency(sim) as e:
            await _airborne_goto(e, 200, 8)
            await e.wait(lambda: e.fc.groundspeed_ms > 4.5, 30, "cruise speed")
            await e.kill_onboard()
            t0 = time.monotonic()
            e.stop()
            took = await e.wait(lambda: e.mode() == "BRAKE", 1.0, "BRAKE")
            assert took <= 1.0
            await e.wait(lambda: e.fc.groundspeed_ms < 0.5, 5.0 - (time.monotonic() - t0), "still")

    asyncio.run(run())


@pytest.mark.budget_s(300)
def test_home_bypasses_onboard(sim: Sim) -> None:
    """``sd-onboard`` killed in flight: FLY HOME, SMART_RTL/RTL <= 1.0 s, landed within 3 m."""

    async def run() -> None:
        async with emergency(sim) as e:
            await _airborne_goto(e, 80, 6, alt=20)
            await e.f.until(lambda: e.f.ex.state == S.HOLD, 60, "at the target")
            await e.kill_onboard()
            e.home()
            await e.wait(lambda: e.mode() in ("SMART_RTL", "RTL"), 1.0, "home mode")
            await e.wait(lambda: not e.fc.armed, 120, "landed")
            assert e.from_home_m() <= 3.0

    asyncio.run(run())


@pytest.mark.budget_s(180)
def test_stop_on_ground_armed(sim: Sim) -> None:
    """``takeoff`` with 5 s between ARM and the climb; STOP 1 s after ARM (armed, on the
    ground): disarmed within 1 s and NAV_TAKEOFF is never sent."""

    async def run() -> None:
        async with emergency(sim, takeoff_delay_after_arm_s=5.0) as e:
            f = e.f
            c = await _install(f)
            assert f.session is not None
            await f.session.send(
                ClearanceConfirmation(
                    clearance_id=new_id(),
                    mission_id=c.mission_id,
                    vehicle_id=VID,
                    charter_id=c.charter_id,
                    charter_version=c.version,
                    valid_for_ms=120_000,
                    confirmed_by="operator:pilot",
                    origin="local",
                ),
                ttl_ms=5000,
            )
            await f.until(lambda: f.ex.snapshot().clearance_available, 10, "clearance onboard")
            await f.order("takeoff", {"alt_m": 20})
            await e.wait(lambda: e.fc.armed, 30, "armed")
            await asyncio.sleep(1.0)
            e.stop()
            await e.wait(lambda: not e.fc.armed, 1.0, "disarmed")
            await asyncio.sleep(6.0)
            assert not e.fc.armed and e.fc.alt_rel_home_m < 0.5
            assert not [m for m in f.sent if _is_command(m, 22)]  # MAV_CMD_NAV_TAKEOFF

    asyncio.run(run())


@pytest.mark.budget_s(260)
def test_brake_holds_without_rc(sim: Sim) -> None:
    """BRAKE for 60 s with the RC transmitter off: drift <= 2 m across, <= 1 m up/down."""

    async def run() -> None:
        async with emergency(sim) as e:
            await _airborne_goto(e, 60, 4)
            e.stop()
            await e.wait(lambda: e.mode() == "BRAKE", 1.0, "BRAKE")
            await e.wait(lambda: e.fc.groundspeed_ms < 0.3, 10, "still")
            await asyncio.sleep(2.0)
            e.f.sim.rc.__exit__()  # no RC from here on
            lat, lon, alt = e.fc.lat, e.fc.lon, e.fc.alt_rel_home_m
            await asyncio.sleep(60.0)
            assert e.mode() == "BRAKE"
            assert distance_m(lat, lon, e.fc.lat, e.fc.lon) <= 2.0
            assert abs(e.fc.alt_rel_home_m - alt) <= 1.0

    asyncio.run(run())


@pytest.mark.budget_s(240)
def test_stop_via_sdlink_when_ground_mavlink_down(sim: Sim) -> None:
    """Only the ground's MAVLink is cut: STOP reaches the flight controller through the
    onboard fallback, BRAKE or LOITER within 2.5 s."""

    async def run() -> None:
        async with emergency(sim) as e:
            await _airborne_goto(e, 200, 5)
            _set_ground_link(sim, False)
            try:
                await e.wait(lambda: not e.core.snapshot().connected, 5, "ground MAVLink down")  # type: ignore[union-attr]
                reply = e.stop()
                assert reply.paths == {"mavlink": "unavailable", "sdlink": "queued"}
                took = await e.wait(
                    lambda: e.f.onboard.telemetry.fc_mode in ("BRAKE", "LOITER"), 2.5, "stopped"
                )
                assert took <= 2.5
                assert any(
                    _is_command(m, 176) and int(m.param2) == int(CopterMode.BRAKE) for m in e.f.sent
                ), "the onboard fallback sent BRAKE"
                assert e.f.ex.state == S.HELD
            finally:
                _set_ground_link(sim, True)

    asyncio.run(run())


# --- the companion watchdog and a dead ground station -------------------------------------


@pytest.mark.budget_s(300)
def test_onboard_kill_in_guided(sim: Sim) -> None:
    """``sd-onboard`` killed in GUIDED: BRAKE/LOITER <= 4 s, SMART_RTL/RTL <= 21 s, home."""

    async def run() -> None:
        async with emergency(sim) as e:
            await _airborne_goto(e, 80, 4, alt=20)
            assert e.mode() == "GUIDED"
            await e.kill_onboard()
            killed = time.monotonic()
            await e.wait(lambda: e.mode() in ("BRAKE", "LOITER"), 4.0, "watchdog stop")
            await e.wait(
                lambda: e.mode() in ("SMART_RTL", "RTL"),
                21.0 - (time.monotonic() - killed),
                "watchdog home",
            )
            assert time.monotonic() - killed <= 21.0
            await e.wait(lambda: not e.fc.armed, 150, "landed")
            assert e.from_home_m() <= 3.0

    asyncio.run(run())


@pytest.mark.budget_s(240)
def test_watchdog_respects_stop(sim: Sim) -> None:
    """STOP, then ``sd-onboard`` dies: still BRAKE/LOITER after 30 s, ONBOARD_SILENT shown."""

    async def run() -> None:
        async with emergency(sim) as e:
            await _airborne_goto(e, 80, 4, alt=20)
            e.stop()
            await e.wait(lambda: e.mode() == "BRAKE", 1.0, "BRAKE")
            await e.kill_onboard()
            await asyncio.sleep(30.0)
            assert e.mode() in ("BRAKE", "LOITER")
            assert e.core.watchdog.alarm_active
            latch = e.actions.latches.get(VID)
            assert latch is not None and latch.action == "stop"

    asyncio.run(run())


@pytest.mark.budget_s(240)
def test_ground_kill_gcs_failsafe(sim: Sim) -> None:
    """``sd-ground`` dies during a goto: the GCS failsafe flies home within 21 s; the
    executor goes to FAILSAFE and steers nothing from then on."""

    async def run() -> None:
        async with emergency(sim) as e:
            await _airborne_goto(e, 200, 4)
            await e.core.close()  # no GCS heartbeat from 250/190 any more
            await e.f.proxy.cut()
            e.f.proxy.broken = True
            dead = time.monotonic()
            await e.f.until(
                lambda: e.f.onboard.telemetry.fc_mode in ("SMART_RTL", "RTL"),
                25,
                "GCS failsafe",
            )
            assert time.monotonic() - dead <= 21.0
            await e.f.until(lambda: e.f.ex.state == S.FAILSAFE, 5, "executor in FAILSAFE")
            index = len(e.f.sent)  # the order may run until the failsafe; nothing after it
            await asyncio.sleep(10.0)
            assert e.f.ex.state == S.FAILSAFE
            assert e.f.steering_since(index) == []
            assert e.f.onboard.telemetry.fc_mode in {m.name for m in HOME_MODES | STOP_MODES}

    asyncio.run(run())


# --- the 10 s watch, the battery guard and a hung ground station -------------------------


@pytest.mark.budget_s(240)
def test_stop_mode_race(sim: Sim) -> None:
    """A late GUIDED from 1/191, 50 ms after STOP (injected by the test, past the onboard
    gate): the flight controller is in a STOP mode at 2 s and stays there for 10 s."""

    async def run() -> None:
        async with emergency(sim) as e:
            await _airborne_goto(e, 120, 4)
            t0 = time.monotonic()
            e.stop()
            await asyncio.sleep(0.05)
            late = mavlink.MAVLink_command_long_message(
                1, 1, 176, 0, mavlink.MAV_MODE_FLAG_CUSTOM_MODE_ENABLED, 4, 0, 0, 0, 0, 0
            )
            e.f.onboard._transport.write_message(late)  # pyright: ignore[reportPrivateUsage]
            await asyncio.sleep(2.0 - (time.monotonic() - t0))
            assert e.mode() in {m.name for m in STOP_MODES}
            for _ in range(100):
                await asyncio.sleep(0.1)
                assert e.mode() in {m.name for m in STOP_MODES}, e.mode()

    asyncio.run(run())


@pytest.mark.budget_s(240)
def test_battery_guard_flies_home(sim: Sim) -> None:
    """The ground's battery guard: the level is crossed in flight (here the guard's level
    is raised above the battery), SMART_RTL/RTL within 1 s and the battery latch."""

    async def run() -> None:
        level = [0.0]
        async with emergency(sim, rtl_pct=level) as e:
            await _airborne_goto(e, 60, 4)
            assert e.fc.battery_pct is not None
            level[0] = e.fc.battery_pct + 1.0  # the battery is now below the level
            took = await e.wait(lambda: e.mode() in ("SMART_RTL", "RTL"), 1.5, "battery home")
            assert took <= 1.5  # the crossing is seen at 2 Hz, then within 1 s
            latch = e.actions.latches.get(VID)
            assert latch is not None and latch.action == "battery_rtl"

    asyncio.run(run())


@pytest.mark.budget_s(240)
def test_hung_ground_station_lets_the_gcs_failsafe_act(sim: Sim) -> None:
    """The main loop stops beating: after 8 s no GCS heartbeat goes out, so the flight
    controller's GCS failsafe flies home (8 s + 20 s)."""

    async def run() -> None:
        liveness = LoopLiveness()
        beating = [True]

        async def beat() -> None:
            while True:
                if beating[0]:
                    liveness.beat("main")
                await asyncio.sleep(0.5)

        beater = asyncio.ensure_future(beat())
        try:
            async with emergency(sim, liveness=liveness) as e:
                await _airborne_goto(e, 200, 3)
                beating[0] = False  # the main loop hangs from here
                hung = time.monotonic()
                await e.f.until(
                    lambda: e.f.onboard.telemetry.fc_mode in ("SMART_RTL", "RTL"),
                    35,
                    "GCS failsafe",
                )
                assert time.monotonic() - hung >= 8.0 + 15.0  # not before the stall + timeout
        finally:
            beater.cancel()

    asyncio.run(run())


@pytest.mark.budget_s(240)
def test_rc_mode_switch_during_fly_home_stands(sim: Sim) -> None:
    """FLY HOME, then a person flips the RC mode switch: whatever mode the switch
    selects stands; the ground's 10 s watch does not take it back."""

    async def run() -> None:
        async with emergency(sim) as e:
            await _airborne_goto(e, 80, 4)
            await e.wait(lambda: e.fc.rc_mode_pwm is not None, 5, "RC channels")
            e.home()
            await e.wait(lambda: e.mode() in ("SMART_RTL", "RTL"), 1.0, "home mode")
            await asyncio.sleep(1.0)
            e.f.sim.rc.mode_pwm = 1900  # the pilot's switch
            await e.wait(lambda: e.mode() not in ("SMART_RTL", "RTL"), 3.0, "RC mode")
            chosen = e.mode()
            for _ in range(50):
                await asyncio.sleep(0.1)
                assert e.mode() == chosen, (chosen, e.mode())

    asyncio.run(run())


def _is_command(msg: Any, command: int) -> bool:
    return msg.get_type() == "COMMAND_LONG" and int(msg.command) == command
