"""The onboard executor against ArduCopter SITL (builtin physics, headless).

A fresh simulator per test. The executor runs in this process: its MAVLink channel goes
through mavtap like the real ``sd-onboard``, a ground MAVLink channel (250/190) keeps the
GCS failsafe away and uploads the fence, and SD-link is a real WebSocket to a test ground
station (``sd_link_testing.FakeGround``, no ground gate), through a proxy that can break it.

* the safety parameters are loaded and cover the hard limits;
* the demo flight: take off to 40 m, fly to P1, search a box, fly home and land, with
  only allowed MAVLink from 1/191 and exactly one ARM;
* a mode change from outside makes the executor let go and stay quiet;
* SD-link lost for 10 s changes nothing; lost for 60 s: hold at 23 s, home at 53 s.
* a silent AI ends in a hold (never home); a lowered height limit makes it sink below it,
  a shrunken area sends it home;
* STOP holds everything onboard until Resume, which leaves BRAKE for LOITER.

A simulated RC transmitter (throttle down on the ground, centred in the air) stands in
for the pilot: without RC the simulator's throttle is zero and LOITER sinks.
"""

from __future__ import annotations

import asyncio
import contextlib
import datetime as dt
import math
import socket
import struct
import threading
import time
from collections.abc import AsyncIterator, Iterator
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import pytest

from sd_geo import coverage_fraction
from sd_link.session import LinkSession
from sd_link_testing import FakeGround, LossyProxy, next_inbound
from sd_mav import CopterMode, Role, SendOrigin, Vehicle, connect
from sd_mav.mission import FENCE, download, fence_hash, fence_items, upload
from sd_obs.clock import MonotonicClock
from sd_onboard.executor.executor import Executor, ExecutorOptions
from sd_onboard.executor.limits import inside_charter
from sd_onboard.io.params import effective_params, parse_parm
from sd_onboard.link_client import OnboardLink
from sd_safety.constants import (
    FENCE_ALT_MARGIN_M,
    FENCE_POLYGON_MARGIN_M,
    FENCE_RADIUS_MARGIN_M,
    LINK_LOSS_DETECT_S,
    SDLINK_GRACE_S,
    SDLINK_HOLD_S,
)
from sd_safety.limits import (
    ABSOLUTE_MAX_ALT_AGL_M,
    ABSOLUTE_MAX_RANGE_FROM_PILOT_M,
    ABSOLUTE_MAX_SPEED_MS,
)
from sd_safety.targets import Plane
from sd_schemas.base import LatLonAlt, new_id
from sd_schemas.charter import AutonomyMode, MissionCharter
from sd_schemas.executor import ExecutorState as S
from sd_schemas.gate import StateRef
from sd_schemas.link import (
    Ack,
    CharterAck,
    CharterInstall,
    ClearanceConfirmation,
    CommanderStatus,
    OperatorOverrideNotice,
    OperatorRelease,
    SkillCommand,
)
from sd_sim import config, paths
from sd_sim.harness import Harness
from sd_sim.launch import Stack
from sd_sim.pytest_plugin import worker_instance
from sd_sim.sitl import CMAC_HOME, SAFETY_PARAM_FILE
from sd_sim.util import new_run_dir

pytestmark = [
    pytest.mark.sitl,
    pytest.mark.slow,
    pytest.mark.skipif(not paths.arducopter_bin().is_file(), reason="ArduCopter SITL is not built"),
]

REPO = Path(__file__).resolve().parents[3]
PARM = REPO / "simulation/sitl/params/searchdrone-safety.parm"
READY_S = 90.0
AREA_HALF_M = 200.0
P1 = (100.0, 50.0)  # east, north of home (m)
SURVEY_HALF_M = 40.0
_STEERING_IGNORED = frozenset(
    {
        "HEARTBEAT",
        "PARAM_REQUEST_READ",
        "MISSION_REQUEST_LIST",
        "MISSION_REQUEST_INT",
        "MISSION_ACK",
    }
)
_ALLOWED_TYPES = frozenset(
    {
        *_STEERING_IGNORED,
        "COMMAND_LONG",
        "SET_POSITION_TARGET_GLOBAL_INT",
        "MISSION_COUNT",
        "MISSION_ITEM_INT",
        "MISSION_SET_CURRENT",
    }
)
_ALLOWED_COMMANDS = frozenset({176, 400, 22, 178, 224, 511, 512})


@dataclass
class Sim:
    stack: Stack
    harness: Harness
    rc: RcSticks

    @property
    def ports(self) -> dict[str, int]:
        return self.stack.ports


class RcSticks:
    """A pilot holding an RC transmitter: throttle down on the ground (arming needs it),
    centred in the air (1500 us holds the height in LOITER), the other sticks centred.
    Sends SITL's RC input over UDP at 50 Hz; without it the simulator's throttle stays at
    its minimum and LOITER sinks to the ground."""

    def __init__(self, port: int) -> None:
        self._sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        self._addr = ("127.0.0.1", port)
        self._stop = threading.Event()
        self._thread = threading.Thread(target=self._run, name="rc-sticks", daemon=True)
        self.throttle = 1000
        self.mode_pwm = 1500  # channel 5, ArduCopter's flight-mode switch

    def __enter__(self) -> RcSticks:
        self._thread.start()
        return self

    def __exit__(self, *exc: object) -> None:
        self._stop.set()
        self._thread.join(2)
        self._sock.close()

    def _run(self) -> None:
        while not self._stop.wait(0.02):
            packet = struct.pack(
                "<8H", 1500, 1500, self.throttle, 1500, self.mode_pwm, 1500, 1500, 1500
            )
            self._sock.sendto(packet, self._addr)


@contextlib.contextmanager
def simulator(safety_file: str = SAFETY_PARAM_FILE) -> Iterator[Sim]:
    """A fresh SITL with linksim and mavtap, ready to fly, with the RC sticks held."""
    instance = worker_instance()
    cfg = config.load(
        {"physics": "builtin", "gui": "false", "video": "off", "instance": str(instance)}
    )
    cfg.home, cfg.personal_home = CMAC_HOME, False
    st = Stack(cfg, new_run_dir(instance, "onboard"), observer=False, safety_file=safety_file)
    harness = Harness(st.ports["harness_mavlink"])
    try:
        st.up()
        harness.wait(lambda s: s.heartbeat, 60, "HEARTBEAT från flygkontrollen")
        harness.wait(lambda s: s.ready, READY_S, "position från EKF och GPS")
        with RcSticks(st.ports["sitl_rc_in"]) as rc:
            yield Sim(st, harness, rc)
    finally:
        harness.close()
        st.down()


@pytest.fixture
def sim() -> Iterator[Sim]:
    with simulator() as s:
        yield s


@dataclass
class Flight:
    ex: Executor
    onboard: Vehicle
    ground: Vehicle
    fake: FakeGround
    proxy: LossyProxy
    plane: Plane
    sim: Sim
    session: LinkSession | None = None
    charter: MissionCharter | None = None
    sent: list[Any] = field(default_factory=list)  # everything 1/191 wrote
    track: list[tuple[float, float, float, float]] = field(default_factory=list)  # t, lat, lon, alt
    acks: dict[Any, list[Ack]] = field(default_factory=dict)
    commander: bool = True  # whether the test ground reports the AI as connected
    tasks: list[asyncio.Future[Any]] = field(default_factory=list)

    def at(self, east: float, north: float) -> tuple[float, float]:
        return self.plane.latlon(east, north)

    def square(self, half: float, east: float = 0.0, north: float = 0.0) -> dict[str, Any]:
        ring = []
        for x, y in [(-half, -half), (half, -half), (half, half), (-half, half), (-half, -half)]:
            lat, lon = self.at(x + east, y + north)
            ring.append([lon, lat])
        return {"type": "Polygon", "coordinates": [ring]}

    async def until(self, cond: Any, limit_s: float, what: str) -> float:
        start = time.monotonic()
        while not cond():
            if time.monotonic() - start > limit_s:
                raise AssertionError(f"{what}: not within {limit_s:.0f} s (state {self.ex.state})")
            await asyncio.sleep(0.1)
        return time.monotonic() - start

    async def order(self, skill: str, params: dict[str, Any] | None = None) -> SkillCommand:
        snap = self.ex.snapshot()
        assert self.session is not None and self.charter is not None
        cmd = SkillCommand(
            command_id=new_id(),
            skill=skill,
            params=params or {},
            issuer="operator:pilot",
            rationale=None,
            charter_version=self.charter.version,
            based_on=StateRef(state_epoch=snap.state_epoch, state_seq=snap.state_seq),
            approval=None,
        )
        self.acks[cmd.command_id] = []
        await self.session.send(cmd, ttl_ms=15_000)
        return cmd

    async def stage(self, cmd: SkillCommand, wanted: set[str], limit_s: float) -> Ack:
        await self.until(
            lambda: any(
                a.stage in wanted | {"rejected", "failed"} for a in self.acks[cmd.command_id]
            ),
            limit_s,
            f"{cmd.skill} {wanted}",
        )
        return next(
            a for a in self.acks[cmd.command_id] if a.stage in wanted | {"rejected", "failed"}
        )

    def steering_since(self, index: int) -> list[Any]:
        out = []
        for msg in self.sent[index:]:
            kind = msg.get_type()
            if kind in _STEERING_IGNORED:
                continue
            if kind == "COMMAND_LONG" and int(msg.command) in (511, 512):
                continue
            out.append(msg)
        return out


async def _read_acks(f: Flight, session: LinkSession) -> None:
    while True:
        inbound = await next_inbound(session)
        body = inbound.body
        if isinstance(body, Ack) and body.ref_id in f.acks:
            f.acks[body.ref_id].append(body)
        elif isinstance(body, CharterAck):
            f.acks.setdefault("charter", []).append(body)  # type: ignore[arg-type]


async def _commander(f: Flight) -> None:
    while True:
        if f.session is not None and f.commander:
            with contextlib.suppress(Exception):
                await f.session.send(CommanderStatus(connected=True, state="ready"))
        await asyncio.sleep(1.0)


def _before_executor(origin: SendOrigin) -> str | None:
    return None if origin is SendOrigin.STARTUP else "starting"


def _vehicle(url: str, role: Role, **kw: Any) -> Vehicle:
    ids = (1, 191) if role is Role.ONBOARD else (250, 190)
    conn = connect(url, role=role, source_system=ids[0], source_component=ids[1])
    return Vehicle(conn, role=role, clock=MonotonicClock(), **kw)


@contextlib.asynccontextmanager
async def flight(sim: Sim, parm: Path = PARM, **options: Any) -> AsyncIterator[Flight]:
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
    ground = _vehicle(f"udpin:127.0.0.1:{ports['ground_mavlink']}", Role.GROUND)
    await ground.start()
    await onboard.start()
    assert await ground.wait_connected(20) and await onboard.wait_connected(20)
    await _wait_home(ground)
    home = ground.telemetry.home
    assert home is not None
    plane = Plane(home.lat, home.lon)
    async with FakeGround(clock=clock) as fake, LossyProxy(fake.url) as proxy:
        link = OnboardLink(url=proxy.url, token=fake.token, vehicle_id="sim-01", clock=clock)
        params = effective_params([parm])
        ex = Executor(
            onboard,
            clock,
            link,
            ExecutorOptions(vehicle_id="sim-01", expected_params=params, **options),
        )
        holder.append(ex)
        link.executor = ex
        f = Flight(ex, onboard, ground, fake, proxy, plane, sim, sent=sent)
        stop = asyncio.Event()
        f.tasks += [asyncio.ensure_future(ex.run(stop)), asyncio.ensure_future(link.run(stop))]
        onboard.subscribe(
            lambda t: (
                f.track.append((time.monotonic(), t.lat, t.lon, t.alt_rel_home_m))
                if t.armed
                else None
            )
        )
        try:
            f.session = await asyncio.wait_for(fake.sessions.get(), 30)
            f.tasks += [
                asyncio.ensure_future(_read_acks(f, f.session)),
                asyncio.ensure_future(_commander(f)),
            ]
            yield f
        finally:
            stop.set()
            for task in f.tasks:
                task.cancel()
            await asyncio.gather(*f.tasks, return_exceptions=True)
            await ex.close()
            await onboard.close()
            await ground.close()


async def _wait_home(ground: Vehicle) -> None:
    for _ in range(300):
        if ground.telemetry.home is not None:
            return
        await asyncio.sleep(0.1)
    raise AssertionError("no HOME_POSITION")


async def _install(f: Flight, **overrides: Any) -> MissionCharter:
    """Upload the fence as the ground station does, then install the charter onboard."""
    home = f.ground.telemetry.home
    assert home is not None
    area = f.square(AREA_HALF_M)
    fence = f.square(AREA_HALF_M + 20 + FENCE_POLYGON_MARGIN_M)
    corners = [(lat, lon) for lon, lat in fence["coordinates"][0][:-1]]
    await upload(f.ground, fence_items(corners), origin=SendOrigin.GROUND)
    stored = await download(f.ground, FENCE, origin=SendOrigin.GROUND)
    data: dict[str, Any] = {
        "charter_id": new_id(),
        "mission_id": new_id(),
        "vehicle_id": "sim-01",
        "version": 1,
        "area": area,
        "home": LatLonAlt(lat=home.lat, lon=home.lon, alt_amsl_m=home.alt_amsl_m),
        "pilot_position": {"lat": home.lat, "lon": home.lon},
        "allowed_skills": ["takeoff", "goto", "survey_sector", "hold", "resume", "return_home"],
        "autonomy_mode": AutonomyMode.WITHIN_CHARTER,
        "sectors": [],
        "fence_hash": fence_hash(stored),
        "approved_by": "operator:pilot",
        "approved_at": dt.datetime.now(dt.UTC),
        **overrides,
    }
    charter = MissionCharter.model_validate(data)
    assert f.session is not None
    await f.session.send(CharterInstall(charter=charter))
    f.charter = charter
    await f.until(lambda: f.ex.charter == charter, 30, "charter installed")
    await f.until(f.ex.preflight_ok, 60, f"preflight ({f.ex.preflight})")
    return charter


async def _takeoff(f: Flight, alt: float = 40.0) -> SkillCommand:
    assert f.session is not None and f.charter is not None
    c = f.charter
    await f.session.send(
        ClearanceConfirmation(
            clearance_id=new_id(),
            mission_id=c.mission_id,
            vehicle_id="sim-01",
            charter_id=c.charter_id,
            charter_version=c.version,
            valid_for_ms=120_000,
            confirmed_by="operator:pilot",
            origin="local",
        ),
        ttl_ms=5000,
    )
    await f.until(lambda: f.ex.snapshot().clearance_available, 10, "clearance onboard")
    cmd = await f.order("takeoff", {"alt_m": alt})
    ack = await f.stage(cmd, {"running"}, 10)
    assert ack.stage == "running", ack
    done = await f.stage(cmd, {"done"}, 60)
    assert done.stage == "done", (done, f.ex.last_statustext)
    f.sim.rc.throttle = 1500  # in the air: the pilot centres the throttle stick
    return cmd


def _arms(f: Flight) -> int:
    return sum(1 for m in f.sent if m.get_type() == "COMMAND_LONG" and int(m.command) == 400)


# --- parameters -----------------------------------------------------------------------------


@pytest.mark.budget_s(150)
def test_params_loaded(sim: Sim) -> None:
    """Every safety parameter is in the flight controller, and covers the hard limits."""

    async def run() -> None:
        async with flight(sim) as f:
            expected = parse_parm(PARM.read_text())
            for name, want in expected.items():
                got = await f.onboard.read_param(name, origin=SendOrigin.STARTUP)
                assert got is not None, f"{name} missing in Copter-4.7.1"
                assert abs(got - want) <= 1e-3, f"{name}: {got} != {want}"
            # test_fc_params_cover_hard_limits
            alt = await f.onboard.read_param("FENCE_ALT_MAX", origin=SendOrigin.STARTUP)
            radius = await f.onboard.read_param("FENCE_RADIUS", origin=SendOrigin.STARTUP)
            speed = await f.onboard.read_param("LOIT_SPEED_MS", origin=SendOrigin.STARTUP)
            assert alt is not None and radius is not None and speed is not None
            assert alt >= ABSOLUTE_MAX_ALT_AGL_M + FENCE_ALT_MARGIN_M
            assert radius >= ABSOLUTE_MAX_RANGE_FROM_PILOT_M + FENCE_RADIUS_MARGIN_M
            assert speed >= ABSOLUTE_MAX_SPEED_MS
            # test_batt_thresholds_match_pct
            cap = await f.onboard.read_param("BATT_CAPACITY", origin=SendOrigin.STARTUP)
            low = await f.onboard.read_param("BATT_LOW_MAH", origin=SendOrigin.STARTUP)
            crit = await f.onboard.read_param("BATT_CRT_MAH", origin=SendOrigin.STARTUP)
            assert cap is not None and low is not None and crit is not None
            assert abs(low - round(0.15 * cap)) <= 1 and abs(crit - round(0.07 * cap)) <= 1

    asyncio.run(run())


# --- the demo flight ------------------------------------------------------------------------


@pytest.mark.budget_s(600)
def test_demo_flight(sim: Sim) -> None:
    """The demo flight: 40 m within 30 s, P1 within 5 m, >= 95 % coverage, home within 3 m.
    Also test_no_forbidden_mavlink: everything 1/191 sent is allowed, exactly one ARM."""

    async def run() -> None:
        async with flight(sim) as f:
            started = time.monotonic()
            charter = await _install(f)
            clicked = time.monotonic()
            takeoff = await _takeoff(f, 40)
            await f.until(lambda: f.ex.state == S.HOLD, 60, "takeoff done")
            assert time.monotonic() - clicked <= 30
            assert abs(f.onboard.telemetry.alt_rel_home_m - 40) <= 2
            assert (await f.stage(takeoff, {"done"}, 5)).stage == "done"
            lat, lon = f.at(*P1)
            goto = await f.order("goto", {"lat": lat, "lon": lon, "alt_m": 40, "speed_ms": 8})
            ack = await f.stage(goto, {"done"}, 120)
            assert ack.stage == "done", (ack, f.ex.vehicle_state())
            t = f.onboard.telemetry
            assert math.dist(f.plane.xy(t.lat, t.lon), P1) <= 5
            box = f.square(SURVEY_HALF_M, *P1)
            begin = len(f.track)
            survey = await f.order(
                "survey_sector", {"polygon": box, "spacing_m": 30, "speed_ms": 7}
            )
            ack = await f.stage(survey, {"done"}, 300)
            assert ack.stage == "done", (ack, f.ex.vehicle_state())
            searched = f.track[begin:]
            corners = [f.plane.xy(la, lo) for lo, la in box["coordinates"][0][:-1]]
            samples = [(*f.plane.xy(la, lo), alt) for _, la, lo, alt in searched]
            assert coverage_fraction(corners, samples, 40, 30) >= 0.95
            home = await f.order("return_home")
            ack = await f.stage(home, {"done"}, 300)
            assert ack.stage == "done", (ack, f.ex.vehicle_state())
            await f.until(lambda: not f.onboard.telemetry.armed, 60, "disarmed")
            t = f.onboard.telemetry
            assert math.dist(f.plane.xy(t.lat, t.lon), (0, 0)) <= 3
            assert all(inside_charter(la, lo, charter) for _, la, lo, _ in f.track)
            assert max(alt for *_, alt in f.track) <= 42
            assert time.monotonic() - started <= 600
            # test_no_forbidden_mavlink
            assert {m.get_type() for m in f.sent} <= _ALLOWED_TYPES
            commands = {int(m.command) for m in f.sent if m.get_type() == "COMMAND_LONG"}
            assert commands <= _ALLOWED_COMMANDS
            assert _arms(f) == 1

    asyncio.run(run())


# --- someone else takes over ------------------------------------------------------------------


@pytest.mark.budget_s(240)
def test_external_mode_change_held(sim: Sim) -> None:
    async def run() -> None:
        async with flight(sim) as f:
            await _install(f)
            await _takeoff(f, 30)
            await f.until(lambda: f.ex.state == S.HOLD, 60, "takeoff done")
            lat, lon = f.at(150, 0)
            await f.order("goto", {"lat": lat, "lon": lon, "alt_m": 30, "speed_ms": 3})
            await f.until(lambda: f.ex.state == S.TRANSIT, 10, "transit")
            # Someone else changes the mode (here the ground station's channel, without
            # any notice over SD-link).
            result = await f.ground.set_mode(CopterMode.LOITER, origin=SendOrigin.GROUND)
            assert result.name == "ACCEPTED"
            await f.until(lambda: f.ex.state == S.HELD, 5, "held")
            assert f.ex.held_by == "external"
            index = len(f.sent)
            refused: list[SkillCommand] = []
            for _ in range(30):
                refused.append(await f.order("goto", {"lat": lat, "lon": lon, "alt_m": 30}))
                await asyncio.sleep(1.0)
            for cmd in refused:
                ack = await f.stage(cmd, {"rejected"}, 10)
                assert ack.reason_codes == ["ONB_CONTROL_HELD"]
            assert f.steering_since(index) == []
            assert f.onboard.telemetry.fc_mode == CopterMode.LOITER.name

    asyncio.run(run())


# --- SD-link lost -------------------------------------------------------------------------


async def _link_loss(f: Flight, seconds: float) -> list[tuple[float, S]]:
    await _install(f)
    await _takeoff(f, 30)
    await f.until(lambda: f.ex.state == S.HOLD, 60, "takeoff done")
    lat, lon = f.at(150, 100)
    await f.order("goto", {"lat": lat, "lon": lon, "alt_m": 30, "speed_ms": 1})
    await f.until(lambda: f.ex.state == S.TRANSIT, 10, "transit")
    f.proxy.broken = True
    cut = time.monotonic()
    changes: list[tuple[float, S]] = []
    last = f.ex.state
    while time.monotonic() - cut < seconds:
        await asyncio.sleep(0.1)
        if f.ex.state != last:
            last = f.ex.state
            changes.append((time.monotonic() - cut, last))
    f.proxy.broken = False
    return changes


@pytest.mark.budget_s(180)
def test_sdlink_loss_10s(sim: Sim) -> None:
    async def run() -> None:
        async with flight(sim) as f:
            changes = await _link_loss(f, 10)
            assert changes == []
            assert f.ex.state == S.TRANSIT

    asyncio.run(run())


@pytest.mark.budget_s(300)
def test_sdlink_loss_60s(sim: Sim) -> None:
    async def run() -> None:
        async with flight(sim) as f:
            changes = await _link_loss(f, 60)
            hold_at = LINK_LOSS_DETECT_S + SDLINK_GRACE_S
            rtl_at = hold_at + SDLINK_HOLD_S
            states = [s for _, s in changes]
            assert states[:2] == [S.HOLD, S.RTL], changes
            assert abs(changes[0][0] - hold_at) <= 1, changes
            assert abs(changes[1][0] - rtl_at) <= 1, changes
            await f.until(lambda: f.ex.state == S.LANDED, 240, "landed")
            t = f.onboard.telemetry
            assert math.dist(f.plane.xy(t.lat, t.lon), (0, 0)) <= 3

    asyncio.run(run())


# --- the limits in the air ------------------------------------------------------------------


async def _new_version(f: Flight, **changes: Any) -> None:
    assert f.session is not None and f.charter is not None
    new = f.charter.model_copy(update={"version": f.charter.version + 1, **changes})
    await f.session.send(CharterInstall(charter=new))
    await f.until(lambda: f.ex.charter == new, 10, f"charter version {new.version}")
    f.charter = new


@pytest.mark.budget_s(420)
def test_limits_in_flight(sim: Sim) -> None:
    """A silent AI never sends the vehicle home: the step finishes and it holds. A lower
    height limit makes it sink below it; a smaller area beyond its buffer sends it home."""

    async def run() -> None:
        async with flight(sim) as f:
            await _install(f, on_ai_silent="return_home", ai_silent_timeout_s=5)
            await _takeoff(f, 40)
            await f.until(lambda: f.ex.state == S.HOLD, 60, "takeoff done")
            lat, lon = f.at(80, 0)
            goto = await f.order("goto", {"lat": lat, "lon": lon, "alt_m": 40, "speed_ms": 4})
            f.commander = False  # the AI falls silent during the step
            ack = await f.stage(goto, {"done"}, 120)
            assert ack.stage == "done", ack
            await asyncio.sleep(10)
            assert f.ex.state == S.HOLD and f.ex.hold_reason == "ai_silent"
            assert f.onboard.telemetry.fc_mode == CopterMode.LOITER.name
            f.commander = True
            # The pilot lowers the height limit in the air: down to 5 m below it.
            await _new_version(f, alt_max_agl_m=30, default_alt_agl_m=25)
            await f.until(lambda: f.ex.state == S.TRANSIT, 15, "descent starts")
            await f.until(lambda: f.ex.state == S.HOLD, 60, "descent done")
            assert f.ex.hold_reason == "limit"
            assert abs(f.onboard.telemetry.alt_rel_home_m - 25) <= 2
            # The area shrinks around home: the vehicle is now outside it and flies home.
            await _new_version(f, area=f.square(40))
            await f.until(lambda: f.ex.state == S.RTL, 15, "home after leaving the area")
            await f.until(
                lambda: f.onboard.telemetry.fc_mode in ("SMART_RTL", "RTL"), 5, "home mode"
            )
            await f.until(lambda: f.ex.state == S.LANDED, 240, "landed")
            t = f.onboard.telemetry
            assert math.dist(f.plane.xy(t.lat, t.lon), (0, 0)) <= 3

    asyncio.run(run())


# --- STOP and Resume --------------------------------------------------------------------------


@pytest.mark.budget_s(240)
def test_stop_then_release_out_of_brake(sim: Sim) -> None:
    """STOP: the ground's BRAKE and the notice; nothing steers from onboard until Resume,
    whose first own action is LOITER out of BRAKE."""

    async def run() -> None:
        async with flight(sim) as f:
            await _install(f)
            await _takeoff(f, 30)
            await f.until(lambda: f.ex.state == S.HOLD, 60, "takeoff done")
            lat, lon = f.at(120, 0)
            await f.order("goto", {"lat": lat, "lon": lon, "alt_m": 30, "speed_ms": 4})
            await f.until(lambda: f.ex.state == S.TRANSIT, 10, "transit")
            assert f.session is not None
            result = await f.ground.set_mode(CopterMode.BRAKE, origin=SendOrigin.GROUND)
            assert result.name == "ACCEPTED"
            await f.session.send(
                OperatorOverrideNotice(
                    override_id=new_id(),
                    kind="stop",
                    action_seq=1,
                    issuer_kind="operator",
                    operator_id="operator:pilot",
                )
            )
            await f.until(lambda: f.ex.state == S.HELD, 5, "held")
            index = len(f.sent)
            resume = await f.order("resume")
            ack = await f.stage(resume, {"rejected"}, 10)
            assert ack.reason_codes == ["ONB_CONTROL_HELD"]
            await asyncio.sleep(10)
            assert f.steering_since(index) == []
            assert f.onboard.telemetry.fc_mode == CopterMode.BRAKE.name
            await f.session.send(OperatorRelease(action_seq=1, operator_id="operator:pilot"))
            await f.until(lambda: f.ex.state == S.HOLD, 10, "released")
            await f.until(
                lambda: f.onboard.telemetry.fc_mode == CopterMode.LOITER.name, 10, "LOITER"
            )
            resume = await f.order("resume")
            ack = await f.stage(resume, {"done"}, 120)
            assert ack.stage == "done", ack

    asyncio.run(run())
