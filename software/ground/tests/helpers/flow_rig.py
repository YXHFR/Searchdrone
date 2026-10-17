"""A whole ground station for the mission-flow tests, with fakes only at the edges.

Real: the database (a migrated ``searchdrone_test_*``), ``Repo``/``FlowRepo``, the
mission flow, the gate, the emergency path (``GroundMav``, ``Latches``,
``VehicleActions``), SD-link (``GroundLink`` on a free loopback port) and both APIs.
Fake: the flight controller (``sd_mav.FakeFlightController``, real MAVLink bytes through
the ground allowlist) and the onboard computer (``FakeOnboard``: an SD-link client that
answers charters and orders the way ``sd-onboard`` does and sends snapshots).

Runs on real time (``MonotonicClock``): the database answers in real time anyway.
"""

from __future__ import annotations

import asyncio
import contextlib
import datetime as dt
import json
from collections.abc import AsyncIterator, Callable
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Protocol

import httpx
from sqlalchemy import URL

from sd_ground.app import commander_asgi, create_commander_app, create_panel_app, panel_asgi
from sd_ground.auth import BrainToken, SessionRecord, new_token
from sd_ground.auth import hash_token as hash_session_token
from sd_ground.config import GroundConfig, Loaded
from sd_ground.context import Ground
from sd_ground.db.engine import make_engine
from sd_ground.db.flow import FlowRepo
from sd_ground.db.repo import Repo
from sd_ground.link import GroundLink
from sd_ground.mav.thread import GroundMav, ThreadSafeMav
from sd_ground.services.latches import Latches
from sd_ground.services.missions import MissionFlow
from sd_ground.services.telemetry import TelemetryService
from sd_ground.services.vehicle_actions import VehicleActions
from sd_link.client import LinkClient
from sd_link.session import Closed, Inbound, LinkSession
from sd_link.tokens import TokenStore, generate_token, hash_token
from sd_mav.connection import Role
from sd_mav.fake import FakeFlightController, FakeVehicle
from sd_obs.clock import MonotonicClock
from sd_safety.targets import Plane
from sd_schemas.base import LatLonAlt, new_id
from sd_schemas.executor import ExecutorState
from sd_schemas.link import (
    SCHEMA_MAJOR,
    CharterAck,
    CharterInstall,
    ClearanceConfirmation,
    Hello,
    HelloAck,
    OperatorOverrideNotice,
    OperatorRelease,
    SkillCommand,
    StateSnapshot,
)

__all__ = ["BRAIN", "VID", "FakeOnboard", "H", "Rig", "WallClock", "rig"]

BRAIN = "brain-token-for-flow-tests-0123456789ab"
H = {"X-SD-Client": "panel", "Content-Type": "application/json"}
VID = "sim-01"


class WallClock(MonotonicClock):
    """The real clock, whose wall time a test may move forward (``skew_s``) to let a
    card or a chat line age without waiting. Monotonic time is never moved."""

    def __init__(self) -> None:
        super().__init__()
        self.skew_s = 0.0

    def utc_now(self) -> dt.datetime:
        return super().utc_now() + dt.timedelta(seconds=self.skew_s)


@dataclass
class FakeOnboard:
    """Answers like ``sd-onboard``: installs every charter, runs every order to ``done``
    (unless told otherwise) and sends a snapshot at 5 Hz."""

    clock: MonotonicClock
    url: str
    token: str
    executor_state: ExecutorState = ExecutorState.IDLE
    hold_reason: str | None = None
    held_by: str | None = None
    stop_latch: bool = False
    accept_charters: bool = True
    order_outcome: str = "done"  # or "running" (stays), "rejected"
    reject_gate: Any = None  # with "rejected": the onboard gate's own decision
    boot_id: int = 1
    state_seq: int = 0
    last_transition_seq: int = 0
    charters: list[CharterInstall] = field(default_factory=list[CharterInstall])
    orders: list[SkillCommand] = field(default_factory=list[SkillCommand])
    clearances: list[ClearanceConfirmation] = field(default_factory=list[ClearanceConfirmation])
    notices: list[Any] = field(default_factory=list[Any])
    active_command_id: Any = None
    progress_pct: float | None = None
    session: LinkSession | None = None
    _stop: asyncio.Event = field(default_factory=asyncio.Event)
    _tasks: list[asyncio.Future[Any]] = field(default_factory=list[asyncio.Future[Any]])

    def set_state(self, state: ExecutorState, **kw: Any) -> None:
        if state != self.executor_state:
            self.last_transition_seq = self.state_seq + 1
        self.executor_state = state
        for k, v in kw.items():
            setattr(self, k, v)

    def hello(self) -> Hello:
        return Hello(
            schema_major=SCHEMA_MAJOR,
            boot_id=self.boot_id,
            executor_state=self.executor_state,
            charter_id=None,
            charter_version=None,
            fc_firmware="ArduCopter V4.7.1",
            operator_latch=None,
            last_action_seq=0,
        )

    def snapshot(self, home: tuple[float, float]) -> StateSnapshot:
        self.state_seq += 1
        charter = self.charters[-1].charter if self.charters else None
        return StateSnapshot.model_validate(
            {
                "state_seq": self.state_seq,
                "state_epoch": self.boot_id,
                "last_transition_state_seq": max(self.last_transition_seq, 1),
                "executor_state": self.executor_state,
                "hold_reason": self.hold_reason,
                "held_by": self.held_by,
                "control_suspended": False,
                "expected_mode": None,
                "flight_mode": "LOITER",
                "armed": False,
                "position": LatLonAlt(lat=home[0], lon=home[1], alt_amsl_m=584.0),
                "alt_rel_home_m": 0.0,
                "groundspeed_ms": 0.0,
                "heading_deg": 0.0,
                "battery_pct": 95.0,
                "active_command_id": self.active_command_id,
                "active_skill": None,
                "progress_pct": self.progress_pct,
                "resume_available": False,
                "charter_id": None if charter is None else charter.charter_id,
                "charter_version": None if charter is None else charter.version,
                "clearance_available": bool(self.clearances),
                "preflight_ok": charter is not None,
                "preflight": {},
                "limits_active": [],
                "stop_latch": self.stop_latch,
                "battery_rtl_active": False,
                "sector_progress": {},
                "tick_ms_p95": 5.0,
            }
        )

    async def start(self, home: tuple[float, float]) -> None:
        client = LinkClient(
            url=self.url,
            token=self.token,
            vehicle_id=VID,
            make_hello=self.hello,
            on_session=self._on_session,
            clock=self.clock,
        )
        self._home = home
        self._tasks.append(asyncio.ensure_future(client.run_forever(self._stop)))

    async def close(self) -> None:
        self._stop.set()
        if self.session is not None:
            with contextlib.suppress(Exception):
                await self.session.close()
        for t in self._tasks:
            t.cancel()
        await asyncio.gather(*self._tasks, return_exceptions=True)

    async def _on_session(self, session: LinkSession, ack: HelloAck) -> None:
        self.session = session
        snaps = asyncio.ensure_future(self._snapshots(session))
        try:
            while True:
                event = await session.next_event()
                if isinstance(event, Closed):
                    return
                if isinstance(event, Inbound):
                    await self._handle(session, event)
        finally:
            snaps.cancel()
            self.session = None

    async def _snapshots(self, session: LinkSession) -> None:
        while True:
            snap = self.snapshot(self._home)  # a broken fake must fail loudly
            with contextlib.suppress(Exception):
                await session.send(snap)
            await asyncio.sleep(0.2)

    async def _handle(self, session: LinkSession, inbound: Inbound) -> None:
        body = inbound.body
        if isinstance(body, CharterInstall):
            self.charters.append(body)
            if self.accept_charters:
                self.set_state(ExecutorState.PREFLIGHT)
            await session.send(
                CharterAck(
                    charter_id=body.charter.charter_id,
                    charter_version=body.charter.version,
                    accepted=self.accept_charters,
                    reason_codes=[] if self.accept_charters else ["ONB_CHARTER_IN_FLIGHT"],
                )
            )
        elif isinstance(body, ClearanceConfirmation):
            self.clearances.append(body)
            await session.ack(inbound.envelope.msg_id, "done")
        elif isinstance(body, SkillCommand):
            self.orders.append(body)
            if self.order_outcome == "rejected":
                codes = (
                    ["ONB_STALE_ORDER"]
                    if self.reject_gate is None
                    else self.reject_gate.reason_codes
                )
                await session.ack(
                    body.command_id, "rejected", reason_codes=codes, gate=self.reject_gate
                )
                return
            await session.ack(body.command_id, "accepted")
            await session.ack(body.command_id, "running")
            self.active_command_id = body.command_id
            if self.order_outcome == "done":
                await session.ack(body.command_id, "done")
                self.active_command_id = None
        elif isinstance(body, OperatorOverrideNotice):
            self.notices.append(body)
            if body.kind in ("stop", "watchdog_stop"):
                self.stop_latch = True
                self.set_state(ExecutorState.HELD, held_by="operator_stop")
        elif isinstance(body, OperatorRelease):
            self.notices.append(body)
            self.stop_latch = False
            self.set_state(ExecutorState.HOLD, hold_reason="operator_released", held_by=None)


@dataclass
class Rig:
    g: Ground
    flow: MissionFlow
    fc: FakeFlightController | None
    core: GroundMav
    actions: VehicleActions
    link: GroundLink
    onboard: Any
    panel: httpx.AsyncClient
    commander: httpx.AsyncClient
    clock: WallClock
    plane: Plane
    # Whether the pilot's panel is open (its live connection pings); a test closes it.
    presence: list[bool] = field(default_factory=lambda: [True])

    @property
    def home(self) -> tuple[float, float]:
        return self.plane.lat0, self.plane.lon0

    def at(self, east: float, north: float) -> tuple[float, float]:
        return self.plane.latlon(east, north)

    def square(self, half: float, east: float = 0.0, north: float = 0.0) -> dict[str, Any]:
        ring = []
        for x, y in [(-half, -half), (half, -half), (half, half), (-half, half), (-half, -half)]:
            lat, lon = self.at(x + east, y + north)
            ring.append([lon, lat])
        return {"type": "Polygon", "coordinates": [ring]}

    async def until(self, cond: Callable[[], Any], limit_s: float, what: str) -> None:
        end = asyncio.get_running_loop().time() + limit_s
        while True:
            value = cond()
            if asyncio.iscoroutine(value):
                value = await value
            if value:
                return
            if asyncio.get_running_loop().time() > end:
                raise AssertionError(f"{what}: not within {limit_s} s")
            await asyncio.sleep(0.05)

    async def mission(self, mission_id: str) -> dict[str, Any]:
        res = await self.panel.get(f"/api/v1/missions/{mission_id}")
        assert res.status_code == 200, res.text
        return res.json()

    async def status(self, mission_id: str) -> str:
        return str((await self.mission(mission_id))["mission"]["status"])

    async def wait_status(self, mission_id: str, status: str, limit_s: float = 10.0) -> None:
        async def check() -> bool:
            return await self.status(mission_id) == status

        try:
            await self.until(check, limit_s, f"mission {status}")
        except AssertionError as exc:
            cmds = (await self.panel.get(f"/api/v1/missions/{mission_id}/commands")).json()
            seen = [(c["skill"], c["status"], c["status_reason"]) for c in cmds]
            raise AssertionError(
                f"{exc}; now {await self.status(mission_id)}; orders {seen}"
            ) from None

    def fly(self, alt: float = 40.0) -> None:
        """The fake flight controller is in the air (no physics in these tests)."""
        if self.fc is None:
            raise RuntimeError("only with the fake flight controller")
        self.fc.armed = True
        self.fc.alt_rel = alt

    def land(self) -> None:
        if self.fc is None:
            raise RuntimeError("only with the fake flight controller")
        self.fc.armed = False
        self.fc.alt_rel = 0.0

    def bus_types(self) -> list[str]:
        return [e.type for e in self.g.bus.replay(self.g.bus.server_epoch, 0) or []]

    async def open_mission(self) -> str:
        res = await self.panel.post(f"/api/v1/vehicles/{VID}/missions/open", json={}, headers=H)
        assert res.status_code == 200, res.text
        return str(res.json()["id"])

    async def approve(self, approval_id: str, version: int = 1, **extra: Any) -> httpx.Response:
        return await self.panel.post(
            f"/api/v1/approvals/{approval_id}/decision",
            json={"decision": "approve", "subject_version": version, **extra},
            headers=H,
        )

    async def ready_mission(self, half: float = 150.0) -> str:
        """Open, propose (as the pilot), approve: READY with fence and install."""
        mid = await self.open_mission()
        res = await self.panel.post(
            f"/api/v1/missions/{mid}/charters",
            json={"area": self.square(half), "reason_sv": "Test"},
            headers=H,
        )
        assert res.status_code == 201, res.text
        charter = res.json()
        res = await self.approve(str(charter["approval_id"]), charter["version"])
        assert res.status_code == 200, res.text
        await self.wait_status(mid, "READY", 30)
        await self.wait_snapshot(ExecutorState.PREFLIGHT)
        return mid

    async def wait_snapshot(self, state: ExecutorState, limit_s: float = 5.0) -> None:
        """Until the ground has a snapshot in ``state`` (an order's ``based_on`` then
        matches the onboard state, as the AI's would)."""

        def seen() -> bool:
            snap, _ = self.flow.snapshot(VID)
            return snap is not None and snap.executor_state is state

        await self.until(seen, limit_s, f"snapshot {state}")

    async def clearance(self, mission_id: str) -> httpx.Response:
        return await self.panel.post(f"/api/v1/missions/{mission_id}/clearance", json={}, headers=H)


def app_url(name: str, password: str) -> URL:
    return URL.create(
        "postgresql+asyncpg",
        username="sd_app",
        password=password,
        host="127.0.0.1",
        port=5433,
        database=name,
    )


async def _harness_token(g: Ground) -> str:
    """A pilot CLI token named ``ci-harness``, as ``sd-ground token create`` makes it: the
    harness approves and confirms only through the panel API, as ``operator:ci-harness``."""
    if g.repo is None:
        raise RuntimeError("no database")
    token = new_token()
    now = dt.datetime.now(dt.UTC)
    rec = SessionRecord(
        id="",
        kind="token",
        role="pilot",
        issuer_ref="operator:ci-harness",
        auth_mode="cli",
        token_sha256=hash_session_token(token),
        created_at=now,
        last_used_at=now,
        expires_at=now + dt.timedelta(hours=24),
        name="ci-harness",
    )
    await g.repo.save_session(rec)
    g.sessions.load(await g.repo.sessions())
    return token


async def _operator_present(g: Ground, presence: list[bool]) -> None:
    """The pilot keeps the panel open: its live connection pings once a second."""
    while True:
        if presence[0]:
            g.operator_seen_mono = g.clock.monotonic()
        await asyncio.sleep(1.0)


async def _flush_events(g: Ground) -> None:
    """Writes the queued events every 0.1 s, as the app's background loop does."""
    while True:
        await g.flush_events()
        await asyncio.sleep(0.1)


class Onboard(Protocol):
    """The onboard side of a rig: ``FakeOnboard`` or the real executor against SITL."""

    async def start(self, home: tuple[float, float]) -> None: ...

    async def close(self) -> None: ...


@contextlib.asynccontextmanager
async def rig(db_name: str, app_password: str, tmp_path: Path, **config: Any) -> AsyncIterator[Rig]:
    """The ground station against the fake flight controller and ``FakeOnboard``."""
    mav_clock = MonotonicClock()
    fc = FakeFlightController(mav_clock)
    onboard_vehicle = FakeVehicle(mav_clock, fc=fc)  # the onboard heartbeat (1/191)
    await onboard_vehicle.start()
    async with ground_stack(
        db_name,
        app_password,
        tmp_path,
        vehicle=FakeVehicle(mav_clock, fc=fc, role=Role.GROUND),
        mav_clock=mav_clock,
        onboard=lambda clock, url, token: FakeOnboard(clock, url, token),
        fc=fc,
        **config,
    ) as r:
        try:
            yield r
        finally:
            await onboard_vehicle.close()


@contextlib.asynccontextmanager
async def ground_stack(
    db_name: str,
    app_password: str,
    tmp_path: Path,
    *,
    vehicle: Any,
    mav_clock: Any,
    onboard: Callable[[WallClock, str, str], Onboard],
    fc: FakeFlightController | None = None,
    sdlink_bind: str = "127.0.0.1:0",
    **config: Any,
) -> AsyncIterator[Rig]:
    """The ground station's mission flow on ``vehicle`` (250/190), with its database, the
    real SD-link server, the panel and commander APIs; ``onboard`` connects to it."""
    clock = WallClock()
    loaded = Loaded(
        config=GroundConfig.model_validate(
            {"data_dir": str(tmp_path / "data"), "auth": {"cli_may_confirm": True}, **config}
        ),
        root=tmp_path,
        env={},
    )
    g = Ground(loaded=loaded, clock=clock, brain_token=BrainToken(BRAIN), version="0.0.0")
    engine = make_engine(app_url(db_name, app_password))
    g.repo = Repo(engine)
    g.db_ok = True
    latches = Latches(clock)
    actions = VehicleActions(clock, latches, vehicle_ids=[VID])
    core = GroundMav(
        vehicle,
        mav_clock,
        vehicle_id=VID,
        latches=latches,
        on_report=actions.on_report,
        on_event=actions.on_event,
        system_action=actions.system,
    )
    mav = ThreadSafeMav(core, asyncio.get_running_loop())
    actions.mav = mav
    token = generate_token()
    link = GroundLink(
        clock=clock,
        latches=latches,
        tokens=TokenStore({VID: hash_token(token)}),
        vehicle_ids=[VID],
        bind=sdlink_bind,
        commander_state=lambda: g.brain_state,
    )
    flow = MissionFlow(g, FlowRepo(engine), link=link, mav=mav, actions=actions)
    link.handler = flow
    actions.notices = link
    g.vehicle_link = mav
    g.sdlink = link
    g.emergency = actions
    g.vehicle_actions = actions
    g.flow = flow
    await core.start()
    await flow.start()
    await link.start()
    presence = [True]
    side = onboard(clock, f"ws://127.0.0.1:{link.port}/sdlink/v1/control", token)
    tasks = [
        asyncio.ensure_future(flow.run()),
        asyncio.ensure_future(actions.pump(g)),
        asyncio.ensure_future(TelemetryService(g).run()),
        asyncio.ensure_future(_flush_events(g)),
        asyncio.ensure_future(_operator_present(g, presence)),
    ]
    panel = httpx.AsyncClient(
        transport=httpx.ASGITransport(app=panel_asgi(g, create_panel_app(g))),  # pyright: ignore[reportArgumentType]
        base_url="http://127.0.0.1:8080",
    )
    commander = httpx.AsyncClient(
        transport=httpx.ASGITransport(app=commander_asgi(g, create_commander_app(g))),  # pyright: ignore[reportArgumentType]
        base_url="http://127.0.0.1:7700",
        headers={"Authorization": f"Bearer {BRAIN}"},
    )
    try:
        panel.headers["Authorization"] = f"Bearer {await _harness_token(g)}"
        home = None
        for _ in range(300):
            snap = core.snapshot()
            if snap is not None and snap.connected and snap.telemetry.home is not None:
                home = snap.telemetry.home
                break
            await asyncio.sleep(0.05)
        if home is None:
            raise AssertionError("no HOME_POSITION from the flight controller")
        r = Rig(
            g,
            flow,
            fc,
            core,
            actions,
            link,
            side,
            panel,
            commander,
            clock,
            Plane(home.lat, home.lon),
            presence,
        )
        await side.start(r.home)
        await r.until(lambda: link.up(VID), 30, "SD-link up")
        yield r
    finally:
        for t in tasks:
            t.cancel()
        await asyncio.gather(*tasks, return_exceptions=True)
        await side.close()
        await link.stop()
        await panel.aclose()
        await commander.aclose()
        await core.close()
        await g.repo.close()


def dump(data: Any) -> str:
    return json.dumps(data, default=str)[:2000]


__all__ += ["Onboard", "app_url", "dump", "ground_stack", "new_id"]
