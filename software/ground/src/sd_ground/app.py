"""Builds the apps and starts ``sd-ground serve`` in the order the safety design needs.

Start order: config check, logging, the MAVLink thread (GCS heartbeat, STOP and FLY
HOME, the companion watchdog, the battery guard; it runs before and without the
database), database (wait up to 30 s; wrong schema version stops the server; no database
= safe start mode with sessions in memory), background tasks, control socket, commander
API (:7700, main loop), then the ingress thread with the panel listeners (:8080 and, in
LAN mode, :8443 with TLS), and finally the login link. The MAVLink thread stops last.
The SD-link server plugs in through ``sd_ground.sources``.
"""

from __future__ import annotations

import asyncio
import contextlib
import signal
import sys
from collections.abc import Awaitable, Callable, MutableMapping
from pathlib import Path
from typing import Any

import uvicorn
from fastapi import FastAPI
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel

from sd_ground.api import commander as commander_api
from sd_ground.api import live as live_api
from sd_ground.api import panel as panel_api
from sd_ground.api import search as search_api
from sd_ground.api import tiles as tiles_api
from sd_ground.api import video_proxy as video_api
from sd_ground.api.deps import install_error_handlers
from sd_ground.auth import BrainToken
from sd_ground.config import Loaded, check, limit_caps, parse_bind
from sd_ground.context import Ground, lan_addresses
from sd_ground.control_socket import ControlSocket, socket_path
from sd_ground.db.engine import make_engine, with_password
from sd_ground.db.flow import FlowRepo
from sd_ground.db.repo import Repo
from sd_ground.db.setup import head_revision
from sd_ground.ingress import IngressApp, IngressThread, MainLoopBridge
from sd_ground.link import GroundLink
from sd_ground.liveness import LoopLiveness
from sd_ground.logging import get_logger, setup_logging
from sd_ground.mav.thread import GroundMav, HeartbeatGate, MavThread, vehicle_events
from sd_ground.services.latches import Latches
from sd_ground.services.missions import MissionFlow
from sd_ground.services.telemetry import TelemetryService
from sd_ground.services.vehicle_actions import OperatorPresence, VehicleActions
from sd_ground.services.video import VideoMonitor
from sd_ground.web import COMMANDER_BODY_LIMIT, LoopbackGuard, PanelGuard

__all__ = [
    "Background",
    "commander_asgi",
    "create_commander_app",
    "create_panel_app",
    "openapi_documents",
    "panel_asgi",
    "read_brain_token",
    "read_search_token",
    "serve",
]

API_VERSION = "1"  # the /api/v1 and /commander/v1 contract, not the package version
DB_CHECK_S = 2.0
SESSION_RELOAD_S = 10.0
log = get_logger("sd_ground")

Scope = MutableMapping[str, Any]
ASGIApp = Callable[[Scope, Any, Any], Awaitable[None]]


def create_panel_app(g: Ground) -> FastAPI:
    app = FastAPI(
        title="Searchdrone panel API",
        version=API_VERSION,
        openapi_url="/api/v1/openapi.json",
        docs_url=None,
        redoc_url=None,
    )
    app.state.ground = g
    install_error_handlers(app)
    app.include_router(panel_api.router)
    app.include_router(live_api.router)
    app.include_router(tiles_api.router)
    app.include_router(video_api.router)
    static = g.loaded.path(g.config.http.static_dir)
    if (static / "index.html").is_file():
        app.mount("/", StaticFiles(directory=static, html=True), name="panel")
    return app


def create_commander_app(g: Ground) -> FastAPI:
    app = FastAPI(
        title="Searchdrone commander API",
        version=API_VERSION,
        openapi_url="/commander/v1/openapi.json",
        docs_url=None,
        redoc_url=None,
    )
    app.state.ground = g
    app.state.commander_hub = commander_api.CommanderHub(g)
    install_error_handlers(app)
    app.include_router(commander_api.router)
    # The search service shares the loopback listener with its own token; its routes are
    # documented on their own (``create_search_doc``), so the commander's list stays its own.
    app.include_router(search_api.router, include_in_schema=False)
    _with_schemas(app, commander_api.RAW_BODY_MODELS)
    return app


def create_search_doc(g: Ground) -> FastAPI:
    """The search API on its own, for its OpenAPI document (served by the commander app)."""
    app = FastAPI(
        title="Searchdrone search API",
        version=API_VERSION,
        openapi_url=None,
        docs_url=None,
        redoc_url=None,
    )
    app.state.ground = g
    app.include_router(search_api.router)
    return app


def _with_schemas(app: FastAPI, models: tuple[type[BaseModel], ...]) -> None:
    """Every ``$ref`` in the document resolves: add the raw-body models' schemas."""
    build = app.openapi

    def openapi() -> dict[str, Any]:
        if app.openapi_schema:
            return app.openapi_schema
        doc = build()
        schemas = doc.setdefault("components", {}).setdefault("schemas", {})
        for model in models:
            js = model.model_json_schema(ref_template="#/components/schemas/{model}")
            for name, sub in js.pop("$defs", {}).items():
                schemas.setdefault(name, sub)
            schemas.setdefault(model.__name__, js)
        app.openapi_schema = doc
        return doc

    app.openapi = openapi  # type: ignore[method-assign]


def _labelled(app: ASGIApp, listener: str) -> ASGIApp:
    async def labelled(scope: Scope, receive: Any, send: Any) -> None:
        scope["sd_listener"] = listener
        await app(scope, receive, send)

    return labelled


def panel_asgi(
    g: Ground,
    main_app: ASGIApp,
    *,
    main_loop: asyncio.AbstractEventLoop | None = None,
    listener: str = "local",
) -> PanelGuard:
    """The full panel listener: guard -> ingress (STOP / HOME) -> bridge -> main app."""
    forward: ASGIApp = main_app if main_loop is None else MainLoopBridge(main_app, main_loop)
    inner = IngressApp(sessions=g.sessions, emergency=g.emergency, forward=forward)
    return PanelGuard(
        _labelled(inner, listener),
        allowed_hosts=g.allowed_hosts,
        is_commander_token=g.brain_token.matches,
        profile=g.config.profile,
        is_search_token=g.search_token.matches,
    )


def commander_asgi(g: Ground, app: ASGIApp) -> LoopbackGuard:
    _, port = parse_bind(g.config.http.commander_bind)
    return LoopbackGuard(app, port=port, body_limit=COMMANDER_BODY_LIMIT)


def read_brain_token(loaded: Loaded) -> str | None:
    return _read_token(loaded.path(loaded.config.auth.brain_token_file))


def read_search_token(loaded: Loaded) -> str | None:
    return _read_token(loaded.path(loaded.config.auth.search_token_file))


def _read_token(path: Path) -> str | None:
    if not path.is_file():
        return None
    value = path.read_text(encoding="utf-8").strip()
    return value or None


class Background:
    """The main loop's housekeeping: database health, sessions, events, telemetry."""

    def __init__(self, g: Ground, liveness: LoopLiveness | None = None) -> None:
        self.g = g
        self.liveness = liveness
        self.telemetry = TelemetryService(g)
        self.video = VideoMonitor(g)
        self._next_reload = 0.0

    async def db_tick(self) -> None:
        g = self.g
        if g.repo is None:
            return
        was_ok = g.db_ok
        try:
            await g.repo.ping()
            g.db_ok = True
        except Exception as exc:
            if was_ok:
                log.error("system.db_down", error=type(exc).__name__)
                g.add_event("system.db_down", severity="critical")
            g.db_ok = False
            return
        if not was_ok:
            log.info("system.db_up")
            g.add_event("system.db_up")
        for sha in g.sessions.unsaved_revocations():
            try:
                await g.repo.revoke_session(sha)
                g.sessions.revocation_saved(sha)
            except Exception:  # noqa: S112 - retried on the next tick
                continue
        for rec in g.sessions.pending():
            try:
                await g.repo.save_session(rec)
                g.sessions.mark_persisted(rec.token_sha256)
            except Exception:  # noqa: S112 - retried on the next tick
                continue
        now = g.clock.monotonic()
        if now >= self._next_reload:
            self._next_reload = now + SESSION_RELOAD_S
            with contextlib.suppress(Exception):
                await g.repo.touch_sessions([r for r in g.sessions.all() if not r.pending_persist])
                g.sessions.load(await g.repo.sessions())
        await g.flush_events()

    async def run_db(self) -> None:
        while True:
            await self.db_tick()
            await self.g.clock.sleep(DB_CHECK_S)

    def tasks(self) -> list[asyncio.Task[None]]:
        return [
            asyncio.create_task(self.run_db(), name="db"),
            asyncio.create_task(self.telemetry.run(), name="telemetry"),
            asyncio.create_task(self.video.run(), name="video"),
        ]


async def _beat(liveness: LoopLiveness, name: str) -> None:
    """The main loop's sign of life for the MAVLink thread's heartbeat gate."""
    while True:
        liveness.beat(name)
        await asyncio.sleep(1.0)


async def _start_db(g: Ground) -> None:
    """Connect (≤ 30 s). Wrong schema version stops the server; no database = safe mode."""
    if g.repo is None:
        return
    loop = asyncio.get_running_loop()
    deadline = loop.time() + g.config.db.connect_timeout_s
    while True:
        try:
            revision = await g.repo.schema_revision()
            break
        except Exception as exc:
            if loop.time() >= deadline:
                log.error("db.unreachable_safe_mode", error=type(exc).__name__)
                g.add_event("system.db_down", severity="critical")
                return
            await asyncio.sleep(1.0)
    if revision != head_revision():
        raise SystemExit("Databasen är inte uppdaterad. Kör: just migrate")
    g.db_ok = True
    g.sessions.load(await g.repo.sessions())


def _uvicorn(app: Any, bind: str, **extra: Any) -> uvicorn.Server:
    host, port = parse_bind(bind)
    config = uvicorn.Config(
        app,
        host=host,
        port=port,
        lifespan="off",
        log_level="warning",
        access_log=False,
        server_header=False,
        proxy_headers=False,  # nothing sits in front of this server; never trust X-Forwarded-*
        forwarded_allow_ips=[],
        date_header=False,
        **extra,
    )
    return uvicorn.Server(config)


async def _serve(loaded: Loaded) -> int:
    errors = check(loaded)
    if errors:
        for line in errors:
            print(line, file=sys.stderr)  # noqa: T201 - operator-facing CLI output
        return 1
    cfg = loaded.config
    setup_logging(loaded.data_dir / "logs" / "sd-ground.jsonl")
    from sd_obs.clock import MonotonicClock

    token = read_brain_token(loaded)
    g = Ground(
        loaded=loaded,
        clock=MonotonicClock(),
        brain_token=BrainToken(token),
        search_token=BrainToken(read_search_token(loaded)),
    )
    if cfg.lan.enabled:
        g.lan_hosts = lan_addresses()
    liveness = LoopLiveness()
    main_beat = asyncio.create_task(_beat(liveness, "main"), name="main-liveness")
    try:
        mav_thread, actions = await start_emergency_path(g, liveness)
    except BaseException:
        main_beat.cancel()
        raise
    engine = make_engine(with_password(cfg.db.url, loaded.secret("SD_GROUND__DB_APP_PASSWORD")))
    g.repo = Repo(engine)
    await _start_db(g)
    link, flow = await start_mission_flow(g, FlowRepo(engine), actions, mav_thread)
    background = Background(g, liveness)
    tasks = [*background.tasks(), main_beat]
    presence = OperatorPresence(actions, mission_active=flow.mission_active)
    tasks.append(asyncio.create_task(actions.pump(g), name="vehicle-actions"))
    tasks.append(asyncio.create_task(presence.run(g), name="operator-presence"))
    tasks.append(asyncio.create_task(flow.run(), name="mission-flow"))
    main_loop = asyncio.get_running_loop()

    control = ControlSocket(
        socket_path(loaded.data_dir), login_link=g.login_link, pair=g.pair, emergency=g.emergency
    )
    control.start()

    commander_app = create_commander_app(g)
    commander_server = _uvicorn(commander_asgi(g, commander_app), cfg.http.commander_bind)
    panel_app = create_panel_app(g)
    servers = [_uvicorn(panel_asgi(g, panel_app, main_loop=main_loop), cfg.http.panel_bind)]
    if cfg.lan.enabled:
        tls = loaded.path(cfg.lan.tls_dir)
        servers.append(
            _uvicorn(
                panel_asgi(g, panel_app, main_loop=main_loop, listener="lan"),
                cfg.lan.bind,
                ssl_certfile=str(tls / "server.crt"),
                ssl_keyfile=str(tls / "server.key"),
            )
        )
    ingress = IngressThread(servers, beat=lambda: liveness.beat("ingress"))
    commander_task = asyncio.create_task(commander_server.serve())
    ingress.start()
    await asyncio.to_thread(ingress.started_event.wait, 10)

    g.add_event("system.start", payload={"version": g.version, "profile": cfg.profile})
    log.info("system.start", version=g.version, profile=cfg.profile)
    if g.test_rig:
        log.warning("auth.test_rig", detail="cli_may_confirm = true: bara för testriggen")
    _code, url = g.login_link()
    print(f"Searchdrone markstation körs. Öppna panelen: {url}", flush=True)  # noqa: T201
    if cfg.lan.enabled:
        _c, pin, lan_url = g.pair()
        print(f"Mobil i samma nät: {lan_url}  PIN: {pin}", flush=True)  # noqa: T201

    stop = asyncio.Event()
    for sig in (signal.SIGINT, signal.SIGTERM):
        main_loop.add_signal_handler(sig, stop.set)
    await stop.wait()

    log.info("system.stop")
    g.add_event("system.stop")
    ingress.stop()
    commander_server.should_exit = True
    await asyncio.wait_for(asyncio.to_thread(ingress.join, 5), 6)
    with contextlib.suppress(Exception):
        await asyncio.wait_for(commander_task, 5)
    with contextlib.suppress(Exception):
        await asyncio.wait_for(g.flush_events(), 5)
    for task in tasks:
        task.cancel()
    await asyncio.gather(*tasks, return_exceptions=True)
    control.stop()
    with contextlib.suppress(Exception):
        await asyncio.wait_for(link.stop(), 5)
    await g.repo.close()
    await asyncio.to_thread(mav_thread.stop)  # the GCS heartbeat ends last
    return 0


async def start_emergency_path(
    g: Ground, liveness: LoopLiveness
) -> tuple[MavThread, VehicleActions]:
    """Start the MAVLink thread and wire STOP / FLY HOME into ``g``. Exit 1 on failure."""
    from sd_mav.connection import GCS_COMPID, GCS_SYSID, Role, connect
    from sd_mav.vehicle import Vehicle
    from sd_obs.clock import MonotonicClock

    cfg = g.config
    latches = Latches(g.clock)
    actions = VehicleActions(g.clock, latches, vehicle_ids=[cfg.default_vehicle_id])

    def make_core() -> GroundMav:
        clock = MonotonicClock()
        conn = connect(
            cfg.mavlink.listen,
            role=Role.GROUND,
            source_system=GCS_SYSID,
            source_component=GCS_COMPID,
        )
        vehicle = Vehicle(
            conn,
            role=Role.GROUND,
            clock=clock,
            on_event=vehicle_events(actions.on_event, now=clock.monotonic),
            heartbeat_allowed=HeartbeatGate(liveness, actions.on_event),
        )
        return GroundMav(
            vehicle,
            clock,
            vehicle_id=cfg.default_vehicle_id,
            latches=latches,
            on_report=actions.on_report,
            on_event=actions.on_event,
            system_action=actions.system,
        )

    mav_thread = MavThread(make_core)
    mav_thread.start()
    await asyncio.to_thread(mav_thread.ready.wait, 10)
    if mav_thread.mav is None:
        error = "timeout" if mav_thread.error is None else type(mav_thread.error).__name__
        log.error("mav.start_failed", error=error)
        raise SystemExit("MAVLink-tråden kunde inte starta. Kontrollera [mavlink] listen.")
    actions.mav = mav_thread.mav
    g.vehicle_link = mav_thread.mav
    g.emergency = actions
    g.vehicle_actions = actions
    return mav_thread, actions


async def start_mission_flow(
    g: Ground, repo: FlowRepo, actions: VehicleActions, mav_thread: MavThread
) -> tuple[GroundLink, MissionFlow]:
    """SD-link to the onboard computer and the mission flow. The link only listens on
    loopback; the emergency path sends its notices through it from now on."""
    cfg = g.config
    link = GroundLink.from_file(
        g.loaded.path(cfg.sdlink.link_tokens_file),
        clock=g.clock,
        latches=actions.latches,
        vehicle_ids=[cfg.default_vehicle_id],
        bind=cfg.sdlink.bind,
        commander_state=lambda: g.brain_state,
        on_event=lambda t, sev, vid, p: actions.on_event(t, sev, {"vehicle_id": vid, **p}),
    )
    flow = MissionFlow(
        g,
        repo,
        caps=limit_caps(g.loaded),
        link=link,
        mav=mav_thread.mav,
        actions=actions,
    )
    link.handler = flow
    actions.notices = link
    g.sdlink = link
    g.flow = flow
    await flow.start()
    await link.start()
    return link, flow


def serve(loaded: Loaded) -> int:
    return asyncio.run(_serve(loaded))


def openapi_documents() -> dict[str, dict[str, Any]]:
    """The OpenAPI documents, built without config files, database or network."""
    from sd_ground.config import GroundConfig
    from sd_obs.clock import MonotonicClock

    loaded = Loaded(config=GroundConfig(), root=Path("/nonexistent"), env={})
    g = Ground(loaded=loaded, clock=MonotonicClock(), brain_token=BrainToken(None), version="0.0.0")
    return {
        "panel": create_panel_app(g).openapi(),
        "commander": create_commander_app(g).openapi(),
        "search": create_search_doc(g).openapi(),
    }
