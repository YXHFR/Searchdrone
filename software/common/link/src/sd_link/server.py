"""The ground side: ``LinkServer`` accepts one connection per vehicle.

Order on a new connection (the WebSocket is always accepted first, then closed with
the right code, so that the client sees the code and not HTTP 403):

1. an ``Origin`` header (only browsers send one) -> 4403;
2. wrong, missing or repeated ``Authorization: Bearer <token>`` -> 4401;
3. the token's vehicle is not known -> 4403;
4. first message within ``HELLO_TIMEOUT_S`` must be ``Hello`` with our
   ``schema_major`` -> otherwise 4400; from another vehicle than the token's -> 4403;
5. duplicates: a newer connection with another ``boot_id``, or one that arrives when
   the old connection has been silent for ``LINK_LOSS_DETECT_S``, replaces the old one
   (old closed with 4409, without waiting for it); otherwise the new one gets 4409;
6. ``on_hello`` builds the ``HelloAck`` (the ground reconciles latches there), it is
   sent, and ``on_session`` gets the running session.

Admission is serialised per vehicle, never across vehicles. ``on_session`` must read
events until ``Closed`` and then return; it is cancelled if it has not returned
``APP_DRAIN_S`` after the connection ended. If it fails, the session is closed.
"""

from __future__ import annotations

import asyncio
import contextlib
import inspect
import logging
from collections.abc import Awaitable, Callable, Mapping
from dataclasses import dataclass
from typing import Final, Self

from websockets.asyncio.server import Server, ServerConnection, serve
from websockets.datastructures import Headers
from websockets.http11 import Request, Response

from sd_link.clock import Clock, first_or_timeout
from sd_link.close_codes import CloseCode
from sd_link.constants import (
    CLOSE_TIMEOUT_S,
    HELLO_TIMEOUT_S,
    LINK_LOSS_DETECT_S,
    LINK_LOST_S,
    MAX_FRAME_BYTES,
    OPEN_TIMEOUT_S,
    SDLINK_PATH,
)
from sd_link.envelope import GROUND_SENDER, DecodeError, build_envelope, decode, encode
from sd_link.envelope import vehicle_sender as _vehicle_sender
from sd_link.reliability import DedupCache
from sd_link.session import LinkSession
from sd_link.tokens import TokenStore
from sd_link.transport import Transport, TransportClosedError, WebSocketTransport
from sd_link.wslog import WS_LOGGER
from sd_schemas.link import SCHEMA_MAJOR, Hello, HelloAck

__all__ = [
    "APP_DRAIN_S",
    "HelloHandler",
    "LinkServer",
    "SessionHandler",
    "VehicleCheck",
    "bearer_token",
]

log = logging.getLogger("sd_link")

APP_DRAIN_S: Final = 2.0

HelloHandler = Callable[[str, Hello], Awaitable[HelloAck]]
SessionHandler = Callable[[LinkSession, Hello], Awaitable[None]]
VehicleCheck = Callable[[str], bool | Awaitable[bool]]
LinkEventHook = Callable[[str, str], None]  # (event name, vehicle_id)
HeaderSource = Mapping[str, str] | Headers


def _header(headers: HeaderSource, name: str) -> tuple[str | None, bool]:
    """(value, repeated). A repeated header is never trusted."""
    if isinstance(headers, Headers):
        values = headers.get_all(name)
        return (values[0] if values else None), len(values) > 1
    return headers.get(name), False


def bearer_token(headers: HeaderSource) -> str | None:
    value, repeated = _header(headers, "Authorization")
    if not value or repeated:
        return None
    scheme, _, token = value.partition(" ")
    if scheme.lower() != "bearer" or not token.strip():
        return None
    return token.strip()


@dataclass(slots=True)
class _Active:
    session: LinkSession
    boot_id: int


class LinkServer:
    def __init__(
        self,
        *,
        tokens: TokenStore,
        is_known_vehicle: VehicleCheck,
        on_hello: HelloHandler,
        on_session: SessionHandler,
        clock: Clock,
        host: str = "127.0.0.1",
        port: int = 7600,
        on_event: LinkEventHook | None = None,
        hello_timeout_s: float = HELLO_TIMEOUT_S,
        lost_after_s: float = LINK_LOST_S,
    ) -> None:
        self._tokens = tokens
        self._is_known = is_known_vehicle
        self._on_hello = on_hello
        self._on_session = on_session
        self._clock = clock
        self.host = host
        self.port = port
        self._on_event = on_event
        self._hello_timeout = hello_timeout_s
        self._lost_after = lost_after_s
        self._active: dict[str, _Active] = {}
        self._locks: dict[str, asyncio.Lock] = {}
        self._dedup: dict[str, DedupCache] = {}  # per vehicle, kept across reconnects
        self._background: set[asyncio.Future[None]] = set()
        self._server: Server | None = None

    # --- lifecycle ---------------------------------------------------------------

    async def start(self) -> None:
        self._server = await serve(
            self._handle_ws,
            self.host,
            self.port,
            process_request=_only_sdlink_path,
            max_size=MAX_FRAME_BYTES,
            compression=None,
            server_header=None,
            ping_interval=None,  # SD-link has its own heartbeat and liveness
            close_timeout=CLOSE_TIMEOUT_S,
            open_timeout=OPEN_TIMEOUT_S,
            logger=WS_LOGGER,
        )
        if self.port == 0:
            self.port = self._server.sockets[0].getsockname()[1]

    async def stop(self) -> None:
        await asyncio.gather(
            *(
                a.session.close(CloseCode.GOING_AWAY, "server stopping")
                for a in self._active.values()
            )
        )
        if self._server is not None:
            self._server.close()
            await self._server.wait_closed()
            self._server = None

    async def __aenter__(self) -> Self:
        await self.start()
        return self

    async def __aexit__(self, *exc: object) -> None:
        await self.stop()

    def session_for(self, vehicle_id: str) -> LinkSession | None:
        active = self._active.get(vehicle_id)
        return None if active is None else active.session

    # --- connection --------------------------------------------------------------

    async def _handle_ws(self, ws: ServerConnection) -> None:
        await self.handle(WebSocketTransport(ws), ws.request.headers if ws.request else Headers())

    async def handle(self, transport: Transport, headers: HeaderSource) -> None:
        """Run one connection to its end. Public so that tests can feed any transport."""
        origin, origin_repeated = _header(headers, "Origin")
        if origin is not None or origin_repeated:
            await self._refuse(transport, CloseCode.FORBIDDEN, "origin not allowed", None)
            return
        vehicle_id = self._tokens.match(bearer_token(headers))
        if vehicle_id is None:
            await self._refuse(transport, CloseCode.AUTH_FAILED, "auth failed", None)
            return
        if not await self._known(vehicle_id):
            await self._refuse(transport, CloseCode.FORBIDDEN, "unknown vehicle", vehicle_id)
            return
        hello = await self._read_hello(transport, vehicle_id)
        if hello is None:
            return
        session = await self._admit(transport, vehicle_id, hello)
        if session is None:
            return
        app = asyncio.ensure_future(self._on_session(session, hello))
        app.add_done_callback(lambda task: self._app_done(task, session))
        try:
            await session.run()
        finally:
            active = self._active.get(vehicle_id)
            if active is not None and active.session is session:
                del self._active[vehicle_id]
            await first_or_timeout(self._clock, lambda: _settle(app), APP_DRAIN_S)
            app.cancel()

    def _app_done(self, task: asyncio.Task[None], session: LinkSession) -> None:
        if task.cancelled() or task.exception() is None:
            return
        log.error("sdlink session handler failed: %r", task.exception())
        self._spawn(session.close(CloseCode.GOING_AWAY, "handler failed"))

    async def _known(self, vehicle_id: str) -> bool:
        result = self._is_known(vehicle_id)
        return await result if inspect.isawaitable(result) else bool(result)

    async def _read_hello(self, transport: Transport, vehicle_id: str) -> Hello | None:
        try:
            done, frame = await first_or_timeout(self._clock, transport.recv, self._hello_timeout)
        except TransportClosedError:
            return None
        if not done or frame is None:
            await self._refuse(transport, CloseCode.BAD_HELLO, "no hello", vehicle_id)
            return None
        try:
            envelope, body = decode(frame)
        except DecodeError:
            await self._refuse(transport, CloseCode.BAD_HELLO, "bad hello", vehicle_id)
            return None
        if not isinstance(body, Hello) or body.schema_major != SCHEMA_MAJOR:
            await self._refuse(transport, CloseCode.BAD_HELLO, "bad hello", vehicle_id)
            return None
        if envelope.vehicle_id != vehicle_id or envelope.sender != _vehicle_sender(vehicle_id):
            await self._refuse(transport, CloseCode.FORBIDDEN, "vehicle mismatch", vehicle_id)
            return None
        return body

    async def _admit(
        self, transport: Transport, vehicle_id: str, hello: Hello
    ) -> LinkSession | None:
        async with self._locks.setdefault(vehicle_id, asyncio.Lock()):
            old = self._active.get(vehicle_id)
            if old is not None:
                silent = self._clock.now() - old.session.last_inbound
                if hello.boot_id == old.boot_id and silent < LINK_LOSS_DETECT_S:
                    await self._refuse(transport, CloseCode.REPLACED, "duplicate", vehicle_id)
                    return None
                log.warning("sdlink replaced: vehicle=%s", vehicle_id)
                self._emit("link.sdlink.replaced", vehicle_id)
                del self._active[vehicle_id]
                # The old peer may be gone: never wait for its close handshake here.
                self._spawn(old.session.close(CloseCode.REPLACED, "replaced"))
            ack = await self._on_hello(vehicle_id, hello)
            try:
                await transport.send(
                    encode(
                        build_envelope(
                            ack,
                            sender=GROUND_SENDER,
                            vehicle_id=vehicle_id,
                            seq=0,
                            sent_utc=self._clock.utc_now(),
                        )
                    )
                )
            except TransportClosedError:
                return None
            session = LinkSession(
                transport,
                side="ground",
                vehicle_id=vehicle_id,
                clock=self._clock,
                seq_start=1,
                dedup=self._dedup.setdefault(vehicle_id, DedupCache()),
                lost_after_s=self._lost_after,
            )
            self._active[vehicle_id] = _Active(session, hello.boot_id)
        self._emit("link.sdlink.up", vehicle_id)
        return session

    async def _refuse(
        self, transport: Transport, code: CloseCode, reason: str, vehicle_id: str | None
    ) -> None:
        level = logging.WARNING if code is CloseCode.REPLACED else logging.CRITICAL
        log.log(level, "sdlink refused (%d %s): vehicle=%s", code, reason, vehicle_id or "-")
        self._emit(f"link.sdlink.refused_{code.value}", vehicle_id or "-")
        done, _ = await first_or_timeout(
            self._clock, lambda: _close_quietly(transport, code, reason), CLOSE_TIMEOUT_S
        )
        if not done:
            transport.abort()

    def _spawn(self, work: Awaitable[None]) -> None:
        task = asyncio.ensure_future(work)
        self._background.add(task)
        task.add_done_callback(self._background.discard)

    def _emit(self, name: str, vehicle_id: str) -> None:
        if self._on_event is not None:
            self._on_event(name, vehicle_id)


async def _settle(task: asyncio.Future[None]) -> None:
    await asyncio.gather(task, return_exceptions=True)


async def _close_quietly(transport: Transport, code: int, reason: str) -> None:
    with contextlib.suppress(TransportClosedError):
        await transport.close(code, reason)


def _only_sdlink_path(connection: ServerConnection, request: Request) -> Response | None:
    if request.path != SDLINK_PATH:
        return connection.respond(404, "Not found\n")
    return None
