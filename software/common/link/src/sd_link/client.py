"""The onboard side: ``LinkClient`` dials the ground and keeps reconnecting.

Per connection: send ``Hello`` (built fresh by ``make_hello``), wait for ``HelloAck``,
then hand the running session to ``on_session``. Reconnect after 0.5 s, doubling to
at most 5 s, +-20 % jitter. After 4401/4403/4400 wait 30 s and log critical.
"""

from __future__ import annotations

import asyncio
import contextlib
import logging
import random
from collections.abc import Awaitable, Callable
from dataclasses import dataclass

from websockets.asyncio.client import connect
from websockets.exceptions import InvalidHandshake, InvalidURI

from sd_link.clock import Clock
from sd_link.close_codes import FATAL_CLOSE_CODES, CloseCode
from sd_link.constants import (
    AUTH_RETRY_S,
    HELLO_ACK_TIMEOUT_S,
    LINK_LOST_S,
    MAX_FRAME_BYTES,
    RECONNECT_INITIAL_S,
    RECONNECT_JITTER,
    RECONNECT_MAX_S,
)
from sd_link.envelope import GROUND_SENDER as _GROUND
from sd_link.envelope import DecodeError, build_envelope, decode, encode, vehicle_sender
from sd_link.session import LinkSession
from sd_link.transport import Transport, TransportClosedError, WebSocketTransport, first_or_timeout
from sd_schemas.link import SCHEMA_MAJOR, Hello, HelloAck

__all__ = ["Backoff", "ClientSessionHandler", "LinkClient", "handshake"]

log = logging.getLogger("sd_link")

ClientSessionHandler = Callable[[LinkSession, HelloAck], Awaitable[None]]


@dataclass(slots=True)
class Backoff:
    """Reconnect delays. ``reset`` after a successful handshake."""

    rng: random.Random
    attempt: int = 0

    def next_delay(self, close_code: int | None) -> float:
        if close_code in FATAL_CLOSE_CODES:
            self.attempt = 0
            return AUTH_RETRY_S
        base = min(RECONNECT_INITIAL_S * (2**self.attempt), RECONNECT_MAX_S)
        self.attempt += 1
        return base * (1 + self.rng.uniform(-RECONNECT_JITTER, RECONNECT_JITTER))

    def reset(self) -> None:
        self.attempt = 0


async def handshake(
    transport: Transport,
    hello: Hello,
    *,
    vehicle_id: str,
    clock: Clock,
    timeout_s: float = HELLO_ACK_TIMEOUT_S,
) -> HelloAck | None:
    """Send Hello and wait for a valid HelloAck. None (and the connection closed) if
    none comes in time or the answer is not a HelloAck with our schema_major."""
    await transport.send(
        encode(
            build_envelope(
                hello,
                sender=vehicle_sender(vehicle_id),
                vehicle_id=vehicle_id,
                seq=0,
                sent_utc=clock.utc_now(),
            )
        )
    )
    done, frame = await first_or_timeout(clock, transport.recv, timeout_s)
    if not done or frame is None:
        log.warning("sdlink no HelloAck within %.1fs", timeout_s)
        await transport.close(CloseCode.GOING_AWAY, "no hello ack")
        return None
    try:
        envelope, body = decode(frame)
    except DecodeError as exc:
        log.warning("sdlink bad HelloAck: %s", exc)
        await transport.close(CloseCode.GOING_AWAY, "bad hello ack")
        return None
    if (
        not isinstance(body, HelloAck)
        or body.schema_major != SCHEMA_MAJOR
        or envelope.sender != _GROUND
        or envelope.vehicle_id != vehicle_id
    ):
        log.warning("sdlink unexpected answer to Hello: %s", envelope.kind)
        await transport.close(CloseCode.GOING_AWAY, "bad hello ack")
        return None
    return body


class LinkClient:
    def __init__(
        self,
        *,
        url: str,
        token: str,
        vehicle_id: str,
        make_hello: Callable[[], Hello],
        on_session: ClientSessionHandler,
        clock: Clock,
        rng: random.Random | None = None,
        lost_after_s: float = LINK_LOST_S,
    ) -> None:
        self._url = url
        self._token = token
        self.vehicle_id = vehicle_id
        self._make_hello = make_hello
        self._on_session = on_session
        self._clock = clock
        self._backoff = Backoff(rng or random.Random())  # noqa: S311  # jitter, not crypto
        self._lost_after = lost_after_s
        self.last_close_code: int | None = None

    def __repr__(self) -> str:  # never show the token
        return f"LinkClient(url={self._url!r}, vehicle_id={self.vehicle_id!r})"

    async def run_forever(self, stop: asyncio.Event) -> None:
        while not stop.is_set():
            code = await self.connect_once()
            self.last_close_code = code
            if stop.is_set():
                return
            delay = self._backoff.next_delay(code)
            if code in FATAL_CLOSE_CODES:
                log.critical("sdlink refused by ground (%s); retry in %.0fs", code, delay)
            await self._sleep_unless(stop, delay)

    async def connect_once(self) -> int | None:
        """One connection from dial to close. Returns the close code (None if the
        connection could not be opened)."""
        headers = {"Authorization": f"Bearer {self._token}"}
        try:
            async with connect(
                self._url,
                additional_headers=headers,
                max_size=MAX_FRAME_BYTES,
                compression=None,
                user_agent_header=None,
                open_timeout=HELLO_ACK_TIMEOUT_S,
            ) as ws:
                transport = WebSocketTransport(ws)
                with contextlib.suppress(TransportClosedError):
                    await self.serve(transport)
                return transport.close_code
        except (OSError, InvalidHandshake, InvalidURI, TimeoutError) as exc:
            log.warning("sdlink connect failed: %s", type(exc).__name__)
            return None

    async def serve(self, transport: Transport) -> None:
        """Handshake and run a session on an open transport."""
        ack = await handshake(
            transport, self._make_hello(), vehicle_id=self.vehicle_id, clock=self._clock
        )
        if ack is None:
            return
        self._backoff.reset()
        session = LinkSession(
            transport,
            side="vehicle",
            vehicle_id=self.vehicle_id,
            clock=self._clock,
            seq_start=1,
            lost_after_s=self._lost_after,
        )
        app = asyncio.ensure_future(self._on_session(session, ack))
        try:
            await session.run()
        finally:
            await asyncio.gather(app, return_exceptions=True)

    async def _sleep_unless(self, stop: asyncio.Event, seconds: float) -> None:
        await first_or_timeout(self._clock, stop.wait, seconds)
