"""Fake peers for tests.

* ``FakeLink``: a ground session and a vehicle session joined in memory, already past
  the handshake. For ground tests that need an onboard side and vice versa.
* ``FakeGround``: a real ``LinkServer`` on 127.0.0.1 that answers ``Hello`` and lets a
  test send anything to the onboard client, **without the ground gate**.
"""

from __future__ import annotations

import asyncio
import contextlib
from collections.abc import AsyncIterator
from dataclasses import dataclass, field
from typing import Self

from sd_link.clock import Clock
from sd_link.server import LinkServer
from sd_link.session import Closed, Inbound, LinkClosedError, LinkEvent, LinkSession
from sd_link.tokens import TokenStore, generate_token, hash_token
from sd_link_testing.memory import MemoryTransport, transport_pair
from sd_obs.clock import MonotonicClock
from sd_schemas.base import SdModel, new_id
from sd_schemas.link import SCHEMA_MAJOR, Hello, HelloAck, LatchRef

__all__ = ["FakeGround", "FakeLink", "next_inbound"]


@dataclass
class FakeLink:
    """``ground`` and ``vehicle`` sessions over an in-memory pipe, both running."""

    clock: Clock
    vehicle_id: str = "sim-01"
    ground: LinkSession = field(init=False)
    vehicle: LinkSession = field(init=False)
    ground_transport: MemoryTransport = field(init=False)
    vehicle_transport: MemoryTransport = field(init=False)
    _tasks: list[asyncio.Task[None]] = field(default_factory=list[asyncio.Task[None]])

    def __post_init__(self) -> None:
        g, v = transport_pair()
        self.ground_transport, self.vehicle_transport = g, v
        self.ground = LinkSession(g, side="ground", vehicle_id=self.vehicle_id, clock=self.clock)
        self.vehicle = LinkSession(v, side="vehicle", vehicle_id=self.vehicle_id, clock=self.clock)

    async def __aenter__(self) -> Self:
        self._tasks = [
            asyncio.ensure_future(self.ground.run()),
            asyncio.ensure_future(self.vehicle.run()),
        ]
        return self

    async def __aexit__(self, *exc: object) -> None:
        await self.ground.close()
        await asyncio.gather(*self._tasks, return_exceptions=True)


async def next_inbound(session: LinkSession, kind: type[SdModel] | None = None) -> Inbound:
    """The next delivered message (optionally of one kind); other events are skipped.
    Raises LinkClosedError when the session ends first."""
    while True:
        event: LinkEvent = await session.next_event()
        if isinstance(event, Closed):
            raise LinkClosedError(f"closed ({event.code})")
        if isinstance(event, Inbound) and (kind is None or isinstance(event.body, kind)):
            return event


class FakeGround:
    """A ground station for onboard tests. ``token`` is what the client must send."""

    def __init__(
        self,
        vehicle_id: str = "sim-01",
        *,
        clock: Clock | None = None,
        operator_latch: LatchRef | None = None,
    ) -> None:
        self.vehicle_id = vehicle_id
        self.token = generate_token()
        self.operator_latch = operator_latch
        self.hellos: list[Hello] = []
        self.sessions: asyncio.Queue[LinkSession] = asyncio.Queue()
        self.server_epoch = new_id()
        self.server = LinkServer(
            tokens=TokenStore({vehicle_id: hash_token(self.token)}),
            is_known_vehicle=lambda v: v == vehicle_id,
            on_hello=self._on_hello,
            on_session=self._on_session,
            clock=clock or MonotonicClock(),
            port=0,
        )

    @property
    def url(self) -> str:
        return f"ws://127.0.0.1:{self.server.port}/sdlink/v1/control"

    async def __aenter__(self) -> Self:
        await self.server.start()
        return self

    async def __aexit__(self, *exc: object) -> None:
        await self.server.stop()

    async def _on_hello(self, vehicle_id: str, hello: Hello) -> HelloAck:
        self.hellos.append(hello)
        return HelloAck(
            schema_major=SCHEMA_MAJOR,
            server_epoch=self.server_epoch,
            operator_latch=self.operator_latch,
        )

    async def _on_session(self, session: LinkSession, hello: Hello) -> None:
        self.sessions.put_nowait(session)

    @contextlib.asynccontextmanager
    async def connected(self) -> AsyncIterator[LinkSession]:
        """Wait for the onboard client and yield its session."""
        yield await self.sessions.get()
