"""In-memory transports: a connected pair, and a lossy wrapper with delay."""

from __future__ import annotations

import asyncio
import random
from dataclasses import dataclass, field

from sd_link.clock import Clock
from sd_link.transport import ABORT_CODE, TransportClosedError

__all__ = ["LossyTransport", "MemoryTransport", "transport_pair"]

_CLOSE = object()


@dataclass(eq=False)
class MemoryTransport:
    """One end of an in-memory pipe. ``sent`` keeps every frame sent, for asserts."""

    _inbox: asyncio.Queue[object] = field(default_factory=asyncio.Queue[object])
    peer: MemoryTransport | None = None
    sent: list[str] = field(default_factory=list[str])
    _close_code: int | None = None
    _close_reason: str = ""
    block_close: bool = False  # close() never returns (a dead peer)
    aborted: bool = False

    async def send(self, text: str) -> None:
        if self._close_code is not None or self.peer is None:
            raise TransportClosedError(self._close_code, self._close_reason)
        self.sent.append(text)
        self.peer.deliver(text)

    def deliver(self, text: str) -> None:
        if self._close_code is None:
            self._inbox.put_nowait(text)

    async def recv(self) -> str | bytes:
        if self._close_code is not None and self._inbox.empty():
            raise TransportClosedError(self._close_code, self._close_reason)
        item = await self._inbox.get()
        if item is _CLOSE:
            raise TransportClosedError(self._close_code, self._close_reason)
        assert isinstance(item, str)  # noqa: S101
        return item

    async def close(self, code: int, reason: str = "") -> None:
        if self.block_close:
            await asyncio.Event().wait()  # a peer that never answers the close frame
        if self._close_code is not None:
            return
        self._mark_closed(code, reason)
        if self.peer is not None:
            self.peer._mark_closed(code, reason)

    def abort(self) -> None:
        self.aborted = True
        if self._close_code is None:
            self._mark_closed(ABORT_CODE, "")
        if self.peer is not None:
            self.peer._mark_closed(ABORT_CODE, "")

    def _mark_closed(self, code: int, reason: str) -> None:
        if self._close_code is None:
            self._close_code, self._close_reason = code, reason
            self._inbox.put_nowait(_CLOSE)

    @property
    def close_code(self) -> int | None:
        return self._close_code


def transport_pair() -> tuple[MemoryTransport, MemoryTransport]:
    a, b = MemoryTransport(), MemoryTransport()
    a.peer, b.peer = b, a
    return a, b


class LossyTransport:
    """Wraps a transport: drops outgoing frames with ``loss`` probability and delays
    the rest by ``delay_s`` on ``clock``, keeping their order. ``broken`` drops all."""

    def __init__(
        self,
        inner: MemoryTransport,
        clock: Clock,
        *,
        loss: float = 0.0,
        delay_s: float = 0.0,
        seed: int = 0,
    ) -> None:
        self.inner = inner
        self.loss = loss
        self.delay_s = delay_s
        self.broken = False
        self.dropped = 0
        self._clock = clock
        self._rng = random.Random(seed)  # noqa: S311  # test noise, not crypto
        self._queue: asyncio.Queue[tuple[float, str]] = asyncio.Queue()
        self._pump = asyncio.ensure_future(self._run())

    async def send(self, text: str) -> None:
        if self.inner.close_code is not None:
            raise TransportClosedError(self.inner.close_code)
        if self.broken or self._rng.random() < self.loss:
            self.dropped += 1
            return
        self._queue.put_nowait((self._clock.now() + self.delay_s, text))

    async def _run(self) -> None:
        while True:
            due, text = await self._queue.get()
            wait = due - self._clock.now()
            if wait > 0:
                await self._clock.sleep(wait)
            try:
                await self.inner.send(text)
            except TransportClosedError:
                return

    async def recv(self) -> str | bytes:
        return await self.inner.recv()

    async def close(self, code: int, reason: str = "") -> None:
        self._pump.cancel()
        await self.inner.close(code, reason)

    def abort(self) -> None:
        self._pump.cancel()  # frames still in flight are lost with the connection
        self.inner.abort()

    @property
    def close_code(self) -> int | None:
        return self.inner.close_code
