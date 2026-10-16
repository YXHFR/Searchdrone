"""The byte pipe under a session: one text frame per message.

``WebSocketTransport`` wraps a ``websockets`` connection; tests use the in-memory
transports in ``sd_link_testing``.
"""

from __future__ import annotations

import asyncio
from collections.abc import Awaitable, Callable
from typing import Protocol

from websockets.asyncio.connection import Connection
from websockets.exceptions import ConnectionClosed

from sd_link.clock import Clock

__all__ = ["Transport", "TransportClosedError", "WebSocketTransport", "first_or_timeout"]


class TransportClosedError(Exception):
    def __init__(self, code: int | None, reason: str = "") -> None:
        super().__init__(f"closed ({code}) {reason}".strip())
        self.code = code
        self.reason = reason


class Transport(Protocol):
    async def send(self, text: str) -> None:
        """Send one frame. Raises TransportClosedError when closed."""
        ...

    async def recv(self) -> str | bytes:
        """The next frame. Raises TransportClosedError when closed."""
        ...

    async def close(self, code: int, reason: str = "") -> None: ...

    @property
    def close_code(self) -> int | None: ...


class WebSocketTransport:
    def __init__(self, connection: Connection) -> None:
        self._ws = connection

    async def send(self, text: str) -> None:
        try:
            await self._ws.send(text)
        except ConnectionClosed as exc:
            raise TransportClosedError(_code(exc), _reason(exc)) from None

    async def recv(self) -> str | bytes:
        try:
            return await self._ws.recv()
        except ConnectionClosed as exc:
            raise TransportClosedError(_code(exc), _reason(exc)) from None

    async def close(self, code: int, reason: str = "") -> None:
        await self._ws.close(code, reason)

    @property
    def close_code(self) -> int | None:
        return self._ws.close_code


def _code(exc: ConnectionClosed) -> int | None:
    frame = exc.rcvd or exc.sent
    return None if frame is None else frame.code


def _reason(exc: ConnectionClosed) -> str:
    frame = exc.rcvd or exc.sent
    return "" if frame is None else frame.reason


async def first_or_timeout[T](
    clock: Clock, work: Callable[[], Awaitable[T]], timeout_s: float
) -> tuple[bool, T | None]:
    """Run ``work`` with a deadline measured on ``clock``. Returns (done, result);
    on timeout the work is cancelled and (False, None) is returned."""

    async def runner() -> T:
        return await work()

    task = asyncio.ensure_future(runner())
    timer = asyncio.ensure_future(clock.sleep(timeout_s))
    try:
        await asyncio.wait({task, timer}, return_when=asyncio.FIRST_COMPLETED)
    except BaseException:
        task.cancel()
        raise
    finally:
        timer.cancel()
    if task.done():
        return True, task.result()
    task.cancel()
    await asyncio.gather(task, return_exceptions=True)
    return False, None
