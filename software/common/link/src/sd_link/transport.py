"""The byte pipe under a session: one text frame per message.

``WebSocketTransport`` wraps a ``websockets`` connection; tests use the in-memory
transports in ``sd_link_testing``.
"""

from __future__ import annotations

from typing import Protocol

from websockets.asyncio.connection import Connection
from websockets.exceptions import ConnectionClosed

__all__ = ["ABORT_CODE", "Transport", "TransportClosedError", "WebSocketTransport"]

ABORT_CODE = 1006  # what a connection reports after abort(): no close frame


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

    async def close(self, code: int, reason: str = "") -> None:
        """Graceful close: frames already queued are still delivered first."""
        ...

    def abort(self) -> None:
        """Drop the connection now, with everything still queued in it."""
        ...

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

    def abort(self) -> None:
        self._ws.transport.abort()

    @property
    def close_code(self) -> int | None:
        return self._ws.close_code


def _code(exc: ConnectionClosed) -> int | None:
    frame = exc.rcvd or exc.sent
    return ABORT_CODE if frame is None else frame.code


def _reason(exc: ConnectionClosed) -> str:
    frame = exc.rcvd or exc.sent
    return "" if frame is None else frame.reason
