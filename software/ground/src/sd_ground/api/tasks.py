"""Run a WebSocket's workers until the first one ends (the peer left or a timeout hit)."""

from __future__ import annotations

import contextlib
from collections.abc import Awaitable, Callable

import anyio
from starlette.websockets import WebSocketDisconnect

__all__ = ["until_first_done"]


async def until_first_done(*workers: Callable[[], Awaitable[None]]) -> None:
    """Cancellation-safe under the server's own cancellation (anyio task group)."""
    async with anyio.create_task_group() as group:

        async def run(worker: Callable[[], Awaitable[None]]) -> None:
            with contextlib.suppress(WebSocketDisconnect, RuntimeError, OSError):  # peer gone
                await worker()
            group.cancel_scope.cancel()

        for worker in workers:
            group.start_soon(run, worker)
