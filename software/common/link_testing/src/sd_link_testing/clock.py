"""``FakeClock``: time moves only when a test calls ``advance``."""

from __future__ import annotations

import asyncio
import heapq
import itertools
from datetime import UTC, datetime, timedelta
from typing import Any

__all__ = ["FakeClock"]

_EPOCH = datetime(2026, 10, 15, 12, 0, tzinfo=UTC)


class FakeClock:
    def __init__(self, start: float = 1000.0) -> None:
        self._now = start
        self._start = start
        self._sleepers: list[tuple[float, int, asyncio.Future[None]]] = []
        self._counter = itertools.count()

    def now(self) -> float:
        return self._now

    def utc_now(self) -> datetime:
        return _EPOCH + timedelta(seconds=self._now - self._start)

    async def sleep(self, seconds: float) -> None:
        if seconds <= 0:
            await asyncio.sleep(0)
            return
        future: asyncio.Future[None] = asyncio.get_running_loop().create_future()
        heapq.heappush(self._sleepers, (self._now + seconds, next(self._counter), future))
        await future

    async def advance(self, seconds: float, step: float = 0.01) -> None:
        """Move time forward in small steps, letting woken tasks run at each step."""
        end = self._now + seconds
        while self._now < end - 1e-9:
            self._now = min(end, self._now + step)
            await self._wake()
        await self._wake()

    async def run_until(self, done: asyncio.Future[Any] | asyncio.Event, limit_s: float) -> None:
        """Advance until ``done`` is set (or ``limit_s`` simulated seconds pass)."""
        end = self._now + limit_s
        while self._now < end and not _is_done(done):
            await self.advance(0.01)
        if not _is_done(done):
            raise TimeoutError(f"not done after {limit_s} simulated seconds")

    async def _wake(self) -> None:
        while self._sleepers and self._sleepers[0][0] <= self._now + 1e-9:
            _, _, future = heapq.heappop(self._sleepers)
            if not future.done():
                future.set_result(None)
        for _ in range(20):  # let woken tasks and their follow-ups run
            await asyncio.sleep(0)


def _is_done(done: asyncio.Future[Any] | asyncio.Event) -> bool:
    return done.is_set() if isinstance(done, asyncio.Event) else done.done()
