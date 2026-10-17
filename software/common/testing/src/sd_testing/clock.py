"""``FakeClock``: simulated time for :class:`sd_obs.clock.Clock`.

Time moves only when a test calls :meth:`FakeClock.advance` (no yielding),
:meth:`FakeClock.run_for` (small steps, letting woken tasks run) or
:meth:`FakeClock.run_until`. Sleepers wake in time order.
"""

from __future__ import annotations

import asyncio
import datetime as dt
import heapq
import itertools
from typing import Any

__all__ = ["FakeClock"]

_START_UTC = dt.datetime(2026, 10, 15, 12, 0, tzinfo=dt.UTC)
_YIELDS = 20  # event-loop turns after each step, so woken tasks and their follow-ups run


class FakeClock:
    """Simulated monotonic and wall-clock time, both moved together."""

    def __init__(self, start_mono: float = 1000.0, start_utc: dt.datetime = _START_UTC) -> None:
        self._mono = start_mono
        self._mono0 = start_mono
        self._utc0 = start_utc
        self._sleepers: list[tuple[float, int, asyncio.Future[None]]] = []
        self._order = itertools.count()

    def monotonic(self) -> float:
        return self._mono

    def monotonic_ms(self) -> int:
        return round(self._mono * 1000)

    def utc_now(self) -> dt.datetime:
        return self._utc0 + dt.timedelta(seconds=self._mono - self._mono0)

    async def sleep(self, seconds: float) -> None:
        if seconds <= 0:
            await asyncio.sleep(0)
            return
        fut: asyncio.Future[None] = asyncio.get_running_loop().create_future()
        heapq.heappush(self._sleepers, (self._mono + seconds, next(self._order), fut))
        await fut

    def advance(self, seconds: float) -> None:
        """Move time forward and wake every sleeper whose time has come (in order)."""
        target = self._mono + seconds
        while self._sleepers and self._sleepers[0][0] <= target + 1e-9:
            due, _, fut = heapq.heappop(self._sleepers)
            self._mono = max(self._mono, due)
            if not fut.done():
                fut.set_result(None)
        self._mono = target

    async def run_for(self, seconds: float, step: float = 0.01) -> None:
        """Advance in steps of ``step`` and let woken tasks run after each step."""
        end = self._mono + seconds
        while self._mono < end - 1e-9:
            self.advance(min(step, end - self._mono))
            await self._yield()
        await self._yield()

    async def run_until(
        self, done: asyncio.Future[Any] | asyncio.Event, limit_s: float, step: float = 0.01
    ) -> None:
        """Advance until ``done`` is set; ``TimeoutError`` after ``limit_s`` simulated s."""
        end = self._mono + limit_s
        while self._mono < end and not _is_done(done):
            await self.run_for(step, step)
        if not _is_done(done):
            raise TimeoutError(f"not done after {limit_s} simulated seconds")

    @staticmethod
    async def _yield() -> None:
        for _ in range(_YIELDS):
            await asyncio.sleep(0)


def _is_done(done: asyncio.Future[Any] | asyncio.Event) -> bool:
    return done.is_set() if isinstance(done, asyncio.Event) else done.done()
