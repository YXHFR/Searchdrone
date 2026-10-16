"""The injectable clock: all time in logic comes from here, so tests can control it.

``MonotonicClock`` is the real clock. ``FakeClock`` only moves when a test calls
``advance()``; sleepers wake in time order. Safety decisions use ``monotonic()`` only;
``utc_now()`` is for timestamps that people read.
"""

from __future__ import annotations

import asyncio
import datetime as dt
import heapq
import itertools
import time
from typing import Protocol

__all__ = ["Clock", "FakeClock", "MonotonicClock"]


class Clock(Protocol):
    def monotonic(self) -> float: ...

    def monotonic_ms(self) -> int: ...

    def utc_now(self) -> dt.datetime: ...

    async def sleep(self, seconds: float) -> None: ...


class MonotonicClock:
    """Real time."""

    def monotonic(self) -> float:
        return time.monotonic()

    def monotonic_ms(self) -> int:
        return time.monotonic_ns() // 1_000_000

    def utc_now(self) -> dt.datetime:
        return dt.datetime.now(dt.UTC)

    async def sleep(self, seconds: float) -> None:
        await asyncio.sleep(max(0.0, seconds))


class FakeClock:
    """Simulated time. ``advance(s)`` moves both clocks and wakes due sleepers."""

    def __init__(
        self,
        start_mono: float = 1000.0,
        start_utc: dt.datetime = dt.datetime(2026, 10, 15, 12, 0, tzinfo=dt.UTC),
    ) -> None:
        self._mono = start_mono
        self._utc0 = start_utc
        self._mono0 = start_mono
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
        while self._sleepers and self._sleepers[0][0] <= target:
            due, _, fut = heapq.heappop(self._sleepers)
            self._mono = max(self._mono, due)
            if not fut.done():
                fut.set_result(None)
        self._mono = target

    async def run_for(self, seconds: float, step: float = 0.05) -> None:
        """Advance in small steps and let woken tasks run between them."""
        steps = max(1, round(seconds / step))
        for _ in range(steps):
            self.advance(seconds / steps)
            for _ in range(5):
                await asyncio.sleep(0)
