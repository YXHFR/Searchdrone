"""The injectable clock: all time in logic comes from here, so tests can control it.

This is the one clock abstraction of the project; every package that needs time takes
a ``Clock``. ``MonotonicClock`` is the real clock. Tests use ``sd_testing.FakeClock``,
which lives in a test-only package. Safety decisions use ``monotonic()`` only;
``utc_now()`` is for timestamps that people read.
"""

from __future__ import annotations

import asyncio
import datetime as dt
import time
from typing import Protocol

__all__ = ["Clock", "MonotonicClock"]


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
