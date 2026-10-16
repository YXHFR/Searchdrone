"""The clock SD-link reads. All link timing goes through an injected ``Clock``.

Tests use ``sd_link_testing.FakeClock``. This is the only module in ``sd_link`` that
reads the system clock.
"""

from __future__ import annotations

import asyncio
import time
from collections.abc import Awaitable, Callable
from datetime import UTC, datetime
from typing import Protocol, runtime_checkable

__all__ = ["Clock", "MonotonicClock", "first_or_timeout"]


@runtime_checkable
class Clock(Protocol):
    def now(self) -> float:
        """Monotonic seconds. Only differences mean anything."""
        ...

    async def sleep(self, seconds: float) -> None: ...

    def utc_now(self) -> datetime:
        """Wall-clock time for humans and logs (``sent_utc``); never logic."""
        ...


class MonotonicClock:
    def now(self) -> float:
        return time.monotonic()

    async def sleep(self, seconds: float) -> None:
        await asyncio.sleep(max(0.0, seconds))

    def utc_now(self) -> datetime:
        return datetime.now(UTC)


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
