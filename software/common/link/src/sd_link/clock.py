"""The clock SD-link reads. All link timing goes through an injected ``Clock``.

Tests use ``sd_link_testing.FakeClock``. This is the only module in ``sd_link`` that
reads the system clock.
"""

from __future__ import annotations

import asyncio
import time
from datetime import UTC, datetime
from typing import Protocol, runtime_checkable

__all__ = ["Clock", "MonotonicClock"]


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
