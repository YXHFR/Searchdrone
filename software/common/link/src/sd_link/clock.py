"""Deadlines on the injected clock. SD-link takes its time from ``sd_obs.clock.Clock``,
the project's one clock abstraction; tests use ``sd_testing.FakeClock``. No module in
``sd_link`` reads the system clock.
"""

from __future__ import annotations

import asyncio
from collections.abc import Awaitable, Callable

from sd_obs.clock import Clock

__all__ = ["Clock", "first_or_timeout"]


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
