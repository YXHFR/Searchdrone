"""The injectable clock: FakeClock moves only when told, and wakes sleepers in order."""

from __future__ import annotations

import asyncio
import datetime as dt

from sd_obs.clock import FakeClock, MonotonicClock


def test_fake_clock_moves_both_clocks() -> None:
    clock = FakeClock(start_mono=10.0, start_utc=dt.datetime(2026, 1, 1, tzinfo=dt.UTC))
    clock.advance(1.5)
    assert clock.monotonic() == 11.5
    assert clock.monotonic_ms() == 11500
    assert clock.utc_now() == dt.datetime(2026, 1, 1, 0, 0, 1, 500000, tzinfo=dt.UTC)


async def test_sleepers_wake_in_order() -> None:
    clock = FakeClock()
    woke: list[str] = []

    async def sleeper(name: str, seconds: float) -> None:
        await clock.sleep(seconds)
        woke.append(name)

    tasks = [asyncio.create_task(sleeper("b", 2.0)), asyncio.create_task(sleeper("a", 1.0))]
    await asyncio.sleep(0)
    clock.advance(0.5)
    await asyncio.sleep(0)
    assert woke == []
    await clock.run_for(2.0)
    await asyncio.gather(*tasks)
    assert woke == ["a", "b"]


async def test_monotonic_clock_is_real_time() -> None:
    clock = MonotonicClock()
    before = clock.monotonic()
    await clock.sleep(0.01)
    assert clock.monotonic() > before
    assert clock.utc_now().tzinfo is not None
