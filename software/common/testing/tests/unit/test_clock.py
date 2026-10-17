"""The one clock: FakeClock moves only when told and wakes sleepers in order; the real
clock follows real time."""

from __future__ import annotations

import asyncio
import datetime as dt

import pytest

from sd_obs.clock import Clock, MonotonicClock
from sd_testing import FakeClock


def test_fake_clock_moves_both_clocks() -> None:
    clock = FakeClock(start_mono=10.0, start_utc=dt.datetime(2026, 1, 1, tzinfo=dt.UTC))
    clock.advance(1.5)
    assert clock.monotonic() == 11.5
    assert clock.monotonic_ms() == 11500
    assert clock.utc_now() == dt.datetime(2026, 1, 1, 0, 0, 1, 500000, tzinfo=dt.UTC)


async def test_sleepers_wake_in_order() -> None:
    clock = FakeClock()
    woke: list[tuple[str, float]] = []

    async def sleeper(name: str, seconds: float) -> None:
        await clock.sleep(seconds)
        woke.append((name, clock.monotonic()))

    tasks = [asyncio.create_task(sleeper("b", 2.0)), asyncio.create_task(sleeper("a", 1.0))]
    await asyncio.sleep(0)
    clock.advance(0.5)
    await asyncio.sleep(0)
    assert woke == []
    await clock.run_for(2.0)
    await asyncio.gather(*tasks)
    assert [name for name, _ in woke] == ["a", "b"]
    assert woke[0][1] == pytest.approx(1001.0, abs=0.011)


async def test_zero_sleep_only_yields() -> None:
    clock = FakeClock()
    await clock.sleep(0)
    assert clock.monotonic() == 1000.0


async def test_run_until_stops_when_done_or_times_out() -> None:
    clock = FakeClock()
    done = asyncio.Event()

    async def setter() -> None:
        await clock.sleep(2.0)
        done.set()

    task = asyncio.create_task(setter())
    await clock.run_until(done, limit_s=3.0)
    await task
    assert 1002.0 <= clock.monotonic() < 1002.1
    with pytest.raises(TimeoutError):
        await clock.run_until(asyncio.Event(), limit_s=0.1)


def test_both_clocks_satisfy_the_protocol() -> None:
    clocks: list[Clock] = [FakeClock(), MonotonicClock()]
    assert all(c.monotonic_ms() > 0 for c in clocks)


async def test_monotonic_clock_is_real_time() -> None:
    clock = MonotonicClock()
    before = clock.monotonic()
    await clock.sleep(0.01)
    assert clock.monotonic() > before
    assert clock.utc_now().tzinfo is not None
