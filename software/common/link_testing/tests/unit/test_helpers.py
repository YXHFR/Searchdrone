"""The test helpers themselves: fake clock, in-memory and lossy transports."""

from __future__ import annotations

import asyncio

import pytest

from sd_link.transport import TransportClosedError
from sd_link_testing import FakeClock, LossyTransport, transport_pair


async def test_fake_clock_moves_only_on_advance() -> None:
    clock = FakeClock(start=10.0)
    woke: list[float] = []

    async def sleeper(s: float) -> None:
        await clock.sleep(s)
        woke.append(clock.now())

    tasks = [asyncio.ensure_future(sleeper(s)) for s in (0.5, 0.2, 1.0)]
    await asyncio.sleep(0)
    await clock.advance(0.3)
    assert woke == [pytest.approx(10.2)]
    await clock.advance(1.0)
    assert woke == [pytest.approx(10.2), pytest.approx(10.5), pytest.approx(11.0)]
    await asyncio.gather(*tasks)
    await clock.sleep(0)  # zero sleep just yields
    assert (clock.utc_now() - FakeClock(start=10.0).utc_now()).total_seconds() == pytest.approx(1.3)


async def test_run_until() -> None:
    clock = FakeClock()
    done = asyncio.Event()

    async def later() -> None:
        await clock.sleep(2.0)
        done.set()

    task = asyncio.ensure_future(later())
    await clock.run_until(done, limit_s=3.0)
    assert done.is_set()
    await task
    with pytest.raises(TimeoutError):
        await clock.run_until(asyncio.Event(), limit_s=0.1)


async def test_memory_pair_delivers_and_closes_both_ends() -> None:
    a, b = transport_pair()
    await a.send("one")
    assert await b.recv() == "one" and a.sent == ["one"]
    await b.close(4409, "replaced")
    await b.close(1000)  # second close ignored
    assert a.close_code == b.close_code == 4409
    with pytest.raises(TransportClosedError) as err:
        await a.recv()
    assert err.value.code == 4409 and err.value.reason == "replaced"
    with pytest.raises(TransportClosedError):
        await a.send("late")
    with pytest.raises(TransportClosedError):
        await b.recv()


async def test_lossy_transport_drops_delays_and_keeps_order() -> None:
    clock = FakeClock()
    a, b = transport_pair()
    lossy = LossyTransport(a, clock, loss=0.5, delay_s=0.3, seed=1)
    for n in range(20):
        await lossy.send(str(n))
    await clock.advance(0.29)
    assert a.sent == []
    await clock.advance(0.02)
    got = [int(x) for x in a.sent]
    assert got == sorted(got) and 0 < len(got) < 20
    assert lossy.dropped == 20 - len(got)
    lossy.broken = True
    await lossy.send("x")
    await clock.advance(1.0)
    assert "x" not in a.sent
    await b.send("back")
    assert await lossy.recv() == "back"
    await lossy.close(1001)
    assert lossy.close_code == 1001
    with pytest.raises(TransportClosedError):
        await lossy.send("after")
