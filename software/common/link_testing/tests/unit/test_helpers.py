"""The test helpers themselves: in-memory and lossy transports."""

from __future__ import annotations

import pytest

from sd_link.transport import TransportClosedError
from sd_link_testing import LossyTransport, transport_pair
from sd_testing import FakeClock


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
    await clock.run_for(0.29)
    assert a.sent == []
    await clock.run_for(0.02)
    got = [int(x) for x in a.sent]
    assert got == sorted(got) and 0 < len(got) < 20
    assert lossy.dropped == 20 - len(got)
    lossy.broken = True
    await lossy.send("x")
    await clock.run_for(1.0)
    assert "x" not in a.sent
    await b.send("back")
    assert await lossy.recv() == "back"
    await lossy.close(1001)
    assert lossy.close_code == 1001
    with pytest.raises(TransportClosedError):
        await lossy.send("after")
