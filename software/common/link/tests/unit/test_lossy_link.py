"""100 orders over a link with 10 % loss and 300 ms delay both ways (fake clock)."""

from __future__ import annotations

import asyncio

from link_helpers import VEHICLE, goto

from sd_link.session import Closed, Expired, Inbound, LinkSession
from sd_link_testing import LossyTransport, transport_pair
from sd_schemas.link import SkillCommand
from sd_testing import FakeClock


async def test_hundred_orders_in_order_and_acknowledged() -> None:
    clock = FakeClock()
    g, v = transport_pair()
    g_lossy = LossyTransport(g, clock, loss=0.10, delay_s=0.3, seed=4)
    v_lossy = LossyTransport(v, clock, loss=0.10, delay_s=0.3, seed=5)
    ground = LinkSession(g_lossy, side="ground", vehicle_id=VEHICLE, clock=clock)
    vehicle = LinkSession(v_lossy, side="vehicle", vehicle_id=VEHICLE, clock=clock)
    tasks = [asyncio.ensure_future(ground.run()), asyncio.ensure_future(vehicle.run())]

    received: list[int] = []
    expired: list[object] = []

    async def onboard() -> None:
        while True:
            event = await vehicle.next_event()
            if isinstance(event, Closed):
                return
            if isinstance(event, Inbound) and isinstance(event.body, SkillCommand):
                received.append(event.body.based_on.state_seq)  # type: ignore[union-attr]
                await vehicle.ack(event.body.command_id, "accepted")

    consumer = asyncio.ensure_future(onboard())
    acked = 0
    for n in range(100):
        cmd = goto(n)
        await ground.send(cmd, ttl_ms=15_000)
        # One movement order at a time: wait until an Ack stops its resends.
        for _ in range(200):
            await clock.run_for(0.05)
            expired += [e for e in ground.events_nowait() if isinstance(e, Expired)]
            if ground.pending_count == 0:
                acked += 1
                break

    assert not expired
    assert acked == 100
    assert received == list(range(100))  # each order exactly once, in order
    assert g_lossy.dropped > 0 and v_lossy.dropped > 0  # the loss really happened
    await ground.close()
    await asyncio.gather(*tasks, consumer)
