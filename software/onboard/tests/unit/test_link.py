"""sd-onboard's SD-link side against a real link server on loopback (no ground gate)."""

from __future__ import annotations

import asyncio

import pytest
from onboard_helpers import EXPECTED_PARAMS, charter

from sd_link_testing import FakeGround, next_inbound
from sd_mav.fake import FakeVehicle
from sd_obs.clock import MonotonicClock
from sd_onboard.executor.executor import Executor, ExecutorOptions
from sd_onboard.link_client import OnboardLink
from sd_schemas.base import new_id
from sd_schemas.gate import StateRef
from sd_schemas.link import (
    Ack,
    CharterAck,
    CharterInstall,
    OnboardEvent,
    SkillCommand,
    StateSnapshot,
)

pytestmark = pytest.mark.component


async def test_link_roundtrip() -> None:
    clock = MonotonicClock()
    async with FakeGround() as ground:
        holder: list[Executor] = []
        vehicle = FakeVehicle(
            clock,
            on_event=lambda k, d: holder[0].on_vehicle_event(k, d) if holder else None,
            send_gate=lambda o: holder[0].send_gate(o) if holder else "starting",
        )
        link = OnboardLink(url=ground.url, token=ground.token, vehicle_id="sim-01", clock=clock)
        ex = Executor(
            vehicle,
            clock,
            link,
            ExecutorOptions(vehicle_id="sim-01", expected_params=EXPECTED_PARAMS, require_rc=False),
        )
        holder.append(ex)
        link.executor = ex
        stop = asyncio.Event()
        await vehicle.start()
        tasks = [asyncio.ensure_future(ex.run(stop)), asyncio.ensure_future(link.run(stop))]
        try:
            session = await asyncio.wait_for(ground.sessions.get(), 10)
            assert ground.hellos[0].boot_id == ex.boot_id
            snap = await asyncio.wait_for(next_inbound(session, StateSnapshot), 5)
            assert snap.body.executor_state.name == "IDLE"  # type: ignore[attr-defined]
            await session.send(CharterInstall(charter=charter()))
            ack = await asyncio.wait_for(next_inbound(session, CharterAck), 10)
            assert ack.body.accepted  # type: ignore[attr-defined]
            event = await asyncio.wait_for(next_inbound(session, OnboardEvent), 5)
            assert event.body.type.startswith("executor.")  # type: ignore[attr-defined]
            cmd = SkillCommand(
                command_id=new_id(),
                skill="orbit",
                params={},
                issuer="operator:pilot",
                rationale=None,
                charter_version=1,
                based_on=StateRef(state_epoch=ex.boot_id, state_seq=0),
                approval=None,
            )
            await session.send(cmd, ttl_ms=15_000)
            while True:
                inbound = await asyncio.wait_for(next_inbound(session, Ack), 5)
                body = inbound.body
                assert isinstance(body, Ack)
                if body.ref_id == cmd.command_id and body.stage == "rejected":
                    assert body.reason_codes == ["SKILL_NOT_IMPLEMENTED"]
                    break
        finally:
            stop.set()
            for task in tasks:
                task.cancel()
            await asyncio.gather(*tasks, return_exceptions=True)
            await ex.close()
            await vehicle.close()
