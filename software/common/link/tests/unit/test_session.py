"""A session over an in-memory pipe with a fake clock."""

from __future__ import annotations

import asyncio
import json

import pytest
from link_helpers import VEHICLE, clearance, goto, sticks, stop_notice, takeover

from sd_link.close_codes import CloseCode
from sd_link.codes import LINK_NO_ACK, SCHEMA_INVALID
from sd_link.envelope import GROUND_SENDER, build_envelope, encode
from sd_link.session import (
    Closed,
    Expired,
    Inbound,
    LinkBusyError,
    LinkClosedError,
    LinkEvent,
    LinkSession,
    LinkState,
    StateChanged,
)
from sd_link_testing import FakeClock, FakeLink, LossyTransport, next_inbound, transport_pair
from sd_schemas.base import new_id
from sd_schemas.link import Ack, Heartbeat, PilotPosition, SkillCommand, StateSnapshot


def kinds(frames: list[str]) -> list[str]:
    return [json.loads(f)["kind"] for f in frames]


def bodies(frames: list[str], kind: str) -> list[dict[str, object]]:
    return [json.loads(f)["body"] for f in frames if json.loads(f)["kind"] == kind]


async def drain(session: LinkSession) -> list[LinkEvent]:
    events: list[LinkEvent] = []
    while not session._events.empty():
        events.append(session._events.get_nowait())
    return events


async def test_reliable_message_is_acknowledged_received() -> None:
    clock = FakeClock()
    async with FakeLink(clock) as link:
        cmd = goto()
        sent = await link.ground.send(cmd, ttl_ms=15_000)
        got = await next_inbound(link.vehicle, SkillCommand)
        assert got.envelope.msg_id == sent.msg_id and got.body == cmd
        assert got.deadline == pytest.approx(got.received_at + 15.0)
        await clock.advance(0.1)
        assert link.ground.pending_count == 0
        acks = bodies(link.vehicle_transport.sent, "Ack")
        assert acks == [
            {
                "ref_id": str(cmd.command_id),
                "stage": "received",
                "reason_codes": [],
                "detail_sv": None,
                "gate": None,
            }
        ]


async def test_heartbeat_both_ways_one_hz_never_acknowledged() -> None:
    clock = FakeClock()
    async with FakeLink(clock) as link:
        await clock.advance(3.0)
        g, v = kinds(link.ground_transport.sent), kinds(link.vehicle_transport.sent)
        assert g.count("Heartbeat") in (3, 4) and v.count("Heartbeat") in (3, 4)
        assert "Ack" not in g + v
        assert link.ground.state is LinkState.UP and link.vehicle.state is LinkState.UP
        assert all(not isinstance(e, Inbound) for e in await drain(link.ground))


async def test_app_ack_is_remembered_and_terminal_ack_is_reliable() -> None:
    clock = FakeClock()
    async with FakeLink(clock) as link:
        cmd = goto()
        sent = await link.ground.send(cmd, ttl_ms=15_000)
        await next_inbound(link.vehicle, SkillCommand)
        await link.vehicle.ack(cmd.command_id, "running")
        done = await link.vehicle.ack(cmd.command_id, "done")
        assert done.reliable and done.corr_id == cmd.command_id
        got = [await next_inbound(link.ground, Ack) for _ in range(3)]
        stages = [g.body.stage for g in got]  # type: ignore[union-attr]
        assert stages == ["received", "running", "done"]
        await clock.advance(0.1)
        assert link.vehicle.pending_count == 0  # ground acknowledged the terminal Ack
        # A duplicate of the order now gets the latest Ack ("done"), not a new effect.
        await link.ground_transport.send(encode(sent))
        await clock.advance(0.1)
        repeated = bodies(link.vehicle_transport.sent, "Ack")[-1]
        assert repeated["stage"] == "done"
        assert all(
            not (isinstance(e, Inbound) and isinstance(e.body, SkillCommand))
            for e in await drain(link.vehicle)
        )


async def test_duplicate_gets_same_ack_and_no_new_effect() -> None:
    clock = FakeClock()
    async with FakeLink(clock) as link:
        sent = await link.ground.send(stop_notice(), ttl_ms=None)
        first = await next_inbound(link.vehicle)
        assert first.envelope.msg_id == sent.msg_id
        for _ in range(3):
            await link.ground_transport.send(encode(sent))
        await clock.advance(0.1)
        acks = bodies(link.vehicle_transport.sent, "Ack")
        assert len(acks) == 4 and all(a == acks[0] for a in acks)
        assert not [
            e
            for e in await drain(link.vehicle)
            if isinstance(e, Inbound) and not isinstance(e.body, Ack)
        ]


async def _lossy_pair(clock: FakeClock, **kw: float) -> tuple[LossyTransport, LossyTransport]:
    g, v = transport_pair()
    return LossyTransport(g, clock, **kw), LossyTransport(v, clock, **kw)  # type: ignore[arg-type]


async def test_lost_send_is_resent_after_one_second_with_same_msg_id() -> None:
    clock = FakeClock()
    g, v = await _lossy_pair(clock)
    ground = LinkSession(g, side="ground", vehicle_id=VEHICLE, clock=clock)
    vehicle = LinkSession(v, side="vehicle", vehicle_id=VEHICLE, clock=clock)
    tasks = [asyncio.ensure_future(ground.run()), asyncio.ensure_future(vehicle.run())]
    g.broken = True
    sent = await ground.send(goto(), ttl_ms=15_000)
    await clock.advance(0.5)
    g.broken = False
    await clock.advance(0.6)
    got = await next_inbound(vehicle, SkillCommand)
    assert got.envelope.msg_id == sent.msg_id
    assert got.envelope.seq > sent.seq
    assert 13_900 <= (got.envelope.ttl_ms or 0) <= 14_000
    await clock.advance(0.2)
    assert ground.pending_count == 0
    await ground.close()
    await asyncio.gather(*tasks)


async def test_no_ack_expires_after_three_resends() -> None:
    clock = FakeClock()
    g, v = await _lossy_pair(clock)
    ground = LinkSession(g, side="ground", vehicle_id=VEHICLE, clock=clock, lost_after_s=60)
    vehicle = LinkSession(v, side="vehicle", vehicle_id=VEHICLE, clock=clock, lost_after_s=60)
    tasks = [asyncio.ensure_future(ground.run()), asyncio.ensure_future(vehicle.run())]
    g.broken = True
    sent = await ground.send(stop_notice())
    await clock.advance(3.9)
    assert ground.pending_count == 1
    await clock.advance(0.2)
    assert ground.pending_count == 0
    events = await drain(ground)
    expired = [e for e in events if isinstance(e, Expired)]
    assert expired == [Expired(expired[0].envelope, LINK_NO_ACK)]
    assert expired[0].envelope.msg_id == sent.msg_id
    assert g.dropped >= 4  # the send and three resends (plus heartbeats)
    await ground.close()
    await asyncio.gather(*tasks)


async def test_link_states_degraded_then_lost_and_closed_1001() -> None:
    clock = FakeClock()
    g, v = await _lossy_pair(clock)
    ground = LinkSession(g, side="ground", vehicle_id=VEHICLE, clock=clock)
    vehicle = LinkSession(v, side="vehicle", vehicle_id=VEHICLE, clock=clock)
    tasks = [asyncio.ensure_future(ground.run()), asyncio.ensure_future(vehicle.run())]
    v.broken = True  # nothing from the vehicle ever reaches the ground
    pending = await ground.send(stop_notice())
    await clock.advance(2.9)
    assert ground.state is LinkState.UP
    await clock.advance(0.2)
    assert ground.state is LinkState.DEGRADED
    await clock.advance(6.8)
    assert ground.state is LinkState.DEGRADED
    await clock.advance(0.2)
    await asyncio.gather(tasks[0])
    assert ground.state is LinkState.CLOSED
    events = await drain(ground)
    states = [e.state for e in events if isinstance(e, StateChanged)]
    assert states == [LinkState.DEGRADED, LinkState.LOST, LinkState.CLOSED]
    closed = events[-1]
    assert isinstance(closed, Closed) and closed.code == CloseCode.GOING_AWAY
    assert pending.msg_id not in {e.msg_id for e in closed.unacked}  # expired at 4 s
    with pytest.raises(LinkClosedError):
        await ground.send(stop_notice())
    await asyncio.gather(*tasks)


async def test_traffic_brings_degraded_back_up() -> None:
    clock = FakeClock()
    g, v = await _lossy_pair(clock)
    ground = LinkSession(g, side="ground", vehicle_id=VEHICLE, clock=clock)
    vehicle = LinkSession(v, side="vehicle", vehicle_id=VEHICLE, clock=clock)
    tasks = [asyncio.ensure_future(ground.run()), asyncio.ensure_future(vehicle.run())]
    v.broken = True
    await clock.advance(3.2)
    assert ground.state is LinkState.DEGRADED
    v.broken = False
    await clock.advance(1.1)
    assert ground.state is LinkState.UP
    await ground.close()
    await asyncio.gather(*tasks)


async def test_unacked_messages_are_handed_back_on_close_never_resent() -> None:
    clock = FakeClock()
    g, v = await _lossy_pair(clock)
    ground = LinkSession(g, side="ground", vehicle_id=VEHICLE, clock=clock)
    vehicle = LinkSession(v, side="vehicle", vehicle_id=VEHICLE, clock=clock)
    tasks = [asyncio.ensure_future(ground.run()), asyncio.ensure_future(vehicle.run())]
    g.broken = True
    order = await ground.send(goto(), ttl_ms=15_000)
    clear = await ground.send(clearance(), ttl_ms=5000)
    await vehicle.close(CloseCode.GOING_AWAY, "reboot")
    await asyncio.gather(*tasks)
    closed = next(e for e in await drain(ground) if isinstance(e, Closed))
    assert [e.msg_id for e in closed.unacked] == [order.msg_id, clear.msg_id]
    assert closed.code == CloseCode.GOING_AWAY and closed.reason == "reboot"
    # A new session on a new connection starts empty: nothing old goes out again.
    g2, _ = transport_pair()
    fresh = LinkSession(g2, side="ground", vehicle_id=VEHICLE, clock=clock)
    assert fresh.pending_count == 0


async def test_manual_control_is_not_reliable_and_never_resent() -> None:
    clock = FakeClock()
    async with FakeLink(clock) as link:
        await link.ground.send(takeover(), ttl_ms=None)
        for i in range(10):
            env = await link.ground.send(sticks(i), ttl_ms=1000)
            assert not env.reliable
        assert link.ground.pending_count == 1  # only ManualTakeover
        got = [await next_inbound(link.vehicle) for _ in range(11)]
        assert got[0].body.__class__.__name__ == "ManualTakeover"
        last = got[-1]
        assert last.deadline == pytest.approx(last.received_at + 1.0)
        assert not last.expired(clock.now()) and last.expired(clock.now() + 1.0)
        await clock.advance(5.0)
        assert kinds(link.ground_transport.sent).count("ManualControl") == 10


async def test_send_refuses_wrong_direction_and_handshake_kinds() -> None:
    clock = FakeClock()
    async with FakeLink(clock) as link:
        with pytest.raises(ValueError, match="may not send"):
            await link.vehicle.send(goto(), ttl_ms=1000)
        with pytest.raises(ValueError, match="may not send"):
            await link.ground.send(
                StateSnapshot.model_construct()  # never reaches validation
            )
        from link_helpers import hello

        with pytest.raises(ValueError, match="may not send"):
            await link.vehicle.send(hello())


async def test_inbound_wrong_direction_is_rejected() -> None:
    clock = FakeClock()
    async with FakeLink(clock) as link:
        # The "vehicle" side sends an order (not allowed) by writing a raw frame.
        bad = build_envelope(
            goto(),
            sender="vehicle:sim-01",
            vehicle_id=VEHICLE,
            seq=5,
            sent_utc=clock.utc_now(),
            ttl_ms=1000,
        )
        await link.vehicle_transport.send(encode(bad))
        await clock.advance(0.1)
        rejected = bodies(link.ground_transport.sent, "Ack")
        assert rejected[-1]["stage"] == "rejected"
        assert rejected[-1]["reason_codes"] == [SCHEMA_INVALID]
        assert rejected[-1]["ref_id"] == str(bad.msg_id)
        delivered = [e for e in await drain(link.ground) if isinstance(e, Inbound)]
        assert not [e for e in delivered if not isinstance(e.body, Ack)]


async def test_invalid_reliable_frame_is_rejected_unreliable_ignored() -> None:
    clock = FakeClock()
    async with FakeLink(clock) as link:
        raw = json.loads(
            encode(
                build_envelope(
                    stop_notice(),
                    sender=GROUND_SENDER,
                    vehicle_id=VEHICLE,
                    seq=9,
                    sent_utc=clock.utc_now(),
                )
            )
        )
        raw["body"]["kind"] = "selfdestruct"
        await link.ground_transport.send(json.dumps(raw))
        await link.ground_transport.send("garbage")
        await clock.advance(0.1)
        acks = bodies(link.vehicle_transport.sent, "Ack")
        assert len(acks) == 1 and acks[0]["reason_codes"] == [SCHEMA_INVALID]
        assert not [
            e
            for e in await drain(link.vehicle)
            if isinstance(e, Inbound) and not isinstance(e.body, Ack)
        ]


async def test_sender_mismatch_closes_forbidden() -> None:
    clock = FakeClock()
    async with FakeLink(clock) as link:
        other = build_envelope(
            PilotPosition(lat=-35.36, lon=149.16, source="charter", accuracy_m=None),
            sender=GROUND_SENDER,
            vehicle_id="sim-02",
            seq=1,
            sent_utc=clock.utc_now(),
        )
        await link.ground_transport.send(encode(other))
        await clock.advance(0.1)
        assert link.vehicle.state is LinkState.CLOSED
        assert link.vehicle_transport.close_code == CloseCode.FORBIDDEN


async def test_busy_when_too_much_is_unacknowledged(monkeypatch: pytest.MonkeyPatch) -> None:
    import sd_link.session as mod

    monkeypatch.setattr(mod, "MAX_PENDING_RELIABLE", 2)
    clock = FakeClock()
    g, _ = await _lossy_pair(clock)
    g.broken = True
    ground = LinkSession(g, side="ground", vehicle_id=VEHICLE, clock=clock)
    await ground.send(stop_notice(1))
    await ground.send(stop_notice(2))
    with pytest.raises(LinkBusyError):
        await ground.send(stop_notice(3))
    await ground.send(Heartbeat())  # unreliable still goes
    await ground.close()
    await ground.close()  # twice is fine


async def test_ack_unknown_ref_is_harmless() -> None:
    clock = FakeClock()
    async with FakeLink(clock) as link:
        await link.vehicle.ack(new_id(), "done")
        got = await next_inbound(link.ground, Ack)
        assert got.body.stage == "done"  # type: ignore[union-attr]
