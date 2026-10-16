"""A session over an in-memory pipe with a fake clock."""

from __future__ import annotations

import asyncio
import json
from collections.abc import AsyncIterator
from contextlib import asynccontextmanager

import pytest
from link_helpers import VEHICLE, clearance, goto, hello, sticks, stop_notice, takeover

from sd_link.close_codes import CloseCode
from sd_link.codes import LINK_NO_ACK, SCHEMA_INVALID
from sd_link.constants import MAX_INVALID_FRAMES
from sd_link.envelope import GROUND_SENDER, build_envelope, encode, vehicle_sender
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
from sd_link.transport import ABORT_CODE
from sd_link_testing import (
    FakeClock,
    FakeLink,
    LossyTransport,
    MemoryTransport,
    next_inbound,
    transport_pair,
)
from sd_schemas.base import SdModel, new_id
from sd_schemas.link import (
    Ack,
    Heartbeat,
    ManualControl,
    ManualTakeover,
    PilotPosition,
    SkillCommand,
    StateSnapshot,
)


def kinds(frames: list[str]) -> list[str]:
    return [json.loads(f)["kind"] for f in frames]


def bodies(frames: list[str], kind: str) -> list[dict[str, object]]:
    return [json.loads(f)["body"] for f in frames if json.loads(f)["kind"] == kind]


def app_messages(events: list[LinkEvent]) -> list[Inbound]:
    """Delivered messages that are not Acks."""
    return [e for e in events if isinstance(e, Inbound) and not isinstance(e.body, Ack)]


class Pair:
    """Ground and vehicle sessions over lossy in-memory transports, both running."""

    def __init__(self, clock: FakeClock, **kw: float) -> None:
        g, v = transport_pair()
        self.g_raw, self.v_raw = g, v
        self.g = LossyTransport(g, clock, **kw)  # type: ignore[arg-type]
        self.v = LossyTransport(v, clock, **kw)  # type: ignore[arg-type]
        self.ground = LinkSession(self.g, side="ground", vehicle_id=VEHICLE, clock=clock)
        self.vehicle = LinkSession(self.v, side="vehicle", vehicle_id=VEHICLE, clock=clock)
        self.tasks = [
            asyncio.ensure_future(self.ground.run()),
            asyncio.ensure_future(self.vehicle.run()),
        ]

    async def stop(self) -> None:
        self.ground.abort()
        await asyncio.gather(*self.tasks)


@asynccontextmanager
async def pair(clock: FakeClock, **kw: float) -> AsyncIterator[Pair]:
    p = Pair(clock, **kw)
    try:
        yield p
    finally:
        await p.stop()


def frame_from(body: SdModel, side: str, clock: FakeClock, **kw: object) -> str:
    sender = GROUND_SENDER if side == "ground" else vehicle_sender(VEHICLE)
    env = build_envelope(
        body,
        sender=sender,
        vehicle_id=VEHICLE,
        seq=99,
        sent_utc=clock.utc_now(),
        **kw,  # type: ignore[arg-type]
    )
    return encode(env)


# --- acknowledgements and duplicates ----------------------------------------------


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
        assert bodies(link.vehicle_transport.sent, "Ack") == [
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
        assert link.ground.state is LinkState.UP
        assert not [e for e in link.ground.events_nowait() if isinstance(e, Inbound)]


async def test_app_acks_terminal_reliable_and_duplicate_gets_the_same_frame() -> None:
    clock = FakeClock()
    async with FakeLink(clock) as link:
        cmd = goto()
        sent = await link.ground.send(cmd, ttl_ms=15_000)
        await next_inbound(link.vehicle, SkillCommand)
        await link.vehicle.ack(cmd.command_id, "running")
        done = await link.vehicle.ack(cmd.command_id, "done")
        assert done.reliable and done.corr_id == cmd.command_id
        stages = [(await next_inbound(link.ground, Ack)).body.stage for _ in range(3)]  # type: ignore[union-attr]
        assert stages == ["received", "running", "done"]
        await clock.advance(0.1)
        assert link.vehicle.pending_count == 0  # the ground acknowledged the terminal Ack
        before = len(link.vehicle_transport.sent)
        await link.ground_transport.send(encode(sent))  # duplicate of the order
        await clock.advance(0.1)
        repeated = link.vehicle_transport.sent[before:]
        assert repeated == [encode(done)]  # exactly the same frame, not a new message
        assert link.vehicle.pending_count == 0  # and not tracked again
        assert not app_messages(link.vehicle.events_nowait())


async def test_duplicate_gets_same_ack_and_no_new_effect() -> None:
    clock = FakeClock()
    async with FakeLink(clock) as link:
        sent = await link.ground.send(stop_notice())
        first = await next_inbound(link.vehicle)
        assert first.envelope.msg_id == sent.msg_id
        for _ in range(3):
            await link.ground_transport.send(encode(sent))
        await clock.advance(0.1)
        acks = bodies(link.vehicle_transport.sent, "Ack")
        assert len(acks) == 4 and all(a == acks[0] for a in acks)
        assert not app_messages(link.vehicle.events_nowait())


async def test_same_command_id_under_a_new_msg_id_is_a_duplicate() -> None:
    clock = FakeClock()
    async with FakeLink(clock) as link:
        cmd = goto()
        await link.ground.send(cmd, ttl_ms=15_000)
        await next_inbound(link.vehicle, SkillCommand)
        await clock.advance(0.1)
        await link.ground_transport.send(frame_from(cmd, "ground", clock, ttl_ms=15_000))
        await clock.advance(0.1)
        assert not app_messages(link.vehicle.events_nowait())


async def test_dedup_survives_into_a_new_session_with_the_same_cache() -> None:
    from sd_link.reliability import DedupCache

    clock = FakeClock()
    cache = DedupCache()
    order = frame_from(stop_notice(), "ground", clock)
    delivered: list[int] = []
    for _ in range(2):
        g, v = transport_pair()
        vehicle = LinkSession(v, side="vehicle", vehicle_id=VEHICLE, clock=clock, dedup=cache)
        task = asyncio.ensure_future(vehicle.run())
        await g.send(order)
        await clock.advance(0.1)
        delivered.append(len(app_messages(vehicle.events_nowait())))
        vehicle.abort()
        await task
    assert delivered == [1, 0]  # the second session recognised the first one's message


# --- resend, expiry, liveness -----------------------------------------------------


async def test_lost_send_is_resent_after_one_second_with_same_msg_id() -> None:
    clock = FakeClock()
    async with pair(clock) as p:
        p.g.broken = True
        sent = await p.ground.send(goto(), ttl_ms=15_000)
        await clock.advance(0.5)
        p.g.broken = False
        await clock.advance(0.6)
        got = await next_inbound(p.vehicle, SkillCommand)
        assert got.envelope.msg_id == sent.msg_id and got.envelope.seq > sent.seq
        assert 13_900 <= (got.envelope.ttl_ms or 0) <= 14_000
        await clock.advance(0.2)
        assert p.ground.pending_count == 0


async def test_no_ack_expires_after_three_resends_and_aborts_the_connection() -> None:
    clock = FakeClock()
    async with pair(clock) as p:
        p.g.broken = True
        sent = await p.ground.send(stop_notice())
        other = await p.ground.send(goto(), ttl_ms=15_000)
        await clock.advance(3.9)
        assert p.ground.pending_count == 2
        await clock.advance(0.2)
        await asyncio.gather(p.tasks[0])
        events = p.ground.events_nowait()
        expired = [e for e in events if isinstance(e, Expired)]
        assert [e.envelope.msg_id for e in expired] == [sent.msg_id, other.msg_id]
        assert all(e.reason == LINK_NO_ACK for e in expired)
        closed = events[-1]
        assert isinstance(closed, Closed) and closed.code == ABORT_CODE
        assert closed.unacked == ()  # every message reported exactly once
        assert p.g_raw.aborted


async def test_expired_order_never_arrives_late() -> None:
    """A dropout of 4-9 s: the frame sits in a buffer while the ground expires it."""
    clock = FakeClock()
    async with pair(clock, delay_s=6.0) as p:
        sent = await p.ground.send(goto(), ttl_ms=15_000)
        await clock.advance(8.0)
        events = p.ground.events_nowait()
        assert [e.envelope.msg_id for e in events if isinstance(e, Expired)] == [sent.msg_id]
        assert not app_messages(p.vehicle.events_nowait())  # dropped with the connection


async def test_every_tracked_message_ends_in_expired_or_unacked() -> None:
    clock = FakeClock()
    async with pair(clock) as p:
        p.g.broken = True
        first = await p.ground.send(stop_notice(1))
        await clock.advance(1.5)
        second = await p.ground.send(stop_notice(2))
        await clock.advance(1.0)
        p.vehicle.abort()  # the connection drops while both are waiting
        await asyncio.gather(*p.tasks)
        events = p.ground.events_nowait()
        closed = events[-1]
        assert isinstance(closed, Closed)
        reported = [e.envelope.msg_id for e in events if isinstance(e, Expired)]
        reported += [e.msg_id for e in closed.unacked]
        assert sorted(reported) == sorted([first.msg_id, second.msg_id])


async def test_link_states_degraded_then_lost_and_closed_1001() -> None:
    clock = FakeClock()
    async with pair(clock) as p:
        p.v.broken = True  # nothing from the vehicle ever reaches the ground
        await clock.advance(2.9)
        assert p.ground.state is LinkState.UP
        await clock.advance(0.2)
        assert p.ground.state is LinkState.DEGRADED
        await clock.advance(6.8)
        assert p.ground.state is LinkState.DEGRADED
        await clock.advance(0.2)
        await asyncio.gather(p.tasks[0])
        events = p.ground.events_nowait()
        states = [e.state for e in events if isinstance(e, StateChanged)]
        assert states == [LinkState.DEGRADED, LinkState.LOST, LinkState.CLOSED]
        assert isinstance(events[-1], Closed) and events[-1].code == CloseCode.GOING_AWAY
        with pytest.raises(LinkClosedError):
            await p.ground.send(stop_notice())


async def test_traffic_brings_degraded_back_up() -> None:
    clock = FakeClock()
    async with pair(clock) as p:
        p.v.broken = True
        await clock.advance(3.2)
        assert p.ground.state is LinkState.DEGRADED
        p.v.broken = False
        await clock.advance(1.1)
        assert p.ground.state is LinkState.UP


async def test_liveness_and_expiry_continue_while_a_write_is_blocked() -> None:
    class Stuck(MemoryTransport):
        stuck = False

        async def send(self, text: str) -> None:
            if self.stuck:
                await asyncio.Event().wait()
            await super().send(text)

    clock = FakeClock()
    g, v = Stuck(), MemoryTransport()
    g.peer, v.peer = v, g
    ground = LinkSession(g, side="ground", vehicle_id=VEHICLE, clock=clock, degraded_after_s=1.5)
    task = asyncio.ensure_future(ground.run())
    await clock.advance(0.1)
    g.stuck = True
    await clock.advance(1.0)  # the next heartbeat write now blocks
    await clock.advance(1.5)
    assert ground.state is LinkState.DEGRADED  # the watch task still runs
    await clock.run_until(task, limit_s=1.0)  # WRITE_TIMEOUT_S after the blocked write
    events = ground.events_nowait()
    assert isinstance(events[-1], Closed) and events[-1].reason == "write blocked"
    assert g.aborted


async def test_unacked_messages_are_handed_back_on_close_never_resent() -> None:
    clock = FakeClock()
    async with pair(clock) as p:
        p.g.broken = True
        order = await p.ground.send(goto(), ttl_ms=15_000)
        clear = await p.ground.send(clearance(), ttl_ms=5000)
        await p.vehicle.close(CloseCode.GOING_AWAY, "reboot")
        await asyncio.gather(*p.tasks)
        closed = next(e for e in p.ground.events_nowait() if isinstance(e, Closed))
        assert [e.msg_id for e in closed.unacked] == [order.msg_id, clear.msg_id]
        assert closed.code == CloseCode.GOING_AWAY and closed.reason == "reboot"


async def test_send_waiting_for_the_lock_when_the_session_ends_raises() -> None:
    clock = FakeClock()
    g, v = transport_pair()
    ground = LinkSession(g, side="ground", vehicle_id=VEHICLE, clock=clock)
    task = asyncio.ensure_future(ground.run())
    await ground._send_lock.acquire()  # a write in progress
    waiting = asyncio.ensure_future(ground.send(stop_notice()))
    await asyncio.sleep(0)
    await v.close(1001)
    await task  # the session has finished and drained
    ground._send_lock.release()
    with pytest.raises(LinkClosedError):
        await waiting
    assert ground.pending_count == 0


async def test_a_crashing_loop_aborts_the_session(monkeypatch: pytest.MonkeyPatch) -> None:
    clock = FakeClock()
    g, _ = transport_pair()
    ground = LinkSession(g, side="ground", vehicle_id=VEHICLE, clock=clock)

    async def boom() -> None:
        raise RuntimeError("bug")

    monkeypatch.setattr(ground, "tick", boom)
    task = asyncio.ensure_future(ground.run())
    await clock.advance(0.1)
    await task
    closed = ground.events_nowait()[-1]
    assert isinstance(closed, Closed) and closed.reason == "tick failed"


# --- manual control ---------------------------------------------------------------


async def test_stick_values_are_best_effort_never_tracked_or_resent() -> None:
    clock = FakeClock()
    async with FakeLink(clock) as link:
        await link.ground.send(takeover())
        for i in range(10):
            assert await link.ground.offer(sticks(i), ttl_ms=1000)
        assert link.ground.pending_count <= 1  # at most the ManualTakeover
        first = await next_inbound(link.vehicle)
        assert isinstance(first.body, ManualTakeover)
        latest = await next_inbound(link.vehicle, ManualControl)
        assert latest.body.input_seq == 9  # type: ignore[union-attr]  # coalesced to the newest
        assert latest.deadline == pytest.approx(latest.received_at + 1.0)
        assert not latest.expired(clock.now()) and latest.expired(clock.now() + 1.0)
        await clock.advance(5.0)
        assert kinds(link.ground_transport.sent).count("ManualControl") == 10
        with pytest.raises(ValueError, match="offer"):
            await link.ground.send(sticks(), ttl_ms=1000)
        with pytest.raises(ValueError, match="best-effort"):
            await link.ground.offer(stop_notice(), ttl_ms=None)


async def test_stick_values_are_dropped_unless_the_link_is_up() -> None:
    clock = FakeClock()
    async with pair(clock) as p:
        p.v.broken = True
        await clock.advance(3.2)
        assert p.ground.state is LinkState.DEGRADED
        assert not await p.ground.offer(sticks(), ttl_ms=1000)
        async with p.ground._send_lock:  # writer busy
            p.v.broken = False
            await clock.advance(1.1)
            assert p.ground.state is LinkState.UP
            assert not await p.ground.offer(sticks(), ttl_ms=1000)
        assert await p.ground.offer(sticks(), ttl_ms=1000)


async def test_a_burst_of_old_stick_values_after_a_stall_is_dropped() -> None:
    clock = FakeClock()
    g, v = transport_pair()
    vehicle = LinkSession(v, side="vehicle", vehicle_id=VEHICLE, clock=clock)
    task = asyncio.ensure_future(vehicle.run())
    await g.send(frame_from(sticks(1), "ground", clock, ttl_ms=1000))
    await clock.advance(0.05)
    assert len(app_messages(vehicle.events_nowait())) == 1
    await clock.advance(2.0)  # the stream stalls (TCP holds the values)
    for i in range(2, 30):  # ... then delivers them all at once
        await g.send(frame_from(sticks(i), "ground", clock, ttl_ms=1000))
    await clock.advance(0.05)
    assert app_messages(vehicle.events_nowait()) == []
    await clock.advance(1.0)
    await g.send(frame_from(sticks(40), "ground", clock, ttl_ms=1000))
    await clock.advance(0.05)
    fresh = app_messages(vehicle.events_nowait())
    assert [m.body.input_seq for m in fresh] == [40]  # type: ignore[union-attr]
    vehicle.abort()
    await task


# --- what a peer may send ---------------------------------------------------------


async def test_send_refuses_wrong_direction_and_handshake_kinds() -> None:
    clock = FakeClock()
    async with FakeLink(clock) as link:
        with pytest.raises(ValueError, match="may not send"):
            await link.vehicle.send(goto(), ttl_ms=1000)
        with pytest.raises(ValueError, match="may not send"):
            await link.ground.send(StateSnapshot.model_construct())
        with pytest.raises(ValueError, match="may not send"):
            await link.vehicle.send(hello())


async def test_inbound_wrong_direction_is_rejected() -> None:
    clock = FakeClock()
    async with FakeLink(clock) as link:
        bad = frame_from(goto(), "vehicle", clock, ttl_ms=1000)
        await link.vehicle_transport.send(bad)
        await clock.advance(0.1)
        rejected = bodies(link.ground_transport.sent, "Ack")[-1]
        assert rejected["stage"] == "rejected" and rejected["reason_codes"] == [SCHEMA_INVALID]
        assert rejected["ref_id"] == json.loads(bad)["msg_id"]
        assert not app_messages(link.ground.events_nowait())


async def test_invalid_reliable_frame_is_rejected_unreliable_ignored() -> None:
    clock = FakeClock()
    async with FakeLink(clock) as link:
        raw = json.loads(frame_from(stop_notice(), "ground", clock))
        raw["body"]["kind"] = "selfdestruct"
        await link.ground_transport.send(json.dumps(raw))
        await link.ground_transport.send("garbage")
        await clock.advance(0.1)
        acks = bodies(link.vehicle_transport.sent, "Ack")
        assert len(acks) == 1 and acks[0]["reason_codes"] == [SCHEMA_INVALID]
        assert not app_messages(link.vehicle.events_nowait())


async def test_receiver_enforces_ttl_rules() -> None:
    clock = FakeClock()
    async with FakeLink(clock) as link:
        raw = json.loads(frame_from(clearance(), "ground", clock, ttl_ms=5000))
        raw["ttl_ms"] = None  # a clearance that would never expire
        await link.ground_transport.send(json.dumps(raw))
        await clock.advance(0.1)
        assert bodies(link.vehicle_transport.sent, "Ack")[-1]["stage"] == "rejected"
        assert not app_messages(link.vehicle.events_nowait())


async def test_too_many_invalid_frames_close_the_connection() -> None:
    clock = FakeClock()
    async with FakeLink(clock) as link:
        for _ in range(MAX_INVALID_FRAMES + 1):
            await link.ground_transport.send("garbage")
        await clock.advance(0.1)
        assert link.vehicle.state is LinkState.CLOSED
        assert link.vehicle_transport.close_code == CloseCode.POLICY_VIOLATION


async def test_unread_events_beyond_the_backlog_close_the_connection(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    import sd_link.session as mod

    monkeypatch.setattr(mod, "MAX_EVENT_BACKLOG", 5)
    clock = FakeClock()
    async with FakeLink(clock) as link:
        for _ in range(6):
            pos = PilotPosition(lat=-35.36, lon=149.16, source="charter", accuracy_m=None)
            await link.ground.send(pos)
        await clock.advance(0.1)
        assert link.vehicle.state is LinkState.CLOSED
        assert link.vehicle_transport.close_code == CloseCode.POLICY_VIOLATION


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


async def test_busy_counts_only_application_messages(monkeypatch: pytest.MonkeyPatch) -> None:
    import sd_link.session as mod

    monkeypatch.setattr(mod, "MAX_PENDING_RELIABLE", 2)
    clock = FakeClock()
    g, _ = transport_pair()
    ground = LinkSession(g, side="ground", vehicle_id=VEHICLE, clock=clock)
    first = await ground.send(stop_notice(1))
    await ground.send(stop_notice(2))
    seq_before = first.seq + 2
    with pytest.raises(LinkBusyError):
        await ground.send(stop_notice(3))
    beat = await ground.send(Heartbeat())  # unreliable still goes
    assert beat.seq == seq_before  # the refused send left no gap in seq
    await ground.close()
    await ground.close()  # twice is fine


async def test_close_gives_up_on_a_peer_that_never_answers() -> None:
    clock = FakeClock()
    g, _ = transport_pair()
    g.block_close = True
    ground = LinkSession(g, side="ground", vehicle_id=VEHICLE, clock=clock)
    closing = asyncio.ensure_future(ground.close())
    await clock.advance(1.1)
    await closing
    assert g.aborted


async def test_ack_unknown_ref_is_harmless() -> None:
    clock = FakeClock()
    async with FakeLink(clock) as link:
        await link.vehicle.ack(new_id(), "done")
        got = await next_inbound(link.ground, Ack)
        assert got.body.stage == "done"  # type: ignore[union-attr]
