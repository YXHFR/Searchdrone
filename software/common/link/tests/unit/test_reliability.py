"""Resend, expiry, idempotency and status order (pure logic, time passed in)."""

from __future__ import annotations

import itertools

import pytest
from link_helpers import envelope_from, goto, stop_notice

from sd_link.constants import DEDUP_WINDOW, SDLINK_MAX_RESENDS, SDLINK_RESEND_S
from sd_link.reliability import (
    TERMINAL_STAGES,
    DedupCache,
    ResendQueue,
    ack_keys,
    advance_stage,
    inbound_deadline,
)
from sd_schemas.base import new_id
from sd_schemas.link import Ack, AckStage


def _seq() -> itertools.count[int]:
    return itertools.count(100)


def test_constants() -> None:
    assert SDLINK_RESEND_S == 1.0
    assert SDLINK_MAX_RESENDS == 3
    assert DEDUP_WINDOW == 10_000


def test_resend_keeps_msg_id_new_seq_and_reduces_ttl() -> None:
    q, seq = ResendQueue(), _seq()
    env = envelope_from(goto(), side="ground", seq=7, ttl_ms=15_000)
    q.track(env, now=0.0)
    assert q.due(0.99, seq.__next__) == ([], [])
    resend, expired = q.due(1.0, seq.__next__)
    assert not expired and len(resend) == 1
    again = resend[0]
    assert again.msg_id == env.msg_id and again.seq == 100 and again.ttl_ms == 14_000
    assert again.body == env.body
    second, _ = q.due(2.0, seq.__next__)
    assert second[0].seq == 101 and second[0].ttl_ms == 13_000


def test_at_most_three_resends_then_expired() -> None:
    q, seq = ResendQueue(), _seq()
    env = envelope_from(stop_notice(), side="ground")
    q.track(env, now=0.0)
    sends = []
    for t in (1.0, 2.0, 3.0):
        resend, expired = q.due(t, seq.__next__)
        assert not expired
        sends += resend
    assert len(sends) == 3
    resend, expired = q.due(3.99, seq.__next__)
    assert (resend, expired) == ([], [])
    resend, expired = q.due(4.0, seq.__next__)
    assert resend == [] and [e.msg_id for e in expired] == [env.msg_id]
    assert len(q) == 0


def test_ttl_run_out_is_not_sent_again() -> None:
    q, seq = ResendQueue(), _seq()
    env = envelope_from(goto(), side="ground", ttl_ms=1500)
    q.track(env, now=0.0)
    resend, _ = q.due(1.0, seq.__next__)
    assert resend[0].ttl_ms == 500
    resend, expired = q.due(2.0, seq.__next__)
    assert resend == [] and len(expired) == 1


def test_any_ack_stops_the_resend_by_msg_id_or_command_id() -> None:
    q, seq = ResendQueue(), _seq()
    cmd = goto()
    a = envelope_from(cmd, side="ground", ttl_ms=15_000)
    b = envelope_from(stop_notice(), side="ground")
    q.track(a, 0.0)
    q.track(b, 0.0)
    assert ack_keys(a) == (a.msg_id, cmd.command_id)
    assert q.ack(cmd.command_id) == a
    assert q.ack(a.msg_id) is None  # already gone
    assert q.ack(b.msg_id) == b
    assert q.ack(new_id()) is None
    assert q.due(10.0, seq.__next__) == ([], [])


def test_drain_returns_oldest_first_and_empties() -> None:
    q = ResendQueue()
    first = envelope_from(stop_notice(1), side="ground")
    second = envelope_from(stop_notice(2), side="ground")
    q.track(second, 2.0)
    q.track(first, 1.0)
    assert [p.envelope for p in q] == [second, first]
    assert q.drain() == [first, second]
    assert len(q) == 0 and q.drain() == []


def test_only_reliable_is_tracked() -> None:
    from sd_schemas.link import Heartbeat

    with pytest.raises(ValueError, match="reliable"):
        ResendQueue().track(envelope_from(Heartbeat()), 0.0)


def test_dedup_remembers_last_ack_and_evicts() -> None:
    cache = DedupCache(size=3)
    cmd = goto()
    env = envelope_from(cmd, side="ground", ttl_ms=1000)
    cache.remember(env)
    assert env.msg_id in cache and cache.last_ack(env.msg_id) is None
    running = Ack(ref_id=cmd.command_id, stage="running")
    cache.record_ack(running)
    assert cache.last_ack(env.msg_id) == running
    cache.record_ack(Ack(ref_id=new_id(), stage="done"))  # unknown: ignored
    others = [envelope_from(stop_notice(i), side="ground") for i in range(3)]
    for other in others:
        cache.remember(other)
    assert env.msg_id not in cache and len(cache) == 3
    cache.record_ack(Ack(ref_id=cmd.command_id, stage="done"))  # evicted: ignored
    assert cache.last_ack(env.msg_id) is None
    assert all(o.msg_id in cache for o in others)


def test_dedup_default_window_holds_ten_thousand() -> None:
    cache = DedupCache()
    first = envelope_from(stop_notice(0), side="ground")
    cache.remember(first)
    for i in range(DEDUP_WINDOW - 1):
        cache.remember(envelope_from(stop_notice(i + 1), side="ground"))
    assert first.msg_id in cache
    cache.remember(envelope_from(stop_notice(), side="ground"))
    assert first.msg_id not in cache


ORDER: list[AckStage] = ["received", "accepted", "running", "done"]


@pytest.mark.parametrize("terminal", sorted(TERMINAL_STAGES))
def test_status_never_moves_backwards(terminal: AckStage) -> None:
    assert advance_stage(None, "running") == "running"
    assert advance_stage("received", "accepted") == "accepted"
    assert advance_stage("running", "received") == "running"
    assert advance_stage("accepted", "accepted") == "accepted"
    assert advance_stage("running", terminal) == terminal
    for later in ORDER + sorted(TERMINAL_STAGES):
        assert advance_stage(terminal, later) == terminal


def test_inbound_deadline_from_receipt() -> None:
    assert inbound_deadline(50.0, None) is None
    assert inbound_deadline(50.0, 1000) == 51.0
