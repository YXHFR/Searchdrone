"""Inbound order on SD-link: charters, clearances and orders one at a time in arrival
order; STOP notices do not wait behind a slow charter install."""

from __future__ import annotations

import asyncio
import datetime as dt
from typing import Any

from onboard_helpers import charter

from sd_link.session import Closed, Expired, Inbound, LinkEvent, LinkState
from sd_obs.clock import MonotonicClock
from sd_onboard.link_client import OnboardLink
from sd_schemas.base import SdModel, new_id
from sd_schemas.link import (
    CharterInstall,
    ClearanceConfirmation,
    Envelope,
    HelloAck,
    ManualTakeover,
    OnboardEvent,
    OperatorOverrideNotice,
    SkillCommand,
)


class _Executor:
    """Records what reaches it; a charter install waits until ``fence_read`` is set."""

    def __init__(self) -> None:
        self.charter: Any = None
        self.calls: list[str] = []
        self.fence_read = asyncio.Event()
        self.fail_next_charter = False

    async def on_charter(self, install: CharterInstall) -> None:
        if self.fail_next_charter:
            self.fail_next_charter = False
            raise RuntimeError("fence read failed")
        self.calls.append(f"charter:{install.charter.version}:start")
        await self.fence_read.wait()
        self.calls.append(f"charter:{install.charter.version}:done")

    def on_clearance(self, c: Any, ref: Any) -> None:
        self.calls.append("clearance")

    def on_override(self, notice: Any) -> None:
        self.calls.append("notice")

    def on_takeover(self, takeover: Any, ref: Any) -> None:
        self.calls.append("take_over" if takeover.active else "hand_back")

    async def on_skill_command(self, cmd: Any, *, deadline_ms: int | None) -> None:
        self.calls.append(f"order:{cmd.skill}")


def _envelope(body: SdModel, mission_id: Any = None) -> Envelope:
    return Envelope.model_construct(
        msg_id=new_id(),
        kind=type(body).__name__,
        sender="ground:main",
        vehicle_id="sim-01",
        mission_id=mission_id,
        seq=0,
        sent_utc=dt.datetime(2026, 10, 15, tzinfo=dt.UTC),
        ttl_ms=None,
        reliable=True,
        corr_id=None,
        body=body.model_dump(mode="json"),
    )


def _inbound(body: SdModel) -> Inbound:
    return Inbound(envelope=_envelope(body), body=body, received_at=0.0, deadline=None)


def _link() -> tuple[OnboardLink, _Executor]:
    link = OnboardLink(
        url="ws://127.0.0.1:1/sdlink/v1/control",
        token="t" * 43,
        vehicle_id="sim-01",
        clock=MonotonicClock(),
    )
    ex = _Executor()
    link.executor = ex  # type: ignore[assignment]
    return link, ex


async def _settle() -> None:
    for _ in range(20):
        await asyncio.sleep(0)


async def test_notice_is_not_held_up_and_the_rest_keeps_its_order() -> None:
    link, ex = _link()
    worker = asyncio.ensure_future(link._handle_in_order())
    try:
        await link._dispatch(_inbound(CharterInstall(charter=charter(version=1))))
        await _settle()  # the install has started reading the fence
        await link._dispatch(_inbound(ClearanceConfirmation.model_construct()))
        await link._dispatch(_inbound(SkillCommand.model_construct(skill="takeoff")))
        await link._dispatch(_inbound(CharterInstall(charter=charter(version=2))))
        await link._dispatch(_inbound(OperatorOverrideNotice.model_construct()))
        await _settle()
        # The fence is still being read: the notice is handled, nothing behind the charter is.
        assert ex.calls == ["charter:1:start", "notice"]
        ex.fence_read.set()
        await _settle()
        assert ex.calls == [
            "charter:1:start",
            "notice",
            "charter:1:done",
            "clearance",
            "order:takeoff",
            "charter:2:start",
            "charter:2:done",
        ]
    finally:
        worker.cancel()
        await asyncio.gather(worker, return_exceptions=True)


async def test_a_failing_install_is_reported_and_the_queue_goes_on() -> None:
    link, ex = _link()
    ex.fail_next_charter = True
    ex.fence_read.set()
    worker = asyncio.ensure_future(link._handle_in_order())
    try:
        await link._dispatch(_inbound(CharterInstall(charter=charter())))
        await link._dispatch(_inbound(SkillCommand.model_construct(skill="hold")))
        await _settle()
        assert ex.calls == ["order:hold"]
        failed = [e for e, _ in link._events if e.type == "executor.inbound_failed"]
        assert failed and failed[0].data["kind"] == "CharterInstall"
        assert failed[0].severity == "critical"
    finally:
        worker.cancel()
        await asyncio.gather(worker, return_exceptions=True)


async def test_take_over_is_at_once_and_a_hand_back_waits_its_turn() -> None:
    """A hand-back lets go of HELD, so it must not overtake an older queued order."""
    link, ex = _link()
    worker = asyncio.ensure_future(link._handle_in_order())
    try:
        await link._dispatch(_inbound(CharterInstall(charter=charter())))
        await _settle()
        await link._dispatch(_inbound(SkillCommand.model_construct(skill="goto")))
        await link._dispatch(_inbound(ManualTakeover.model_construct(active=True)))
        await link._dispatch(_inbound(ManualTakeover.model_construct(active=False)))
        await _settle()
        assert ex.calls == ["charter:1:start", "take_over"]
        ex.fence_read.set()
        await _settle()
        assert ex.calls == [
            "charter:1:start",
            "take_over",
            "charter:1:done",
            "order:goto",
            "hand_back",
        ]
    finally:
        worker.cancel()
        await asyncio.gather(worker, return_exceptions=True)


class _Session:
    """Sends into nowhere; the test hands it the link events."""

    state = LinkState.UP
    last_inbound = 0.0

    def __init__(self) -> None:
        self.sent: list[SdModel] = []
        self.missions: list[Any] = []
        self.events: asyncio.Queue[LinkEvent] = asyncio.Queue()

    async def send(self, body: SdModel, *, mission_id: Any = None, **_: Any) -> None:
        self.sent.append(body)
        self.missions.append(mission_id)

    async def next_event(self) -> LinkEvent:
        return await self.events.get()


class _LinkExecutor(_Executor):
    def on_connected(self, ack: Any) -> None:
        pass

    def on_link(self, up: bool, last_inbound_ms: int | None) -> None:
        pass

    def snapshot(self) -> SdModel:
        return OnboardEvent(type="executor.snapshot", severity="debug")


async def test_events_lost_with_the_link_are_sent_again_first() -> None:
    """An event sent into a link that was already gone (no Ack) is not lost: it goes
    again on the next session, in its order and ahead of newer events."""
    link, _ = _link()
    link.executor = _LinkExecutor()  # type: ignore[assignment]
    link.event("executor.find_hold_started", "warning", {"n": 1})
    link.event("executor.link_lost", "warning", {"n": 2})
    session = _Session()
    task = asyncio.ensure_future(
        link._on_session(session, HelloAck.model_construct())  # type: ignore[arg-type]
    )
    await _settle()
    first, second = (b for b in session.sent if isinstance(b, OnboardEvent) and b.data)
    assert not link._events
    session.events.put_nowait(Expired(_envelope(first), "LINK_NO_ACK"))
    session.events.put_nowait(Closed(None, "no ack", (_envelope(second),)))
    await asyncio.wait_for(task, 5)
    link.event("executor.find_hold_ended", "warning", {"n": 3})
    assert [e.data["n"] for e, _ in link._events] == [1, 2, 3]


async def test_events_carry_the_mission_they_happened_in() -> None:
    """An event goes out with the mission whose charter was in force when it happened,
    also when it is sent again after the link was lost (the mission may have closed)."""
    link, _ = _link()
    ex = _LinkExecutor()
    link.executor = ex  # type: ignore[assignment]
    link.event("executor.boot", "info", {"n": 0})
    ex.charter = charter(version=1)
    mid = ex.charter.mission_id
    link.event("executor.find_hold_started", "warning", {"n": 1})
    ex.charter = None
    session = _Session()
    task = asyncio.ensure_future(
        link._on_session(session, HelloAck.model_construct())  # type: ignore[arg-type]
    )
    await _settle()
    sent = [
        (b, m)
        for b, m in zip(session.sent, session.missions, strict=True)
        if isinstance(b, OnboardEvent) and b.data
    ]
    assert [(b.data["n"], m) for b, m in sent] == [(0, None), (1, mid)]
    session.events.put_nowait(Closed(None, "no ack", (_envelope(sent[1][0], mid),)))
    await asyncio.wait_for(task, 5)
    assert [(e.data["n"], m) for e, m in link._events] == [(1, mid)]
