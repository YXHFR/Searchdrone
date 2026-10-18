"""``GroundLink.send_acked``: a message and the onboard executor's Ack that names it."""

from __future__ import annotations

import asyncio
import datetime as dt
import time
import uuid
from typing import Any

from test_ground_link_hello import VID, hello, make

from sd_ground.link import ACK_MEMORY
from sd_link.session import Closed, Inbound
from sd_schemas.base import new_id
from sd_schemas.link import Ack, AckStage, CommanderStatus, Envelope

MESSAGE = CommanderStatus(connected=True, state="ready")


def _id() -> uuid.UUID:
    return uuid.UUID(str(new_id()))


def _sent(link: Any, msg_id: uuid.UUID | None, *, delay_s: float = 0.0) -> None:
    async def send_message(vehicle_id: str, message: Any, **_: Any) -> uuid.UUID | None:
        if delay_s:
            await asyncio.sleep(delay_s)
        return msg_id

    link.send_message = send_message


def _ack(link: Any, msg_id: uuid.UUID, stage: AckStage, vehicle_id: str = VID) -> None:
    link._on_ack(vehicle_id, Ack(ref_id=msg_id, stage=stage))


async def _answer_later(link: Any, msg_id: uuid.UUID, stage: AckStage, **kw: Any) -> None:
    await asyncio.sleep(0.01)
    _ack(link, msg_id, stage, **kw)


async def test_the_executor_s_done_is_acked() -> None:
    link, _, _ = make()
    msg_id = _id()
    _sent(link, msg_id)
    task = asyncio.ensure_future(_answer_later(link, msg_id, "done"))
    assert await link.send_acked(VID, MESSAGE, 1.0) == "acked"
    await task


async def test_an_answer_before_the_wait_is_not_lost() -> None:
    link, _, _ = make()
    msg_id = _id()
    _sent(link, msg_id)
    _ack(link, msg_id, "done")
    assert await link.send_acked(VID, MESSAGE, 1.0) == "acked"


async def test_the_link_s_receipt_is_not_the_executor_s_answer() -> None:
    """``received`` comes from the link before the executor has the message."""
    link, _, _ = make()
    msg_id = _id()
    _sent(link, msg_id)
    task = asyncio.ensure_future(_answer_later(link, msg_id, "received"))
    assert await link.send_acked(VID, MESSAGE, 0.1) == "timeout"
    await task


async def test_a_rejected_or_expired_notice_is_refused() -> None:
    for stage in ("rejected", "expired"):
        link, _, _ = make()
        msg_id = _id()
        _sent(link, msg_id)
        task = asyncio.ensure_future(_answer_later(link, msg_id, stage))
        assert await link.send_acked(VID, MESSAGE, 1.0) == "refused", stage
        await task


async def test_another_vehicle_s_or_message_s_answer_does_not_count() -> None:
    link, _, _ = make()
    msg_id = _id()
    _sent(link, msg_id)
    _ack(link, msg_id, "done", vehicle_id="sim-02")
    _ack(link, _id(), "done")
    assert await link.send_acked(VID, MESSAGE, 0.05) == "timeout"
    assert link._ack_waiters == {}  # pyright: ignore[reportPrivateUsage]


async def test_the_send_and_the_wait_share_one_deadline() -> None:
    """A stalled write cannot hold the caller longer than ``timeout_s``."""
    link, _, _ = make()
    _sent(link, _id(), delay_s=5.0)
    start = time.monotonic()
    assert await link.send_acked(VID, MESSAGE, 0.1) == "timeout"
    assert time.monotonic() - start < 1.0


async def test_link_down_sends_nothing() -> None:
    link, _, _ = make()
    _sent(link, None)
    assert await link.send_acked(VID, MESSAGE, 1.0) == "down"


async def test_the_ack_memory_is_capped_per_vehicle() -> None:
    """Another vehicle's many answers cannot push out an early answer to ours."""
    link, _, _ = make()
    msg_id = _id()
    _ack(link, msg_id, "done")
    for _ in range(ACK_MEMORY + 50):
        _ack(link, _id(), "done", vehicle_id="sim-02")
    assert len(link._acked["sim-02"]) == ACK_MEMORY  # pyright: ignore[reportPrivateUsage]
    _sent(link, msg_id)
    assert await link.send_acked(VID, MESSAGE, 0.05) == "acked"


class _Session:
    """An SD-link session that delivers the given events, then closes."""

    vehicle_id = VID

    def __init__(self, events: list[Any]) -> None:
        self.events = list(events)

    async def next_event(self) -> Any:
        await asyncio.sleep(0)
        return self.events.pop(0) if self.events else Closed(code=None, reason="done")

    async def send(self, *_: Any, **__: Any) -> Any:
        return None


async def test_an_ack_from_the_session_reaches_the_waiting_send() -> None:
    """The executor's Ack goes from the session loop to ``send_acked``, not only in tests
    that call ``_on_ack`` themselves."""
    link, _, _ = make()
    msg_id = _id()
    _sent(link, msg_id)
    waiting = asyncio.ensure_future(link.send_acked(VID, MESSAGE, 1.0))
    await asyncio.sleep(0)
    body = Ack(ref_id=msg_id, stage="done")
    envelope = Envelope(
        msg_id=new_id(),
        kind="Ack",
        sender=f"vehicle:{VID}",
        vehicle_id=VID,
        mission_id=None,
        seq=0,
        sent_utc=dt.datetime.now(dt.UTC),
        ttl_ms=None,
        reliable=False,
        corr_id=None,
        body=body.model_dump(mode="json"),
    )
    inbound = Inbound(envelope=envelope, body=body, received_at=0.0, deadline=None)
    await link._on_session(_Session([inbound]), hello(None, 0))  # type: ignore[arg-type]  # pyright: ignore[reportPrivateUsage]
    assert await waiting == "acked"
