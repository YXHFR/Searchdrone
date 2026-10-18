"""``GroundLink.send_acked``: a message and the onboard Ack that names it."""

from __future__ import annotations

import asyncio
import uuid
from typing import Any

from test_ground_link_hello import VID, make

from sd_schemas.base import new_id
from sd_schemas.link import Ack, CommanderStatus


def _id() -> uuid.UUID:
    return uuid.UUID(str(new_id()))


def _sent(link: Any, msg_id: uuid.UUID | None) -> None:
    async def send_message(vehicle_id: str, message: Any, **_: Any) -> uuid.UUID | None:
        return msg_id

    link.send_message = send_message


MESSAGE = CommanderStatus(connected=True, state="ready")


async def test_the_answer_comes_back() -> None:
    link, _, _ = make()
    msg_id = _id()
    _sent(link, msg_id)

    async def answer() -> None:
        await asyncio.sleep(0.01)
        link._on_ack(Ack(ref_id=msg_id, stage="done"))  # pyright: ignore[reportPrivateUsage]

    task = asyncio.ensure_future(answer())
    assert await link.send_acked(VID, MESSAGE, 1.0) == "acked"
    await task


async def test_an_answer_before_the_wait_is_not_lost() -> None:
    link, _, _ = make()
    msg_id = _id()
    _sent(link, msg_id)
    link._on_ack(Ack(ref_id=msg_id, stage="done"))  # pyright: ignore[reportPrivateUsage]
    assert await link.send_acked(VID, MESSAGE, 1.0) == "acked"


async def test_no_answer_in_time_and_another_message_s_answer() -> None:
    link, _, _ = make()
    _sent(link, _id())
    link._on_ack(Ack(ref_id=_id(), stage="done"))  # pyright: ignore[reportPrivateUsage]
    assert await link.send_acked(VID, MESSAGE, 0.05) == "timeout"
    assert link._ack_waiters == {}  # pyright: ignore[reportPrivateUsage]


async def test_link_down_sends_nothing() -> None:
    link, _, _ = make()
    _sent(link, None)
    assert await link.send_acked(VID, MESSAGE, 1.0) == "down"
