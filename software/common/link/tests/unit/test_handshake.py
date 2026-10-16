"""The client handshake over an in-memory pipe."""

from __future__ import annotations

import asyncio

from link_helpers import VEHICLE, frame, hello, stop_notice

from sd_link.client import handshake
from sd_link.close_codes import CloseCode
from sd_link_testing import FakeClock, transport_pair
from sd_schemas.base import new_id
from sd_schemas.link import SCHEMA_MAJOR, HelloAck


async def _run(answer: str | None, clock: FakeClock) -> tuple[object, int | None]:
    client, server = transport_pair()
    task = asyncio.ensure_future(handshake(client, hello(), vehicle_id=VEHICLE, clock=clock))
    first = await server.recv()
    assert '"kind":"Hello"' in str(first)
    if answer is not None:
        await server.send(answer)
    else:
        await asyncio.sleep(0)
        await clock.advance(5.1)
    result = await task
    return result, client.close_code


def _ack(schema_major: int = SCHEMA_MAJOR) -> HelloAck:
    return HelloAck(schema_major=schema_major, server_epoch=new_id(), operator_latch=None)


async def test_valid_hello_ack() -> None:
    result, code = await _run(frame(_ack(), side="ground"), FakeClock())
    assert isinstance(result, HelloAck) and code is None


async def test_no_hello_ack_in_time_closes() -> None:
    result, code = await _run(None, FakeClock())
    assert result is None and code == CloseCode.GOING_AWAY


async def test_bad_answers_close() -> None:
    for answer in (
        "garbage",
        frame(stop_notice(), side="ground"),
        frame(_ack(SCHEMA_MAJOR + 1), side="ground"),
    ):
        result, code = await _run(answer, FakeClock())
        assert result is None and code == CloseCode.GOING_AWAY
