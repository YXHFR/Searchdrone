"""LinkServer and LinkClient over real WebSockets on 127.0.0.1."""

from __future__ import annotations

import asyncio
import json
import random
from collections.abc import AsyncIterator
from contextlib import asynccontextmanager
from typing import Any

import pytest
from link_helpers import VEHICLE, frame, goto, hello, stop_notice
from websockets.asyncio.client import ClientConnection, connect
from websockets.exceptions import ConnectionClosed, InvalidStatus

from sd_link.client import Backoff, LinkClient
from sd_link.clock import MonotonicClock
from sd_link.close_codes import CloseCode
from sd_link.constants import AUTH_RETRY_S, SDLINK_PATH
from sd_link.server import LinkServer, bearer_token
from sd_link.session import Inbound, LinkSession
from sd_link.tokens import TokenStore, generate_token, hash_token
from sd_link_testing import FakeClock, FakeGround, LossyProxy, next_inbound
from sd_schemas.base import new_id
from sd_schemas.link import SCHEMA_MAJOR, Hello, HelloAck, SkillCommand

pytestmark = pytest.mark.component

TOKEN = generate_token()


class Ground:
    def __init__(self, clock: Any = None, known: tuple[str, ...] = (VEHICLE,)) -> None:
        self.sessions: asyncio.Queue[LinkSession] = asyncio.Queue()
        self.events: list[tuple[str, str]] = []
        self.hellos: list[Hello] = []
        self.server = LinkServer(
            tokens=TokenStore({VEHICLE: hash_token(TOKEN), "sim-09": hash_token("x" * 43)}),
            is_known_vehicle=lambda v: v in known,
            on_hello=self.on_hello,
            on_session=self.on_session,
            clock=clock or MonotonicClock(),
            port=0,
            on_event=lambda name, v: self.events.append((name, v)),
        )

    async def on_hello(self, vehicle_id: str, h: Hello) -> HelloAck:
        self.hellos.append(h)
        return HelloAck(schema_major=SCHEMA_MAJOR, server_epoch=new_id(), operator_latch=None)

    async def on_session(self, session: LinkSession, h: Hello) -> None:
        self.sessions.put_nowait(session)

    @property
    def url(self) -> str:
        return f"ws://127.0.0.1:{self.server.port}{SDLINK_PATH}"


@asynccontextmanager
async def running(ground: Ground) -> AsyncIterator[Ground]:
    async with ground.server:
        yield ground


@asynccontextmanager
async def dial(
    url: str, token: str | None = TOKEN, origin: str | None = None
) -> AsyncIterator[ClientConnection]:
    headers = {"Authorization": f"Bearer {token}"} if token else {}
    async with connect(url, additional_headers=headers, origin=origin) as ws:  # type: ignore[arg-type]
        yield ws


async def close_code(ws: ClientConnection) -> int | None:
    try:
        while True:
            await asyncio.wait_for(ws.recv(), 6)
    except ConnectionClosed as exc:
        return exc.rcvd.code if exc.rcvd else None


async def test_correct_token_gets_hello_ack() -> None:
    async with running(Ground()) as g, dial(g.url) as ws:
        await ws.send(frame(hello()))
        answer = json.loads(await asyncio.wait_for(ws.recv(), 1))
        assert answer["kind"] == "HelloAck" and answer["seq"] == 0
        assert answer["sender"] == "ground:main" and answer["vehicle_id"] == VEHICLE
        session = await asyncio.wait_for(g.sessions.get(), 1)
        assert session.vehicle_id == VEHICLE
        assert ("link.sdlink.up", VEHICLE) in g.events


@pytest.mark.parametrize(
    ("token", "origin", "expected"),
    [
        ("wrong-" + "x" * 40, None, CloseCode.AUTH_FAILED),
        (None, None, CloseCode.AUTH_FAILED),
        (TOKEN, "http://evil.example", CloseCode.FORBIDDEN),
        ("x" * 43, None, CloseCode.FORBIDDEN),  # valid token, unknown vehicle
    ],
)
async def test_refusals_are_close_codes_not_http(
    token: str | None, origin: str | None, expected: int
) -> None:
    async with running(Ground()) as g, dial(g.url, token, origin) as ws:
        assert await close_code(ws) == expected


@pytest.mark.parametrize(
    ("first", "expected"),
    [
        (lambda: frame(hello(schema_major=SCHEMA_MAJOR + 1)), CloseCode.BAD_HELLO),
        (lambda: frame(stop_notice(), side="ground"), CloseCode.BAD_HELLO),
        (lambda: "not json", CloseCode.BAD_HELLO),
        (lambda: frame(hello()).replace("sim-01", "sim-02"), CloseCode.FORBIDDEN),
    ],
)
async def test_bad_hello(first: Any, expected: int) -> None:
    async with running(Ground()) as g, dial(g.url) as ws:
        await ws.send(first())
        assert await close_code(ws) == expected


async def test_no_hello_within_five_seconds() -> None:
    clock = FakeClock()
    async with running(Ground(clock)) as g, dial(g.url) as ws:
        await asyncio.sleep(0.05)
        await clock.advance(4.9)
        assert ws.close_code is None
        await clock.advance(0.2)
        assert await close_code(ws) == CloseCode.BAD_HELLO


async def test_wrong_path_is_404() -> None:
    async with running(Ground()) as g:
        with pytest.raises(InvalidStatus) as err:
            async with connect(g.url.replace("control", "other")):
                pass
        assert err.value.response.status_code == 404


async def _connect_hello(url: str, boot_id: int) -> ClientConnection:
    ws = await connect(url, additional_headers={"Authorization": f"Bearer {TOKEN}"})
    await ws.send(frame(hello(boot_id)))
    return ws


async def test_duplicate_same_boot_with_traffic_closes_the_new_one() -> None:
    clock = FakeClock()
    async with running(Ground(clock)) as g:
        old = await _connect_hello(g.url, boot_id=7)
        assert json.loads(await old.recv())["kind"] == "HelloAck"
        await g.sessions.get()
        new = await _connect_hello(g.url, boot_id=7)
        assert await close_code(new) == CloseCode.REPLACED
        assert old.close_code is None
        await old.close()


async def test_duplicate_with_new_boot_replaces_the_old_one() -> None:
    clock = FakeClock()
    async with running(Ground(clock)) as g:
        old = await _connect_hello(g.url, boot_id=7)
        await old.recv()
        await g.sessions.get()
        new = await _connect_hello(g.url, boot_id=8)
        assert json.loads(await asyncio.wait_for(new.recv(), 1))["kind"] == "HelloAck"
        assert await close_code(old) == CloseCode.REPLACED
        assert ("link.sdlink.replaced", VEHICLE) in g.events
        assert g.server.session_for(VEHICLE) is await g.sessions.get()
        await new.close()


async def test_duplicate_same_boot_after_three_silent_seconds_replaces() -> None:
    clock = FakeClock()
    async with running(Ground(clock)) as g:
        old = await _connect_hello(g.url, boot_id=7)
        await old.recv()
        await g.sessions.get()
        await clock.advance(3.1)  # the old connection sends nothing (half-open TCP)
        new = await _connect_hello(g.url, boot_id=7)
        got = None
        while got != "HelloAck":
            got = json.loads(await asyncio.wait_for(new.recv(), 1))["kind"]
        assert await close_code(old) == CloseCode.REPLACED
        await new.close()


def test_bearer_parsing() -> None:
    assert bearer_token({"Authorization": "Bearer abc"}) == "abc"
    assert bearer_token({"Authorization": "bearer  abc "}) == "abc"
    assert bearer_token({"Authorization": "Basic abc"}) is None
    assert bearer_token({"Authorization": "Bearer "}) is None
    assert bearer_token({}) is None


def test_backoff() -> None:
    b = Backoff(random.Random(1))  # noqa: S311
    delays = [b.next_delay(1001) for _ in range(6)]
    for got, base in zip(delays, [0.5, 1.0, 2.0, 4.0, 5.0, 5.0], strict=True):
        assert base * 0.8 <= got <= base * 1.2
    b.reset()
    assert 0.4 <= b.next_delay(None) <= 0.6
    assert b.next_delay(CloseCode.AUTH_FAILED) == AUTH_RETRY_S
    assert b.next_delay(CloseCode.FORBIDDEN) == AUTH_RETRY_S
    assert b.next_delay(CloseCode.BAD_HELLO) == AUTH_RETRY_S
    assert b.attempt == 0
    assert 0.4 <= b.next_delay(CloseCode.REPLACED) <= 0.6  # normal backoff


async def test_client_handshake_session_and_order_roundtrip() -> None:
    async with FakeGround() as ground:
        got: asyncio.Queue[Inbound] = asyncio.Queue()
        acks: list[HelloAck] = []

        async def on_session(session: LinkSession, ack: HelloAck) -> None:
            acks.append(ack)
            inbound = await next_inbound(session, SkillCommand)
            await session.ack(inbound.body.command_id, "accepted")  # type: ignore[union-attr]
            got.put_nowait(inbound)
            await next_inbound(session)  # keep the session until it closes

        client = LinkClient(
            url=ground.url,
            token=ground.token,
            vehicle_id=VEHICLE,
            make_hello=lambda: hello(boot_id=3),
            on_session=on_session,
            clock=MonotonicClock(),
        )
        assert "token" not in repr(client) and ground.token not in repr(client)
        stop = asyncio.Event()
        runner = asyncio.ensure_future(client.run_forever(stop))
        async with ground.connected() as session:
            cmd = goto()
            await session.send(cmd, ttl_ms=15_000)
            inbound = await asyncio.wait_for(got.get(), 2)
            assert inbound.body == cmd
            assert ground.hellos[0].boot_id == 3
            assert acks[0].server_epoch == ground.server_epoch
            stop.set()
            await session.close()
        await asyncio.wait_for(runner, 3)


async def test_client_reconnects_after_a_drop_with_a_fresh_hello() -> None:
    async with FakeGround() as ground, LossyProxy(ground.url) as proxy:
        boots = iter(range(1, 10))

        async def on_session(session: LinkSession, ack: HelloAck) -> None:
            await next_inbound(session)

        client = LinkClient(
            url=proxy.url,
            token=ground.token,
            vehicle_id=VEHICLE,
            make_hello=lambda: hello(boot_id=next(boots)),
            on_session=on_session,
            clock=MonotonicClock(),
            rng=random.Random(0),  # noqa: S311
        )
        stop = asyncio.Event()
        runner = asyncio.ensure_future(client.run_forever(stop))
        await asyncio.wait_for(ground.sessions.get(), 3)
        await proxy.cut()
        second = await asyncio.wait_for(ground.sessions.get(), 3)
        assert [h.boot_id for h in ground.hellos] == [1, 2]
        stop.set()
        await second.close()
        await asyncio.wait_for(runner, 3)


async def test_client_with_wrong_token_sees_4401() -> None:
    async with FakeGround() as ground:
        client = LinkClient(
            url=ground.url,
            token=generate_token(),
            vehicle_id=VEHICLE,
            make_hello=hello,
            on_session=lambda s, a: asyncio.sleep(0),
            clock=MonotonicClock(),
        )
        assert await asyncio.wait_for(client.connect_once(), 3) == CloseCode.AUTH_FAILED


async def test_client_unreachable_returns_none() -> None:
    client = LinkClient(
        url="ws://127.0.0.1:9/sdlink/v1/control",
        token=TOKEN,
        vehicle_id=VEHICLE,
        make_hello=hello,
        on_session=lambda s, a: asyncio.sleep(0),
        clock=MonotonicClock(),
    )
    assert await asyncio.wait_for(client.connect_once(), 3) is None


async def test_orders_over_a_real_lossy_proxy() -> None:
    async with (
        FakeGround() as ground,
        LossyProxy(ground.url, loss=0.10, delay_s=0.02, seed=3) as proxy,
    ):
        seen: list[int] = []

        async def on_session(session: LinkSession, ack: HelloAck) -> None:
            while True:
                inbound = await next_inbound(session)
                if isinstance(inbound.body, SkillCommand):
                    seen.append(inbound.body.based_on.state_seq)  # type: ignore[union-attr]

        client = LinkClient(
            url=proxy.url,
            token=ground.token,
            vehicle_id=VEHICLE,
            make_hello=hello,
            on_session=on_session,
            clock=MonotonicClock(),
        )
        stop = asyncio.Event()
        runner = asyncio.ensure_future(client.run_forever(stop))
        async with ground.connected() as session:
            for n in range(15):
                await session.send(goto(n), ttl_ms=15_000)
                for _ in range(500):
                    if session.pending_count == 0:
                        break
                    await asyncio.sleep(0.01)
                assert session.pending_count == 0
            assert seen == list(range(15))
            assert proxy.dropped > 0
            stop.set()
            await session.close()
        runner.cancel()
        await asyncio.gather(runner, return_exceptions=True)
