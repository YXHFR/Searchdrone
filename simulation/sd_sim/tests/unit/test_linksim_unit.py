"""linksim: profiles, UDP down, and the control API's three guards."""

import asyncio
import json
import socket

import pytest

from sd_sim.linksim import control, core
from sd_sim.linksim.control import ControlApi, Request
from sd_sim.udprelay import FROM_VEHICLE

PROFILES = core.load_profiles()
N = 100_000  # 1 % loss within +-10 % needs this many packets to be stable


def free_port(kind: int = socket.SOCK_DGRAM) -> int:
    with socket.socket(socket.AF_INET, kind) as s:
        s.bind(("127.0.0.1", 0))
        return s.getsockname()[1]


def test_profiles() -> None:
    assert set(PROFILES) == {"perfect", "field_typical", "dead"}
    assert PROFILES["field_typical"].delay_ms == 40


def test_field_typical_loss_and_delay() -> None:
    state = core.LinkState(PROFILES["field_typical"], seed=1)
    decide = state.udp_policy("mavlink")
    results = [decide(FROM_VEHICLE) for _ in range(N)]
    lost = sum(r is None for r in results)
    assert 0.9 * 0.01 * N <= lost <= 1.1 * 0.01 * N
    delays = [r for r in results if r is not None]
    mean = sum(delays) / len(delays)
    assert 0.9 * 0.040 <= mean <= 1.1 * 0.040
    assert min(delays) >= 0.025 - 1e-9 and max(delays) <= 0.055 + 1e-9


def test_perfect_and_dead() -> None:
    perfect = core.LinkState(PROFILES["perfect"]).udp_policy("video")
    assert all(perfect(FROM_VEHICLE) == 0.0 for _ in range(1000))
    dead = core.LinkState(PROFILES["dead"]).udp_policy("video")
    assert all(dead(FROM_VEHICLE) is None for _ in range(1000))


def test_flow_down_drops_everything() -> None:
    state = core.LinkState(PROFILES["perfect"])
    state.up["mavlink"] = False
    decide = state.udp_policy("mavlink")
    assert all(decide(d) is None for d in ("from_vehicle", "to_vehicle") for _ in range(100))


async def test_udp_relay_forwards_and_radio_down_blocks() -> None:
    p = {
        "radio_mavlink_in": free_port(),
        "ground_mavlink": free_port(),
        "video_raw_udp": free_port(),
        "video_out_udp": free_port(),
        "linksim_sdlink": free_port(socket.SOCK_STREAM),
        "ground_sdlink": free_port(socket.SOCK_STREAM),
    }
    sim = core.LinkSim(p, PROFILES["perfect"])
    await sim.start()
    ground = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    ground.bind(("127.0.0.1", p["ground_mavlink"]))
    ground.setblocking(False)
    vehicle = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    vehicle.bind(("127.0.0.1", 0))
    vehicle.setblocking(False)
    loop = asyncio.get_running_loop()
    try:
        vehicle.sendto(b"\xfe\x00\x01\x01\x01\x00\x00\x00", ("127.0.0.1", p["radio_mavlink_in"]))
        data, addr = await asyncio.wait_for(loop.sock_recvfrom(ground, 100), 2)
        assert data.startswith(b"\xfe")
        ground.sendto(b"reply", addr)
        back = await asyncio.wait_for(loop.sock_recv(vehicle, 100), 2)
        assert back == b"reply"
        sim.set_radio(False)
        for _ in range(20):
            vehicle.sendto(b"x", ("127.0.0.1", p["radio_mavlink_in"]))
        with pytest.raises(TimeoutError):
            await asyncio.wait_for(loop.sock_recv(ground, 100), 0.3)
        assert sim.snapshot()["flows"]["mavlink"]["from_vehicle"]["dropped"] == 20  # type: ignore[index]
    finally:
        ground.close()
        vehicle.close()
        sim.close()


def make_api() -> tuple[ControlApi, core.LinkSim]:
    p: dict[str, int] = dict.fromkeys(
        (
            "radio_mavlink_in",
            "ground_mavlink",
            "video_raw_udp",
            "video_out_udp",
            "linksim_sdlink",
            "ground_sdlink",
        ),
        0,
    )
    sim = core.LinkSim(p, PROFILES["perfect"])
    return ControlApi(sim, PROFILES, "secret-token", 8790), sim


def req(
    method: str,
    path: str,
    *,
    token: str = "secret-token",  # noqa: S107 - test value
    host: str = "127.0.0.1:8790",
    ctype: str = "application/json",
    body: bytes = b"",
) -> Request:
    headers = {"host": host, "content-type": ctype}
    if token:
        headers["authorization"] = f"Bearer {token}"
    return Request(method, path, headers, body)


def test_api_needs_token() -> None:
    api, sim = make_api()
    status, _ = api.handle(req("POST", "/radio", token="", body=b'{"state": "down"}'))
    assert status == 401
    status, _ = api.handle(req("POST", "/radio", token="wrong", body=b'{"state": "down"}'))
    assert status == 401
    assert sim.state.up["mavlink"]


def test_api_checks_host() -> None:
    api, sim = make_api()
    status, _ = api.handle(
        req("POST", "/radio", host="evil.example:8790", body=b'{"state": "down"}')
    )
    assert status == 421
    assert sim.state.up["link"]
    assert api.handle(req("GET", "/health", host="localhost:8790"))[0] == 200


def test_api_needs_json() -> None:
    api, sim = make_api()
    status, _ = api.handle(req("POST", "/radio", ctype="text/plain", body=b'{"state": "down"}'))
    assert status == 415
    assert sim.state.up["video"]


def test_api_routes() -> None:
    api, sim = make_api()
    assert api.handle(req("POST", "/flows/video", body=b'{"state": "down"}'))[0] == 200
    assert not sim.state.up["video"] and sim.state.up["mavlink"]
    assert api.handle(req("POST", "/radio", body=b'{"state": "down"}'))[0] == 200
    assert not any(sim.state.up.values())
    assert api.handle(req("PUT", "/profile", body=b'{"name": "dead", "seed": 2}'))[0] == 200
    assert sim.state.profile.name == "dead"
    assert api.handle(req("PUT", "/profile", body=b'{"name": "nope"}'))[0] == 400
    assert api.handle(req("POST", "/flows/other", body=b'{"state": "up"}'))[0] == 404
    assert api.handle(req("POST", "/radio", body=b'{"state": "sideways"}'))[0] == 400
    assert api.handle(req("GET", "/state"))[1]["profile"] == "dead"


def test_api_refuses_empty_token() -> None:
    with pytest.raises(ValueError):
        ControlApi(make_api()[1], PROFILES, "", 8790)


async def test_api_over_http() -> None:
    port = free_port(socket.SOCK_STREAM)
    api = ControlApi(make_api()[1], PROFILES, "secret-token", port)
    await api.start()
    try:

        async def call(raw: bytes) -> tuple[int, dict[str, object]]:
            reader, writer = await asyncio.open_connection("127.0.0.1", port)
            writer.write(raw)
            await writer.drain()
            data = await reader.read()
            writer.close()
            head, _, body = data.partition(b"\r\n\r\n")
            return int(head.split()[1]), json.loads(body)

        auth = "Authorization: Bearer secret-token\r\n"
        ok = await call(f"GET /health HTTP/1.1\r\nHost: 127.0.0.1:{port}\r\n{auth}\r\n".encode())
        assert ok == (200, {"ok": True})
        noauth = await call(f"GET /health HTTP/1.1\r\nHost: 127.0.0.1:{port}\r\n\r\n".encode())
        assert noauth[0] == 401
        assert (await call(b"garbage\r\n\r\n"))[0] == 400
    finally:
        api.close()


async def test_request_too_large() -> None:
    reader = asyncio.StreamReader()
    reader.feed_data(b"POST /radio HTTP/1.1\r\nContent-Length: 999999\r\n\r\n")
    reader.feed_eof()
    with pytest.raises(control.BadRequestError):
        await control.read_request(reader)
