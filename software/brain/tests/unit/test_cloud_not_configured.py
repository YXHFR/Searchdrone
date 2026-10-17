"""B1-14: without a key and a model the cloud runtime says "not configured" and sends nothing.

Every socket is blocked here, loopback included (pytest-socket), and DNS look-ups are
recorded: a cloud runtime that is not configured must not even try to reach the network.
The block starts inside the running event loop (the loop itself needs a socket pair).
"""

from __future__ import annotations

import socket
from collections.abc import AsyncIterator

import httpx
import pytest
import pytest_socket
from brain_helpers import chat_reply, mock_client

from sd_brain.config import CloudConfig, LlmConfig
from sd_brain.runtime.base import NotConfiguredError, Sampling
from sd_brain.runtime.cloud import CloudModelRuntime
from sd_brain.runtime.fallback import BestAvailableRuntime
from sd_brain.runtime.local import LocalModelRuntime

KEY = "k" * 40
MSG = [{"role": "user", "content": "hej"}]


@pytest.fixture
async def no_network(monkeypatch: pytest.MonkeyPatch) -> AsyncIterator[list[object]]:
    lookups: list[object] = []

    def record(*args: object, **_: object) -> list[object]:
        lookups.append(args[0] if args else None)
        raise OSError("no DNS in this test")

    monkeypatch.setattr(socket, "getaddrinfo", record)
    pytest_socket.disable_socket(allow_unix_socket=False)
    try:
        yield lookups
    finally:
        pytest_socket.enable_socket()


async def test_sockets_really_are_blocked(no_network: list[object]) -> None:
    with pytest.raises(pytest_socket.SocketBlockedError):
        socket.socket(socket.AF_INET, socket.SOCK_STREAM)


@pytest.mark.parametrize(
    ("model", "key"),
    [("", None), ("", KEY), ("vendor/model", None)],
    ids=["nothing", "key-without-model", "model-without-key"],
)
async def test_cloud_not_configured_sends_nothing(
    no_network: list[object], model: str, key: str | None
) -> None:
    async with httpx.AsyncClient(timeout=1.0, trust_env=False) as client:  # a real client
        rt = CloudModelRuntime(CloudConfig(model=model), key, client)
        assert rt.configured is False
        with pytest.raises(NotConfiguredError, match="inte konfigurerad"):
            await rt.chat(MSG, [], 5, Sampling())
    assert no_network == []


async def test_best_available_without_cloud_answers_locally(no_network: list[object]) -> None:
    async with httpx.AsyncClient(timeout=1.0, trust_env=False) as cloud_client:
        cloud = CloudModelRuntime(CloudConfig(), None, cloud_client)
        local = LocalModelRuntime(
            LlmConfig(),
            KEY,
            mock_client(lambda _r: httpx.Response(200, json=chat_reply("lokal"))),
        )
        best = BestAvailableRuntime(cloud, local)
        assert best.engine() == "local"
        reply = await best.chat(MSG, [], 5, Sampling())
    assert reply.content == "lokal" and reply.engine == "local"
    assert no_network == []
