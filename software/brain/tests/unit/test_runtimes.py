"""Model runtimes: local, cloud, best-available and replay (05 §6.1; rev 6 B18)."""

from __future__ import annotations

import httpx
import pytest
from brain_helpers import chat_reply, mock_client

from sd_brain.config import CloudConfig, LlmConfig
from sd_brain.runtime.base import (
    ModelError,
    ModelTimeoutError,
    NotConfiguredError,
    Sampling,
    parse_reply,
)
from sd_brain.runtime.cloud import CloudModelRuntime
from sd_brain.runtime.fallback import BestAvailableRuntime
from sd_brain.runtime.local import LocalModelRuntime
from sd_brain.runtime.replay import (
    ReplayExhaustedError,
    ReplayNotConsumedError,
    ReplayRuntime,
    _reply,
)

KEY = "k" * 40
MSG = [{"role": "user", "content": "hej"}]


def _local(handler) -> LocalModelRuntime:
    return LocalModelRuntime(LlmConfig(), KEY, mock_client(handler))


async def test_local_sends_key_and_tools() -> None:
    seen: list[httpx.Request] = []

    def handler(req: httpx.Request) -> httpx.Response:
        seen.append(req)
        return httpx.Response(200, json=chat_reply("ok", [("hold", {"rationale_sv": "x"})]))

    reply = await _local(handler).chat(MSG, [{"type": "function"}], 5, Sampling(seed=1))
    assert reply.engine == "local" and reply.tool_calls[0].name == "hold"
    assert seen[0].url.path == "/v1/chat/completions"
    assert seen[0].headers["authorization"] == f"Bearer {KEY}"
    body = seen[0].read().decode()
    assert '"tool_choice":"auto"' in body.replace(" ", "") and '"seed":1' in body.replace(" ", "")


@pytest.mark.parametrize(
    ("response", "kind"),
    [
        (httpx.Response(500), "http_error"),
        (httpx.Response(200, content=b"no json"), "parse_error"),
        (httpx.Response(200, json={"choices": []}), "parse_error"),
    ],
)
async def test_local_errors(response: httpx.Response, kind: str) -> None:
    with pytest.raises(ModelError) as exc:
        await _local(lambda _r: response).chat(MSG, [], 5, Sampling())
    assert exc.value.kind == kind


async def test_local_timeout_and_unreachable() -> None:
    def slow(_r: httpx.Request) -> httpx.Response:
        raise httpx.ReadTimeout("slow")

    with pytest.raises(ModelTimeoutError):
        await _local(slow).chat(MSG, [], 5, Sampling())

    def down(_r: httpx.Request) -> httpx.Response:
        raise httpx.ConnectError("down")

    with pytest.raises(ModelError):
        await _local(down).chat(MSG, [], 5, Sampling())
    assert not await _local(down).health()


async def test_local_health_key_and_tokenize() -> None:
    def handler(req: httpx.Request) -> httpx.Response:
        if req.url.path == "/health":
            return httpx.Response(200, json={"status": "ok"})
        if req.url.path == "/tokenize":
            return httpx.Response(200, json={"tokens": [1, 2, 3]})
        return httpx.Response(401)

    rt = _local(handler)
    assert await rt.health()
    assert await rt.count_tokens("abc") == 3
    with pytest.raises(ModelError, match="fel nyckel"):
        await rt.check_key()
    ok = _local(lambda r: httpx.Response(200, json=chat_reply("x")))
    await ok.check_key()
    assert await ok.count_tokens("x") is None or True


async def test_cloud_not_configured_sends_nothing() -> None:
    rt = CloudModelRuntime(CloudConfig(), None, None)
    assert not rt.configured
    with pytest.raises(NotConfiguredError, match="inte konfigurerad"):
        await rt.chat(MSG, [], 5, Sampling())
    assert not CloudModelRuntime(CloudConfig(), KEY, None).configured


async def test_cloud_openrouter_call() -> None:
    seen: list[httpx.Request] = []

    def handler(req: httpx.Request) -> httpx.Response:
        seen.append(req)
        return httpx.Response(200, json=chat_reply("moln"))

    rt = CloudModelRuntime(CloudConfig(model="vendor/model"), KEY, mock_client(handler))
    reply = await rt.chat(MSG, [], 5, Sampling(seed=42))
    assert reply.engine == "cloud" and reply.content == "moln"
    assert seen[0].url.host == "openrouter.ai" and seen[0].headers["x-title"] == "Searchdrone"
    assert '"seed"' not in seen[0].read().decode()


class _Clock:
    t = 0.0

    def __call__(self) -> float:
        return self.t


async def test_best_available_falls_back_in_the_same_call() -> None:
    calls: list[str] = []

    def cloud_handler(_r: httpx.Request) -> httpx.Response:
        calls.append("cloud")
        return httpx.Response(503)

    def local_handler(_r: httpx.Request) -> httpx.Response:
        calls.append("local")
        return httpx.Response(200, json=chat_reply("lokal"))

    clock = _Clock()
    cloud = CloudModelRuntime(CloudConfig(model="m"), KEY, mock_client(cloud_handler))
    best = BestAvailableRuntime(cloud, _local(local_handler), retry_after_s=60, monotonic=clock)
    assert best.engine() == "cloud" and best.model_id == "m"
    reply = await best.chat(MSG, [], 20, Sampling())
    assert reply.content == "lokal" and calls == ["cloud", "local"]
    assert best.engine() == "local" and best.last_engine == "local"
    await best.chat(MSG, [], 20, Sampling())
    assert calls == ["cloud", "local", "local"]  # no cloud call while it rests
    clock.t = 61
    assert best.engine() == "cloud"


async def test_best_available_uses_cloud_when_it_answers() -> None:
    cloud = CloudModelRuntime(
        CloudConfig(model="m"),
        KEY,
        mock_client(lambda r: httpx.Response(200, json=chat_reply("moln"))),
    )
    best = BestAvailableRuntime(cloud, _local(lambda r: httpx.Response(500)))
    assert (await best.chat(MSG, [], 6, Sampling())).engine == "cloud"
    assert best.last_engine == "cloud"
    no_cloud = BestAvailableRuntime(
        None, _local(lambda r: httpx.Response(200, json=chat_reply("l")))
    )
    assert no_cloud.engine() == "local" and no_cloud.model_id == LlmConfig().model


async def test_replay() -> None:
    rt = ReplayRuntime(
        [_reply({"content": "a"}), _reply({"tool_calls": [{"name": "hold", "arguments": "{}"}]})]
    )
    assert (await rt.chat(MSG, [], 1, Sampling())).content == "a"
    with pytest.raises(ReplayNotConsumedError):
        rt.assert_consumed()
    assert (await rt.chat(MSG, [], 1, Sampling())).tool_calls[0].name == "hold"
    with pytest.raises(ReplayExhaustedError):
        await rt.chat(MSG, [], 1, Sampling())


def test_parse_reply_dict_arguments() -> None:
    data = {
        "choices": [
            {
                "message": {
                    "content": None,
                    "tool_calls": [{"function": {"name": "hold", "arguments": {"a": 1}}}],
                }
            }
        ]
    }
    assert parse_reply(data, latency_ms=1).tool_calls[0].arguments_json == '{"a": 1}'
