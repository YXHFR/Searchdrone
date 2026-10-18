"""Model runtimes: cloud (main model, fallback model), best-available and replay."""

from __future__ import annotations

import json

import httpx
import pytest
from brain_helpers import chat_reply, mock_client

from sd_brain.config import BrainConfig, CloudConfig
from sd_brain.runtime.base import (
    ModelError,
    ModelTimeoutError,
    NotConfiguredError,
    Sampling,
    parse_reply,
)
from sd_brain.runtime.cloud import CloudModelRuntime
from sd_brain.runtime.fallback import BestAvailableRuntime, CloudUnavailableError
from sd_brain.runtime.replay import (
    ReplayExhaustedError,
    ReplayNotConsumedError,
    ReplayRuntime,
    _reply,
)

KEY = "k" * 40
MSG = [{"role": "user", "content": "hej"}]


def _cloud(handler, **cfg) -> CloudModelRuntime:  # type: ignore[no-untyped-def]
    return CloudModelRuntime(CloudConfig(**cfg), KEY, mock_client(handler))


@pytest.mark.parametrize(
    ("response", "kind"),
    [
        (httpx.Response(500), "http_error"),
        (httpx.Response(200, content=b"no json"), "parse_error"),
        (httpx.Response(200, json={"choices": []}), "parse_error"),
    ],
)
async def test_cloud_errors(response: httpx.Response, kind: str) -> None:
    with pytest.raises(ModelError) as exc:
        await _cloud(lambda _r: response, fallback_model="").chat(MSG, [], 5, Sampling())
    assert exc.value.kind == kind


async def test_cloud_timeout_and_unreachable() -> None:
    def slow(_r: httpx.Request) -> httpx.Response:
        raise httpx.ReadTimeout("slow")

    with pytest.raises(ModelTimeoutError):
        await _cloud(slow).chat(MSG, [], 5, Sampling())

    def down(_r: httpx.Request) -> httpx.Response:
        raise httpx.ConnectError("down")

    with pytest.raises(ModelError) as exc:
        await _cloud(down, fallback_model="").chat(MSG, [], 5, Sampling())
    assert KEY not in str(exc.value)


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


async def test_cloud_sends_reasoning_provider_and_tools() -> None:
    seen: list[httpx.Request] = []

    def handler(req: httpx.Request) -> httpx.Response:
        seen.append(req)
        return httpx.Response(200, json=chat_reply("ok", [("hold", {"rationale_sv": "x"})]))

    reply = await _cloud(handler).chat(MSG, [{"type": "function"}], 5, Sampling())
    body = json.loads(seen[0].read())
    assert reply.tool_calls[0].name == "hold"
    assert body["model"] == "z-ai/glm-5.3-flash"
    assert body["reasoning"] == {"effort": "low", "exclude": True}
    assert body["provider"] == {"data_collection": "deny", "require_parameters": True}
    assert body["tool_choice"] == "auto" and body["tools"] == [{"type": "function"}]
    assert seen[0].headers["authorization"] == f"Bearer {KEY}"


@pytest.mark.parametrize("profile", ["field", "bench"])
async def test_real_flight_profile_sends_zdr_to_the_cloud(profile: str) -> None:
    seen: list[httpx.Request] = []

    def handler(req: httpx.Request) -> httpx.Response:
        seen.append(req)
        return httpx.Response(200, json=chat_reply("ok", [("hold", {"rationale_sv": "x"})]))

    cfg = BrainConfig.model_validate({"profile": profile}).cloud
    await CloudModelRuntime(cfg, KEY, mock_client(handler)).chat(MSG, [], 5, Sampling())
    assert json.loads(seen[0].read())["provider"] == {
        "data_collection": "deny",
        "require_parameters": True,
        "zdr": True,
    }


async def test_cloud_fallback_model_takes_over_on_error() -> None:
    models: list[str] = []

    def handler(req: httpx.Request) -> httpx.Response:
        model = json.loads(req.read())["model"]
        models.append(model)
        if model == "z-ai/glm-5.3-flash":
            return httpx.Response(503)
        return httpx.Response(200, json={**chat_reply("fallback"), "model": model})

    reply = await _cloud(handler).chat(MSG, [], 5, Sampling())
    assert models == ["z-ai/glm-5.3-flash", "anthropic/claude-haiku-5.5"]
    assert reply.content == "fallback" and reply.model_id == "anthropic/claude-haiku-5.5"


async def test_cloud_no_fallback_after_a_timeout() -> None:
    models: list[str] = []

    def handler(req: httpx.Request) -> httpx.Response:
        models.append(json.loads(req.read())["model"])
        raise httpx.ReadTimeout("slow")

    with pytest.raises(ModelTimeoutError):
        await _cloud(handler).chat(MSG, [], 5, Sampling())
    assert models == ["z-ai/glm-5.3-flash"]  # no time left: the command mode answers


class _Clock:
    t = 0.0

    def __call__(self) -> float:
        return self.t


async def test_best_available_goes_to_command_mode_and_rests() -> None:
    calls: list[str] = []

    def cloud_handler(_r: httpx.Request) -> httpx.Response:
        calls.append("cloud")
        return httpx.Response(503)

    clock = _Clock()
    cloud = CloudModelRuntime(
        CloudConfig(model="m", fallback_model=""), KEY, mock_client(cloud_handler)
    )
    best = BestAvailableRuntime(cloud, retry_after_s=60, monotonic=clock)
    assert best.engine() == "cloud" and best.model_id == "m"
    with pytest.raises(ModelError):
        await best.chat(MSG, [], 20, Sampling())
    assert calls == ["cloud"]
    assert best.engine() == "command" and best.last_engine == "command"
    with pytest.raises(CloudUnavailableError):
        await best.chat(MSG, [], 20, Sampling())
    assert calls == ["cloud"]  # no cloud call while it rests
    clock.t = 61
    assert best.engine() == "cloud"


async def test_best_available_uses_cloud_when_it_answers() -> None:
    cloud = CloudModelRuntime(
        CloudConfig(model="m"),
        KEY,
        mock_client(lambda r: httpx.Response(200, json=chat_reply("moln"))),
    )
    best = BestAvailableRuntime(cloud)
    assert (await best.chat(MSG, [], 6, Sampling())).engine == "cloud"
    assert best.last_engine == "cloud"
    no_cloud = BestAvailableRuntime(None)
    assert no_cloud.engine() == "command" and no_cloud.model_id == ""


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
