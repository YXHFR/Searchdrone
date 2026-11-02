"""Model runtimes: cloud (main model, fallback model), best-available and replay."""

from __future__ import annotations

import json
import logging
import math
from typing import Any

import httpx
import pytest
from brain_helpers import chat_reply, mock_client

from sd_brain.config import BrainConfig, CloudConfig
from sd_brain.runtime.base import (
    MAX_TOKEN_COUNT,
    ModelError,
    ModelTimeoutError,
    NotConfiguredError,
    RawToolCall,
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


@pytest.mark.parametrize(
    "bad",
    [
        {"choices": [{"message": "text"}]},
        {"choices": ["text"]},
        {"choices": [{"message": {"tool_calls": ["x"]}}]},
        {"choices": [{"message": {"tool_calls": [{"function": "x"}]}}]},
        ["not", "an", "object"],
        # an empty string, list or 0 is no object either: never an empty, good answer
        {"choices": [{"message": ""}]},
        {"choices": [{"message": []}]},
        {"choices": [0]},
        "",
    ],
)
async def test_an_answer_of_the_wrong_shape_is_a_parse_error(bad: Any) -> None:
    with pytest.raises(ModelError) as exc:
        parse_reply(bad, latency_ms=1)
    assert exc.value.kind == "parse_error"
    # ... so the fallback model answers, as after any broken answer.
    asked: list[str] = []

    def handler(req: httpx.Request) -> httpx.Response:
        model = json.loads(req.read())["model"]
        asked.append(model)
        return httpx.Response(200, json=bad if len(asked) == 1 else chat_reply("spare"))

    rt = _cloud(handler, model="main/m", fallback_model="spare/m")
    assert (await rt.chat(MSG, [], 5, Sampling())).content == "spare"
    assert asked == ["main/m", "spare/m"]


async def test_cloud_logs_the_tokens_of_each_call(caplog: pytest.LogCaptureFixture) -> None:
    body = chat_reply("hemligt svar")
    body["usage"] = {
        "prompt_tokens": 900,
        "completion_tokens": 512,
        "completion_tokens_details": {"reasoning_tokens": 470},
    }
    rt = CloudModelRuntime(
        CloudConfig(model="vendor/model"),
        KEY,
        mock_client(lambda _r: httpx.Response(200, json=body)),
    )
    with caplog.at_level(logging.INFO, logger="sd_brain.runtime"):
        await rt.chat(MSG, [], 5, Sampling(max_tokens=512))
    [line] = [r.getMessage() for r in caplog.records if r.getMessage().startswith("cloud_call")]
    assert "completion_tokens=512 reasoning_tokens=470 max_tokens=512" in line
    assert "prompt_tokens=900" in line and "finish='stop'" in line
    assert KEY not in line and "hemligt" not in line and "hej" not in line


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
    # no ground station has said its profile: zero data retention
    assert body["provider"] == {"data_collection": "deny", "require_parameters": True, "zdr": True}
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


def test_parse_reply_reads_the_reasoning_tokens() -> None:
    data: dict[str, Any] = {
        "choices": [{"message": {"content": "ok"}}],
        "usage": {
            "prompt_tokens": 900,
            "completion_tokens": 512,
            "completion_tokens_details": {"reasoning_tokens": 470},
        },
    }
    reply = parse_reply(data, latency_ms=1)
    assert (reply.completion_tokens, reply.reasoning_tokens) == (512, 470)
    data["usage"] = {"completion_tokens": 40}  # a model that states no reasoning share
    assert parse_reply(data, latency_ms=1).reasoning_tokens == 0
    data["usage"] = {"completion_tokens": 40, "completion_tokens_details": None}
    assert parse_reply(data, latency_ms=1).reasoning_tokens == 0


def test_left_out_or_null_parts_are_absent() -> None:
    bare = parse_reply({"choices": [{"message": None}], "usage": None}, latency_ms=1)
    assert (bare.content, bare.tool_calls, bare.completion_tokens) == (None, [], 0)
    no_details = {"prompt_tokens": 4, "completion_tokens": 9, "completion_tokens_details": None}
    reply = parse_reply({"choices": [{"message": {}}], "usage": no_details}, latency_ms=1)
    assert (reply.completion_tokens, reply.reasoning_tokens) == (9, 0)


@pytest.mark.parametrize(
    "usage",
    [
        "",
        0,
        ["x"],
        "many",
        {"completion_tokens": "many"},
        {"prompt_tokens": -5, "completion_tokens": "9"},
        {"completion_tokens": True, "cost": "1.0"},
        {"completion_tokens_details": ""},
        {"completion_tokens_details": 0},
        {"completion_tokens_details": "many"},
        {"completion_tokens_details": {"reasoning_tokens": "many"}},
        # too large for a float, or for the budget's books: malformed, never an error
        {"prompt_tokens": 10**400},
        {"prompt_tokens": MAX_TOKEN_COUNT + 1},
        {"completion_tokens": 1e300},
        {"completion_tokens": 1.5},
    ],
)
async def test_malformed_usage_keeps_the_answer(usage: Any) -> None:
    """Usage is metadata: a broken count drops the numbers, never the answer."""
    data = {"choices": [{"message": {"content": "ok"}}], "usage": usage}
    reply = parse_reply(data, latency_ms=1)
    assert reply.content == "ok"
    assert (reply.prompt_tokens, reply.completion_tokens, reply.reasoning_tokens) == (0, 0, 0)
    assert reply.cost_usd is None and not reply.usage_valid
    # ... and no second, paid call goes to the fallback model.
    asked: list[str] = []

    def handler(req: httpx.Request) -> httpx.Response:
        asked.append(json.loads(req.read())["model"])
        return httpx.Response(200, json=data)

    rt = _cloud(handler, model="main/m", fallback_model="spare/m")
    assert (await rt.chat(MSG, [], 5, Sampling())).content == "ok"
    assert asked == ["main/m"]


def test_a_tool_call_comes_through_next_to_broken_usage() -> None:
    call = {"id": "call_7", "function": {"name": "hold", "arguments": '{"s": 30}'}}
    data = {"choices": [{"message": {"tool_calls": [call]}}], "usage": {"prompt_tokens": 10**400}}
    reply = parse_reply(data, latency_ms=1)
    assert reply.tool_calls == [RawToolCall(id="call_7", name="hold", arguments_json='{"s": 30}')]
    assert not reply.usage_valid


@pytest.mark.parametrize(
    "cost", [10**400, 1001, float("inf"), float("nan"), -0.01, True, "0.01", [1]]
)
def test_a_broken_price_is_dropped_and_the_counts_stay(cost: Any) -> None:
    """The ground then prices the tokens itself; never an OverflowError."""
    usage = {"prompt_tokens": 900, "completion_tokens": 30, "cost": cost}
    reply = parse_reply({"choices": [{"message": {"content": "ok"}}], "usage": usage}, latency_ms=1)
    assert reply.content == "ok" and reply.cost_usd is None
    assert (reply.prompt_tokens, reply.completion_tokens, reply.usage_valid) == (900, 30, True)


async def test_a_price_too_large_for_a_float_reaches_the_answer_over_http() -> None:
    body = chat_reply("ok")
    body["usage"] = {"prompt_tokens": 9, "completion_tokens": 3, "cost": 10**400}
    rt = _cloud(lambda _r: httpx.Response(200, json=body), model="main/m")
    reply = await rt.chat(MSG, [], 5, Sampling())
    assert (reply.content, reply.cost_usd, reply.usage_valid) == ("ok", None, True)


async def test_a_body_nested_too_deep_is_a_parse_error() -> None:
    deep = b"[" * 100_000 + b"]" * 100_000
    rt = _cloud(lambda _r: httpx.Response(200, content=deep), model="main/m")
    with pytest.raises(ModelError) as exc:
        await rt.chat(MSG, [], 5, Sampling())
    assert exc.value.kind == "parse_error"


def test_a_good_tool_call_next_to_a_broken_one_is_a_parse_error() -> None:
    good = {"id": "call_1", "function": {"name": "hold", "arguments": "{}"}}
    data = {"choices": [{"message": {"tool_calls": [good, {"function": "x"}]}}]}
    with pytest.raises(ModelError) as exc:
        parse_reply(data, latency_ms=1)
    assert exc.value.kind == "parse_error"


def test_well_formed_usage_is_valid() -> None:
    usage = {"prompt_tokens": MAX_TOKEN_COUNT, "completion_tokens": 512.0, "cost": 0.01}
    reply = parse_reply({"choices": [{"message": {"content": "ok"}}], "usage": usage}, latency_ms=1)
    assert (reply.prompt_tokens, reply.completion_tokens, reply.cost_usd) == (
        MAX_TOKEN_COUNT,
        512,
        0.01,
    )
    assert reply.usage_valid
    no_reasoning = {"prompt_tokens": 9, "completion_tokens": 3, "completion_tokens_details": None}
    reply = parse_reply(
        {"choices": [{"message": {"content": "ok"}}], "usage": no_reasoning}, latency_ms=1
    )
    assert (
        reply.usage_valid and reply.cost_usd is None
    )  # the reasoning share and price may be left out


@pytest.mark.parametrize(
    "data",
    [
        {"choices": [{"message": {"content": "ok"}}]},
        {"choices": [{"message": {"content": "ok"}}], "usage": None},
        {"choices": [{"message": {"content": "ok"}}], "usage": {}},
        {"choices": [{"message": {"content": "ok"}}], "usage": {"completion_tokens": 3}},
        {"choices": [{"message": {"content": "ok"}}], "usage": {"prompt_tokens": 9}},
        {
            "choices": [{"message": {"content": "ok"}}],
            "usage": {"prompt_tokens": 9, "completion_tokens": None},
        },
    ],
)
def test_left_out_counts_keep_the_answer_but_are_no_valid_usage(data: Any) -> None:
    """A provider that states no counts is booked at the estimate, never as free; its
    counts read 0, as malformed usage does."""
    reply = parse_reply(data, latency_ms=1)
    assert reply.content == "ok" and not reply.usage_valid
    assert (reply.prompt_tokens, reply.completion_tokens, reply.cost_usd) == (0, 0, None)


_CALL = {"id": "c1", "function": {"name": "goto", "arguments": "{}"}}


@pytest.mark.parametrize(
    ("message", "usage"),
    [
        ({"content": "ok"}, {"prompt_tokens": 0, "completion_tokens": 0}),
        ({"content": "ok"}, {"prompt_tokens": 9, "completion_tokens": 0, "cost": 0.0}),
        ({"content": None, "tool_calls": [_CALL]}, {"prompt_tokens": 9, "completion_tokens": 0}),
        ({"content": ""}, {"prompt_tokens": 0, "completion_tokens": 3}),
    ],
)
def test_counts_no_real_call_has_are_no_valid_usage(message: Any, usage: Any) -> None:
    """No prompt, or an answer without completion tokens: booked as free, the call would
    be left out of the budget, so it is booked at the estimate."""
    reply = parse_reply({"choices": [{"message": message}], "usage": usage}, latency_ms=1)
    assert not reply.usage_valid
    assert (reply.prompt_tokens, reply.completion_tokens, reply.cost_usd) == (0, 0, None)


def test_an_empty_answer_may_cost_no_completion_tokens() -> None:
    usage = {"prompt_tokens": 9, "completion_tokens": 0}
    reply = parse_reply({"choices": [{"message": {"content": " "}}], "usage": usage}, latency_ms=1)
    assert reply.usage_valid and (reply.prompt_tokens, reply.completion_tokens) == (9, 0)


def test_a_negative_zero_price_is_zero() -> None:
    usage = {"prompt_tokens": 9, "completion_tokens": 3, "cost": -0.0}
    reply = parse_reply({"choices": [{"message": {"content": "ok"}}], "usage": usage}, latency_ms=1)
    assert reply.cost_usd is not None and math.copysign(1.0, reply.cost_usd) == 1.0


def test_arguments_nested_too_deep_are_a_parse_error() -> None:
    deep: Any = {}
    for _ in range(100_000):
        deep = {"a": deep}
    data = {
        "choices": [
            {"message": {"tool_calls": [{"function": {"name": "hold", "arguments": deep}}]}}
        ]
    }
    with pytest.raises(ModelError) as exc:
        parse_reply(data, latency_ms=1)
    assert exc.value.kind == "parse_error"


def test_the_token_limit_is_the_budgets() -> None:
    from sd_schemas.cloud_budget import BudgetCommit

    fields = BudgetCommit.model_json_schema()["properties"]
    assert fields["prompt_tokens"]["maximum"] == MAX_TOKEN_COUNT
    assert fields["completion_tokens"]["maximum"] == MAX_TOKEN_COUNT


def test_the_defaults_are_a_flights() -> None:
    """Sampling() and a bare LlmCommander send what [llm]'s defaults send."""
    from sd_brain.backends.llm import LlmCommander
    from sd_brain.config import BehaviourConfig, LlmConfig
    from sd_brain.context import ContextBuilder
    from sd_brain.runtime.base import llm_sampling
    from sd_schemas.skills import SKILL_REGISTRY

    llm = LlmConfig()
    assert Sampling() == llm_sampling(llm).model_copy(update={"seed": None})
    budget = llm.prompt_budget_tokens
    ctx = ContextBuilder(
        SKILL_REGISTRY, takeoff_policy=BehaviourConfig().takeoff_policy, prompt_budget_tokens=budget
    )
    cmd = LlmCommander(ReplayRuntime([]), ctx, SKILL_REGISTRY, BehaviourConfig(), Sampling())
    assert cmd.context_tokens == llm.context_tokens
