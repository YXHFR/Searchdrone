"""The cloud model as the AI commander, the command mode as its only reserve.

Every model call here goes to a fake OpenAI-compatible server on 127.0.0.1
(``fake_openai.FakeOpenAI``), never the internet. It checks the fields sent (reasoning,
provider), that the fallback model takes over when the main model fails, that a model
that does not answer leaves the turn to the command mode within ``LLM_CALL_TIMEOUT_S``,
and that STOP and FLY HOME work with no cloud at all.
"""

from __future__ import annotations

import time
from typing import Any

import httpx
from brain_helpers import REG, make_loop
from fake_openai import FakeOpenAI

from sd_brain.backends import LlmCommander
from sd_brain.config import BehaviourConfig, CloudConfig
from sd_brain.constants import LLM_CALL_TIMEOUT_S
from sd_brain.context import ContextBuilder
from sd_brain.decide.service import DecisionService
from sd_brain.loop import COMMAND_MODE_TEXT
from sd_brain.runtime.base import Sampling
from sd_brain.runtime.cloud import CloudModelRuntime
from sd_brain.runtime.fallback import BestAvailableRuntime

KEY = "sk-test-" + "k" * 40
MAIN = "z-ai/glm-5.3-flash"
FALLBACK = "anthropic/claude-haiku-5.5"


def _commander(
    fake: FakeOpenAI, client: httpx.AsyncClient, **cfg: Any
) -> tuple[LlmCommander, BestAvailableRuntime]:
    cloud = CloudModelRuntime(CloudConfig(base_url=fake.base_url, **cfg), KEY, client)
    best = BestAvailableRuntime(cloud, retry_after_s=60)
    ctx = ContextBuilder(REG, takeoff_policy="request_then_clearance", prompt_budget_tokens=6000)
    return LlmCommander(best, ctx, REG, BehaviourConfig(), Sampling()), best


async def test_turn_sends_reasoning_and_provider_to_the_cloud() -> None:
    with FakeOpenAI() as fake:
        async with httpx.AsyncClient(trust_env=False) as client:
            cmd, best = _commander(fake, client)
            loop, fg, _ = make_loop(cmd, engine=best.engine)
            loop.handle_event(fg.operator_message("Hur mycket batteri har vi?"))
            await loop.settle()
    assert [m.text for m in fg.posted] == ["Batteriet är 80 procent."]
    body = fake.requests[0]
    assert body["model"] == MAIN
    assert body["reasoning"] == {"effort": "low", "exclude": True}
    # no ground station has said its profile: zero data retention
    assert body["provider"] == {"data_collection": "deny", "require_parameters": True, "zdr": True}
    assert body["tools"] and body["tool_choice"] == "auto"
    assert "seed" not in body
    assert fake.auth[0] == f"Bearer {KEY}"
    assert loop.heartbeat()["engine"] == "cloud"


async def test_fallback_model_answers_when_the_main_model_fails() -> None:
    with FakeOpenAI() as fake:
        fake.behaviour[MAIN] = "500"
        async with httpx.AsyncClient(trust_env=False) as client:
            cmd, best = _commander(fake, client)
            loop, fg, _ = make_loop(cmd, engine=best.engine)
            loop.handle_event(fg.operator_message("Hur mycket batteri har vi?"))
            await loop.settle()
    assert fake.models_asked() == [MAIN, FALLBACK]
    assert [m.text for m in fg.posted] == ["Batteriet är 80 procent."]
    assert best.engine() == "cloud"  # the cloud answered, through the fallback model


async def test_timeout_goes_to_command_mode_within_the_call_limit() -> None:
    with FakeOpenAI() as fake:
        fake.behaviour[MAIN] = "slow"
        fake.slow_s = LLM_CALL_TIMEOUT_S + 5
        async with httpx.AsyncClient(trust_env=False) as client:
            cmd, best = _commander(fake, client, call_timeout_s=1.0)
            loop, fg, _ = make_loop(cmd, engine=best.engine)
            start = time.monotonic()
            loop.handle_event(fg.operator_message("håll position"))
            await loop.settle()
            took = time.monotonic() - start
    assert took < LLM_CALL_TIMEOUT_S
    assert took < 3.0  # the configured limit (1 s) plus the command mode
    assert fake.models_asked() == [MAIN]  # no time left for the fallback model
    assert [c.skill for c in fg.commands] == ["hold"]  # the command mode carried it out
    assert fg.commands[0].backend == "grammar"
    assert best.engine() == "command"
    beat = loop.heartbeat()
    assert (beat["status"], beat["backend"], beat["engine"]) == ("degraded", "grammar", "command")


async def test_command_mode_answers_what_it_does_not_understand() -> None:
    with FakeOpenAI() as fake:
        fake.behaviour[MAIN] = "500"
        fake.behaviour[FALLBACK] = "500"
        async with httpx.AsyncClient(trust_env=False) as client:
            cmd, best = _commander(fake, client)
            loop, fg, _ = make_loop(cmd, engine=best.engine)
            loop.handle_event(fg.operator_message("vad tycker du om vädret?"))
            await loop.settle()
            assert fake.models_asked() == [MAIN, FALLBACK]
            # The cloud rests: the next line goes straight to the command mode.
            loop.handle_event(fg.operator_message("hjälp"))
            await loop.settle()
    assert fake.models_asked() == [MAIN, FALLBACK]
    assert fg.posted[0].text == "Förstod inte. Skriv 'hjälp'."
    assert fg.posted[1].text.startswith("Kommandon:")
    assert not fg.commands


async def test_without_key_there_is_no_ai_chat() -> None:
    """No key (or no network at start): the command mode answers, nothing is sent."""
    with FakeOpenAI() as fake:
        async with httpx.AsyncClient(trust_env=False) as client:
            cloud = CloudModelRuntime(CloudConfig(base_url=fake.base_url), None, client)
            best = BestAvailableRuntime(cloud)
            ctx = ContextBuilder(
                REG, takeoff_policy="request_then_clearance", prompt_budget_tokens=6000
            )
            cmd = LlmCommander(best, ctx, REG, BehaviourConfig(), Sampling())
            loop, fg, _ = make_loop(cmd, engine=best.engine)
            loop.handle_event(fg.operator_message("status"))
            await loop.settle()
    assert fake.requests == []
    assert fg.posted and fg.posted[0].kind == "status"
    assert loop.heartbeat()["engine"] == "command"


async def test_stop_and_fly_home_work_without_any_cloud() -> None:
    """The stop words act at once with no model and no decision engine."""
    for text, intent in (("Stopp!", "stop"), ("flyg hem", "return_home")):
        with FakeOpenAI() as fake:
            fake.behaviour[MAIN] = "slow"
            async with httpx.AsyncClient(trust_env=False) as client:
                cloud = CloudModelRuntime(CloudConfig(base_url=fake.base_url), None, client)
                best = BestAvailableRuntime(cloud)
                ctx = ContextBuilder(
                    REG, takeoff_policy="request_then_clearance", prompt_budget_tokens=6000
                )
                cmd = LlmCommander(best, ctx, REG, BehaviourConfig(), Sampling())
                loop, fg, _ = make_loop(cmd, decide=DecisionService([]), engine=best.engine)
                start = time.monotonic()
                loop.handle_event(fg.operator_message(text))
                await loop.settle()
                took = time.monotonic() - start
        assert [i.intent for i in fg.intents] == [intent]
        assert fg.intents[0].engine == "rule" and fg.intents[0].probability == 1.0
        assert took < 1.0
        assert fake.requests == []
        # The command mode leaves STOP / FLY HOME to the ground station: no order of its own.
        assert not fg.commands


def test_command_mode_text_points_to_the_buttons() -> None:
    assert "knapparna" in COMMAND_MODE_TEXT and "hjälp" in COMMAND_MODE_TEXT
