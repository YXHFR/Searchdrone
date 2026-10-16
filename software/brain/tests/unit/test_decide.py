"""The decision interface and the quick stop check (rev 6 B18, F19)."""

from __future__ import annotations

import asyncio
import json
from typing import Any, Literal

import httpx
import pytest
from brain_helpers import mock_client

from sd_brain.decide import (
    DecideError,
    DecisionService,
    JevEngine,
    LocalDecisionEngine,
    OpenAiDecisionsEngine,
    stop_check,
    stop_check_request,
)
from sd_brain.decide.engines import EngineAnswer, validate_answer
from sd_schemas.base import new_id
from sd_schemas.decide import (
    ChoiceAnswer,
    DecideRequest,
    DecisionEngine,
    NoulQuestion,
    ScoreQuestion,
)

KEY = "k" * 40


def _req() -> DecideRequest:
    return DecideRequest(
        request_id=new_id(),
        purpose="test",
        state={"x": 1},
        questions={
            "yes": NoulQuestion(instructions="Is it?"),
            "level": ScoreQuestion(instructions="How much?", max_score=2),
        },
    )


async def test_jev_adapter() -> None:
    seen: list[dict[str, Any]] = []

    def handler(req: httpx.Request) -> httpx.Response:
        assert req.url.path == "/v1/systemone" and req.headers["authorization"] == f"Bearer {KEY}"
        seen.append(json.loads(req.read()))
        return httpx.Response(
            200,
            json={
                "model": "jev-1.13.0",
                "answers": {
                    "yes": {"noul": 0.8},
                    "level": {"score": 1, "probabilities": [0.2, 0.6, 0.2], "confidence": 0.5},
                },
            },
        )

    ans = await JevEngine("https://jev.example", "jev-latest", KEY, mock_client(handler)).answer(
        _req(), 1
    )
    assert ans.engine == "jev" and ans.location == "cloud" and ans.model_id == "jev-1.13.0"
    assert ans.answers["yes"].p_true == pytest.approx(0.8)  # type: ignore[union-attr]
    assert ans.answers["level"].score == 1  # type: ignore[union-attr]
    assert seen[0]["questions"]["yes"] == {"type": "noul", "instructions": "Is it?"}
    assert len(seen[0]["questions"]["level"]["criteria"]) == 3


async def test_jev_choice_and_errors() -> None:
    req = stop_check_request("stanna!", {})

    def ok(r: httpx.Request) -> httpx.Response:
        return httpx.Response(
            200,
            json={
                "answers": {
                    "intent": {
                        "choice": "stop",
                        "probabilities": {"none": 0.1, "stop": 0.85, "return_home": 0.05},
                    }
                }
            },
        )

    ans = await JevEngine("https://j.example", "m", KEY, mock_client(ok)).answer(req, 1)
    assert (
        isinstance(ans.answers["intent"], ChoiceAnswer) and ans.answers["intent"].choice == "stop"
    )
    for body in ({"answers": []}, {"answers": {"intent": {"choice": "stop"}}}, {"nope": 1}):
        eng = JevEngine(
            "https://j.example",
            "m",
            KEY,
            mock_client(lambda r, b=body: httpx.Response(200, json=b)),
        )
        with pytest.raises(DecideError):
            await eng.answer(req, 1)
    with pytest.raises(DecideError):
        await JevEngine("https://j.example", "m", None, None).answer(req, 1)
    assert not JevEngine("https://j.example", "m", None, None).configured


async def test_openai_decisions_adapter() -> None:
    def handler(req: httpx.Request) -> httpx.Response:
        body = json.loads(req.read())
        assert req.url.path == "/v1/decisions" and body["model"] == "gpt-6-luna"
        assert body["questions"][0] == {
            "name": "yes",
            "type": "predicate",
            "instructions": "Is it?",
        }
        assert [lv["label"] for lv in body["questions"][1]["levels"]] == ["0", "1", "2"]
        return httpx.Response(
            200,
            json={
                "answers": [
                    {"name": "yes", "type": "predicate", "probability": 0.3},
                    {
                        "name": "level",
                        "type": "score",
                        "score": 1.4,
                        "probabilities": [
                            {"index": 0, "label": "0", "probability": 0.1},
                            {"index": 1, "label": "1", "probability": 0.4},
                            {"index": 2, "label": "2", "probability": 0.5},
                        ],
                    },
                ],
                "metadata": {"model": "gpt-6-luna-1"},
            },
        )

    eng = OpenAiDecisionsEngine("https://api.example", "gpt-6-luna", KEY, mock_client(handler))
    ans = await eng.answer(_req(), 1)
    assert ans.model_id == "gpt-6-luna-1" and ans.answers["level"].score == 2  # type: ignore[union-attr]

    def refusal(r: httpx.Request) -> httpx.Response:
        return httpx.Response(
            200,
            json={
                "answers": [
                    {"name": "yes", "type": "refusal"},
                    {"name": "level", "type": "refusal"},
                ]
            },
        )

    with pytest.raises(DecideError):
        await OpenAiDecisionsEngine("https://a.example", "m", KEY, mock_client(refusal)).answer(
            _req(), 1
        )

    def choice(r: httpx.Request) -> httpx.Response:
        return httpx.Response(
            200,
            json={
                "answers": [
                    {
                        "name": "intent",
                        "type": "choice",
                        "choice": "return_home",
                        "probabilities": [
                            {"value": "none", "probability": 0.2},
                            {"value": "stop", "probability": 0.1},
                            {"value": "return_home", "probability": 0.7},
                        ],
                    }
                ]
            },
        )

    ans = await OpenAiDecisionsEngine("https://a.example", "m", KEY, mock_client(choice)).answer(
        stop_check_request("åk hem", {}), 1
    )
    assert ans.answers["intent"].choice == "return_home"  # type: ignore[union-attr]


async def test_local_decision_engine() -> None:
    def handler(req: httpx.Request) -> httpx.Response:
        body = json.loads(req.read())
        assert body["response_format"]["type"] == "json_schema"
        content = json.dumps({"yes": 0.9, "level": [0.0, 0.0, 1.0]})
        return httpx.Response(200, json={"choices": [{"message": {"content": content}}]})

    ans = await LocalDecisionEngine(
        "http://127.0.0.1:8091/v1", "m", KEY, mock_client(handler)
    ).answer(_req(), 1)
    assert ans.location == "local" and ans.answers["level"].score == 2  # type: ignore[union-attr]
    bad = lambda r: httpx.Response(200, json={"choices": [{"message": {"content": "nej"}}]})  # noqa: E731
    with pytest.raises(DecideError):
        await LocalDecisionEngine("http://127.0.0.1:8091/v1", "m", KEY, mock_client(bad)).answer(
            _req(), 1
        )
    with pytest.raises(DecideError):
        await LocalDecisionEngine(
            "http://127.0.0.1:8091/v1", "m", KEY, mock_client(lambda r: httpx.Response(500))
        ).answer(_req(), 1)


class _Fake:
    name: DecisionEngine
    location: Literal["cloud", "local"]
    configured: bool

    def __init__(
        self,
        name: DecisionEngine,
        delay: float,
        choice: str | None,
        location: Literal["cloud", "local"] = "cloud",
    ) -> None:
        self.name, self.location, self.delay, self.choice = name, location, delay, choice
        self.configured = True

    async def answer(self, req: DecideRequest, timeout_s: float) -> EngineAnswer:
        await asyncio.sleep(self.delay)
        if self.choice is None:
            raise DecideError("down")
        probs = {"none": 0.05, "stop": 0.05, "return_home": 0.05}
        probs[self.choice] = 0.9
        ans = ChoiceAnswer(choice=self.choice, probabilities=probs)
        return EngineAnswer(
            self.name, self.location, {"intent": ans}, f"{self.name}-m", int(self.delay * 1000)
        )  # type: ignore[arg-type]


async def test_service_best_engine_decides_and_logs_all() -> None:
    logged: list[Any] = []
    svc = DecisionService(
        [_Fake("jev", 0.05, "stop"), _Fake("local", 0.0, "none", "local")],
        on_answer=lambda req, outcomes: logged.append(outcomes),
    )
    res = await svc.decide(stop_check_request("x", {}), 1.0)
    assert res is not None and res.engine == "jev" and res.fallback_from is None
    await svc.drain()
    assert [(o.engine, o.used) for o in logged[0]] == [("jev", True), ("local", False)]


async def test_service_falls_back_when_best_fails_or_is_slow() -> None:
    svc = DecisionService([_Fake("jev", 0.0, None), _Fake("local", 0.0, "stop", "local")])
    res = await svc.decide(stop_check_request("x", {}), 1.0)
    assert res is not None and res.engine == "local" and res.fallback_from == "jev"
    slow = DecisionService(
        [_Fake("jev", 0.5, "none"), _Fake("local", 0.0, "stop", "local")], late_answer_s=0.6
    )
    res = await slow.decide(stop_check_request("x", {}), 0.1)
    assert res is not None and res.engine == "local"
    await slow.drain()
    none = DecisionService([_Fake("jev", 0.0, None)])
    assert await none.decide(stop_check_request("x", {}), 0.1) is None
    assert await DecisionService([]).decide(stop_check_request("x", {}), 0.1) is None


async def test_stop_check() -> None:
    svc = DecisionService([_Fake("jev", 0.0, "stop")])
    res = await stop_check(svc, "vänta, stanna där!", timeout_s=0.5, threshold=0.5)
    assert res.intent == "stop" and res.source == "engine" and res.engine == "jev"
    home = await stop_check(
        DecisionService([_Fake("local", 0.0, "return_home", "local")]),
        "kom tillbaka nu",
        timeout_s=0.5,
        threshold=0.5,
    )
    assert home.intent == "return_home" and home.location == "local"
    calm = await stop_check(
        DecisionService([_Fake("jev", 0.0, "none")]), "hur går det?", timeout_s=0.5, threshold=0.5
    )
    assert calm.intent is None
    high = await stop_check(svc, "stanna", timeout_s=0.5, threshold=0.95)
    assert high.intent is None
    # The fixed words always count, also with no engine at all.
    floor = await stop_check(DecisionService([]), "Stopp!", timeout_s=0.5, threshold=0.5)
    assert floor.intent == "stop" and floor.source == "rule"
    assert (
        await stop_check(DecisionService([]), "flyg hem", timeout_s=0.5, threshold=0.5)
    ).intent == "return_home"
    assert (
        await stop_check(DecisionService([]), "stanna", timeout_s=0.5, threshold=0.5)
    ).intent is None
    assert (
        await stop_check(
            DecisionService([_Fake("jev", 0.0, None)]), "stanna", timeout_s=0.5, threshold=0.5
        )
    ).source == "none"


def test_validate_answer() -> None:
    assert validate_answer({"type": "noul", "p_true": 0.5}).type == "noul"
