"""Decision engines: a state plus typed questions in, probabilities out.

* :class:`JevEngine` - Jev (TypeSafe AI), a hosted decision model.
* :class:`OpenAiDecisionsEngine` - the OpenAI Decisions API.

The two hosted adapters follow the vendors' published request shapes as far as they
are documented; the base URL and model are settings, and CI tests them against fake
servers only.
"""

from __future__ import annotations

import json
import time
from dataclasses import dataclass
from typing import Any, Literal, Protocol

import httpx
from pydantic import ValidationError

from sd_schemas.decide import (
    ChoiceAnswer,
    ChoiceQuestion,
    DecideAnswer,
    DecideRequest,
    DecisionEngine,
    NoulAnswer,
    NoulQuestion,
    ScoreAnswer,
    ScoreQuestion,
)

__all__ = [
    "DecideError",
    "EngineAnswer",
    "JevEngine",
    "OpenAiDecisionsEngine",
    "QuestionEngine",
]


class DecideError(RuntimeError):
    """An engine did not give a usable answer."""


@dataclass(frozen=True)
class EngineAnswer:
    engine: DecisionEngine
    location: Literal["cloud", "local"]
    answers: dict[str, DecideAnswer]
    model_id: str | None
    latency_ms: int


class QuestionEngine(Protocol):
    name: DecisionEngine
    location: Literal["cloud", "local"]

    @property
    def configured(self) -> bool: ...

    async def answer(self, req: DecideRequest, timeout_s: float) -> EngineAnswer: ...


def _normalise(probs: dict[str, float]) -> dict[str, float]:
    total = sum(max(0.0, p) for p in probs.values())
    if total <= 0:
        raise DecideError("all probabilities are zero")
    return {k: min(1.0, max(0.0, p) / total) for k, p in probs.items()}


def _choice(probs: dict[str, float], q: ChoiceQuestion) -> ChoiceAnswer:
    known = {k: float(probs.get(k, 0.0)) for k in q.criteria}
    norm = _normalise(known)
    best = max(norm, key=lambda k: norm[k])
    return ChoiceAnswer(choice=best, probabilities=norm)


def _score(probs: list[float], q: ScoreQuestion) -> ScoreAnswer:
    if len(probs) != q.max_score + 1:
        raise DecideError("score distribution has the wrong length")
    norm = _normalise({str(i): float(p) for i, p in enumerate(probs)})
    values = [norm[str(i)] for i in range(len(probs))]
    return ScoreAnswer(score=max(range(len(values)), key=lambda i: values[i]), probabilities=values)


async def _post(
    client: httpx.AsyncClient, url: str, key: str, body: dict[str, Any], timeout_s: float
) -> dict[str, Any]:
    try:
        resp = await client.post(
            url,
            json=body,
            headers={"Authorization": f"Bearer {key}"},
            timeout=max(0.05, timeout_s),
        )
    except httpx.HTTPError as exc:
        raise DecideError(f"engine unreachable: {type(exc).__name__}") from None
    if resp.status_code != 200:
        raise DecideError(f"engine answered HTTP {resp.status_code}")
    try:
        data = resp.json()
    except ValueError:
        raise DecideError("engine answered with non-JSON") from None
    if not isinstance(data, dict):
        raise DecideError("engine answer is not an object")
    return data


# --- Jev (TypeSafe AI) -------------------------------------------------------------------


def _jev_questions(req: DecideRequest) -> dict[str, Any]:
    out: dict[str, Any] = {}
    for name, q in req.questions.items():
        if isinstance(q, NoulQuestion):
            out[name] = {"type": "noul", "instructions": q.instructions}
        elif isinstance(q, ChoiceQuestion):
            out[name] = {"type": "choice", "instructions": q.instructions, "criteria": q.criteria}
        elif isinstance(q, ScoreQuestion):
            levels = [f"level {i} of {q.max_score}" for i in range(q.max_score + 1)]
            out[name] = {"type": "score", "instructions": q.instructions, "criteria": levels}
        else:
            raise DecideError("box questions are not supported by this engine yet")
    return out


def _jev_answer(q: Any, raw: Any) -> DecideAnswer:
    if isinstance(q, NoulQuestion):
        p = (
            raw.get("noul", raw.get("p_true", raw.get("probability")))
            if isinstance(raw, dict)
            else raw
        )
        if not isinstance(p, int | float | str):
            raise DecideError("yes/no answer without a probability")
        return NoulAnswer(p_true=min(1.0, max(0.0, float(p))))
    if not isinstance(raw, dict):
        raise DecideError("answer is not an object")
    probs = raw.get("probabilities")
    if isinstance(q, ChoiceQuestion):
        if not isinstance(probs, dict):
            raise DecideError("choice answer without probabilities")
        return _choice({str(k): float(v) for k, v in probs.items()}, q)
    if isinstance(probs, dict):
        probs = [probs.get(str(i), 0.0) for i in range(q.max_score + 1)]
    if not isinstance(probs, list):
        raise DecideError("score answer without probabilities")
    return _score([float(v) for v in probs], q)


class JevEngine:
    """Jev from TypeSafe AI: ``POST {base_url}/v1/systemone`` with ``state`` and ``questions``."""

    name: DecisionEngine = "jev"
    location: Literal["cloud", "local"] = "cloud"

    def __init__(
        self, base_url: str, model: str, api_key: str | None, client: httpx.AsyncClient | None
    ) -> None:
        self.base_url = base_url.rstrip("/")
        self.model = model
        self._key = api_key
        self._client = client

    @property
    def configured(self) -> bool:
        return bool(self._key and self._client is not None)

    async def answer(self, req: DecideRequest, timeout_s: float) -> EngineAnswer:
        if self._key is None or self._client is None:
            raise DecideError("Jev is not configured")
        start = time.monotonic()
        body = {"model": self.model, "state": req.state, "questions": _jev_questions(req)}
        data = await _post(
            self._client, self.base_url + "/v1/systemone", self._key, body, timeout_s
        )
        raw_answers = data.get("answers")
        if not isinstance(raw_answers, dict):
            raise DecideError("Jev answer without answers")
        try:
            answers = {name: _jev_answer(q, raw_answers[name]) for name, q in req.questions.items()}
        except (KeyError, TypeError, ValueError, ValidationError) as exc:
            raise DecideError(f"Jev answer does not fit: {type(exc).__name__}") from None
        latency = int((time.monotonic() - start) * 1000)
        return EngineAnswer("jev", "cloud", answers, data.get("model") or self.model, latency)


# --- OpenAI Decisions API ----------------------------------------------------------------


def _openai_questions(req: DecideRequest) -> list[dict[str, Any]]:
    out: list[dict[str, Any]] = []
    for name, q in req.questions.items():
        if isinstance(q, NoulQuestion):
            out.append({"name": name, "type": "predicate", "instructions": q.instructions})
        elif isinstance(q, ChoiceQuestion):
            out.append(
                {
                    "name": name,
                    "type": "choice",
                    "instructions": q.instructions,
                    "choices": [{"value": k, "description": v} for k, v in q.criteria.items()],
                }
            )
        elif isinstance(q, ScoreQuestion):
            out.append(
                {
                    "name": name,
                    "type": "score",
                    "instructions": q.instructions,
                    "levels": [{"label": str(i)} for i in range(q.max_score + 1)],
                }
            )
        else:
            raise DecideError("box questions are not supported by this engine yet")
    return out


def _openai_answer(q: Any, raw: dict[str, Any]) -> DecideAnswer:
    kind = raw.get("type")
    if kind == "refusal":
        raise DecideError("the engine refused the question")
    if isinstance(q, NoulQuestion):
        return NoulAnswer(p_true=min(1.0, max(0.0, float(raw["probability"]))))
    entries = raw.get("probabilities") or []
    if isinstance(q, ChoiceQuestion):
        probs = {str(e["value"]): float(e["probability"]) for e in entries}
        return _choice(probs, q)
    by_index = {int(e.get("index", i)): float(e["probability"]) for i, e in enumerate(entries)}
    return _score([by_index.get(i, 0.0) for i in range(q.max_score + 1)], q)


class OpenAiDecisionsEngine:
    """OpenAI Decisions API: ``POST {base_url}/v1/decisions`` with ``input`` and ``questions``."""

    name: DecisionEngine = "openai_decisions"
    location: Literal["cloud", "local"] = "cloud"

    def __init__(
        self, base_url: str, model: str, api_key: str | None, client: httpx.AsyncClient | None
    ) -> None:
        self.base_url = base_url.rstrip("/")
        self.model = model
        self._key = api_key
        self._client = client

    @property
    def configured(self) -> bool:
        return bool(self._key and self._client is not None)

    async def answer(self, req: DecideRequest, timeout_s: float) -> EngineAnswer:
        if self._key is None or self._client is None:
            raise DecideError("OpenAI Decisions is not configured")
        start = time.monotonic()
        body = {
            "model": self.model,
            "input": json.dumps(req.state, ensure_ascii=False),
            "questions": _openai_questions(req),
        }
        data = await _post(
            self._client, self.base_url + "/v1/decisions", self._key, body, timeout_s
        )
        raw_answers = data.get("answers")
        if not isinstance(raw_answers, list):
            raise DecideError("answer without answers")
        by_name = {str(a.get("name")): a for a in raw_answers if isinstance(a, dict)}
        try:
            answers = {name: _openai_answer(q, by_name[name]) for name, q in req.questions.items()}
        except (KeyError, TypeError, ValueError, ValidationError) as exc:
            raise DecideError(f"answer does not fit: {type(exc).__name__}") from None
        model = (data.get("metadata") or {}).get("model") or self.model
        latency = int((time.monotonic() - start) * 1000)
        return EngineAnswer("openai_decisions", "cloud", answers, model, latency)
