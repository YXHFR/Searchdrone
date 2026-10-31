"""Image matching engines: one candidate's crops and the profile in, an assessment out.

``CloudMatchEngine`` asks a cloud model (the fast tier: temperature 0, at most 300 answer
tokens, 8 s, no reasoning) through ``CloudModelRuntime``, so every call goes through the
cloud image policy and the ground station's budget first. The answer must follow the
schema from :mod:`sd_search.match.prompt`; a broken answer is asked for once more
(``format_repaired``). Claims nobody can check from a still are flagged: that someone
moves (``motion_claim``) or who they are (``identity_claim``).

A failed call raises :class:`~sd_brain.runtime.base.ModelError` (a refused budget is
:class:`~sd_brain.runtime.budget.BudgetDeniedError`); the cascade decides what that means.
"""

from __future__ import annotations

import json
import re
import time
from collections.abc import Callable
from dataclasses import dataclass, field
from typing import Any, Final, Literal, Protocol

from pydantic import BaseModel, ConfigDict, Field, ValidationError

from sd_brain.runtime.base import ModelError, ModelReply, Sampling
from sd_brain.runtime.budget import BudgetContext
from sd_brain.runtime.images import CloudImagePolicy, ImagePart
from sd_schemas.search import (
    AssessmentEngine,
    AssessmentFlag,
    CandidateAssessment,
    FieldVerdict,
    SearchProfile,
)
from sd_search.match.prompt import answer_format, build_messages

__all__ = [
    "FAST_SAMPLING",
    "CloudMatchEngine",
    "Crop",
    "MatchEngine",
    "MatchRequest",
    "claim_flags",
]

FAST_SAMPLING: Final = Sampling(temperature=0.0, top_p=1.0, max_tokens=300)
FAST_TIMEOUT_S: Final = 8.0
CROPS_FAST: Final = 2
_MOTION = re.compile(r"rör sig|vinkar|rörde|går mot|\bmoving\b|\bwaving\b", re.IGNORECASE)
_IDENTITY = re.compile(r"känner igen|samma person som|ansikte", re.IGNORECASE)


@dataclass(frozen=True)
class Crop:
    detection_id: str
    px_size: float
    jpeg: bytes


@dataclass(frozen=True)
class MatchRequest:
    mission_id: str
    candidate_id: str
    profile: SearchProfile | None  # approved only
    crops: list[Crop] = field(default_factory=list[Crop])  # largest first
    agl_m: float | None = None

    @property
    def profile_version(self) -> int:
        return 0 if self.profile is None else self.profile.profile_version


class MatchEngine(Protocol):
    engine: AssessmentEngine
    model_id: str

    async def assess(self, req: MatchRequest) -> CandidateAssessment: ...


class _Answer(BaseModel):
    """The part of the assessment the model fills."""

    model_config = ConfigDict(extra="ignore")
    person: Literal["yes", "no", "unsure"]
    person_p: float = Field(ge=0, le=1)
    fields: dict[str, FieldVerdict] = Field(default_factory=dict[str, FieldVerdict])
    overall: float = Field(ge=0, le=1)
    explanation_sv: str


class ChatRuntime(Protocol):
    model_id: str

    async def chat(
        self,
        messages: list[dict[str, Any]],
        tools: list[dict[str, Any]],
        timeout_s: float,
        sampling: Sampling,
        *,
        response_format: dict[str, Any] | None = None,
        budget_context: BudgetContext | None = None,
    ) -> ModelReply: ...


def claim_flags(explanation: str) -> list[AssessmentFlag]:
    flags: list[AssessmentFlag] = []
    if _MOTION.search(explanation):
        flags.append("motion_claim")
    if _IDENTITY.search(explanation):
        flags.append("identity_claim")
    return flags


def _parse(content: str | None, keys: list[str]) -> _Answer:
    answer = _Answer.model_validate(json.loads(content or ""))
    fields = {k: v for k, v in answer.fields.items() if k in keys}
    return answer.model_copy(update={"fields": fields})


class CloudMatchEngine:
    """The fast tier (cascade steps 1-2): one call per candidate, 1-2 crops."""

    def __init__(
        self,
        runtime: ChatRuntime,
        *,
        engine: AssessmentEngine = "fast",
        sampling: Sampling = FAST_SAMPLING,
        timeout_s: float = FAST_TIMEOUT_S,
        max_crops: int = CROPS_FAST,
        monotonic: Callable[[], float] = time.monotonic,
    ) -> None:
        self.runtime = runtime
        self.engine: AssessmentEngine = engine
        self.model_id = runtime.model_id
        self.sampling = sampling
        self.timeout_s = timeout_s
        self.max_crops = max_crops
        self.policy = CloudImagePolicy()
        self._mono = monotonic

    async def assess(self, req: MatchRequest) -> CandidateAssessment:
        crops = req.crops[: self.max_crops]
        # A crop is never scaled to fit: the link caps it at 512 px (CROP_MAX_PX), so a
        # larger one is something else (a still, a view) and the policy refuses it.
        images = self.policy.check([ImagePart("crop", "image/jpeg", c.jpeg) for c in crops])
        keys = [] if req.profile is None else req.profile.field_keys
        messages = build_messages(
            req.profile, images, agl_m=req.agl_m, px_size=max((c.px_size for c in crops), default=0)
        )
        fmt = answer_format(keys)
        ctx = BudgetContext("match", req.mission_id)
        started = self._mono()
        reply = await self.runtime.chat(
            messages, [], self.timeout_s, self.sampling, response_format=fmt, budget_context=ctx
        )
        cost = reply.cost_sek or 0.0
        flags: list[AssessmentFlag] = []
        try:
            answer = _parse(reply.content, keys)
        except (ValueError, ValidationError) as exc:
            # One more try with the reason; a second broken answer is a failed call.
            repair = [
                *messages,
                {"role": "assistant", "content": (reply.content or "")[:2000]},
                {
                    "role": "user",
                    "content": "That answer did not follow the schema "
                    f"({type(exc).__name__}). Answer again with only the JSON object.",
                },
            ]
            left = max(1.0, self.timeout_s - (self._mono() - started))
            reply = await self.runtime.chat(
                repair, [], left, self.sampling, response_format=fmt, budget_context=ctx
            )
            cost += reply.cost_sek or 0.0
            try:
                answer = _parse(reply.content, keys)
            except (ValueError, ValidationError):
                err = ModelError("the model's answer did not follow the schema twice")
                err.kind = "parse_error"
                raise err from None
            flags.append("format_repaired")
        explanation = answer.explanation_sv.strip()[:300]
        flags += claim_flags(explanation)
        return CandidateAssessment.model_validate(
            {
                "candidate_id": req.candidate_id,
                "profile_version": req.profile_version,
                "engine": self.engine,
                "model_id": (reply.model_id or self.model_id)[:120],
                "person": answer.person,
                "person_p": answer.person_p,
                "fields": answer.fields,
                "overall": answer.overall,
                "explanation_sv": explanation,
                "flags": flags[:3],
                "crops_used": [c.detection_id for c in crops],
                "latency_ms": round((self._mono() - started) * 1000),
                "cost_sek": round(cost, 6),
            }
        )
