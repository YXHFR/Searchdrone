"""``OracleMatcher``: a stand-in for the cloud model in the simulator, from a VLM profile.

The simulator knows the truth (is this candidate a person, is it the missing person), so
the oracle draws an answer the way a measured (or, for now, modelled) model would:

* ``person``: given a real person, ``yes`` with ``tp_yes``, ``unsure`` with ``tp_unsure``,
  else ``no``; given anything else, ``yes`` with ``fp_yes``, ``unsure`` with ``fp_unsure``.
* each field of the profile: given the missing person, ``match`` with
  ``match_given_true``, ``conflict`` with ``conflict_given_true``, else ``not_visible``;
  given anyone else, ``match`` with ``match_given_false``, ``not_visible`` as often as for
  the true person, else ``conflict``.
* the latency: log-normal through the profile's p50 and p95 (milliseconds).

It never makes a network call and refuses to start outside the simulator (``SD_SIM=1``):
an oracle's answers must never pass for a model's on a real search. Its assessments say
``engine = "oracle"`` and the profile's id. It has the matching engine's shape
(``engine``, ``model_id``, ``assess``) without importing the search service.
"""

from __future__ import annotations

import asyncio
import json
import math
import os
import random
from collections.abc import Callable
from dataclasses import dataclass
from importlib.resources import files
from pathlib import Path
from typing import Any, Literal, Protocol

from pydantic import BaseModel, ConfigDict, Field, model_validator

from sd_schemas.search import CandidateAssessment, FieldVerdict, SearchProfile

__all__ = ["OracleMatcher", "Truth", "VlmProfile", "load_profile"]

DEFAULT_PROFILE = "vlm_oracle_v0"
Z95 = 1.6448536


class _Strict(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)


class PersonRates(_Strict):
    tp_yes: float = Field(ge=0, le=1)
    tp_unsure: float = Field(ge=0, le=1)
    fp_yes: float = Field(ge=0, le=1)
    fp_unsure: float = Field(ge=0, le=1)

    @model_validator(mode="after")
    def _sums(self) -> PersonRates:
        if self.tp_yes + self.tp_unsure > 1 or self.fp_yes + self.fp_unsure > 1:
            raise ValueError("yes + unsure may not be above 1")
        return self


class FieldRates(_Strict):
    match_given_true: float = Field(ge=0, le=1)
    conflict_given_true: float = Field(ge=0, le=1)
    match_given_false: float = Field(ge=0, le=1)

    @property
    def not_visible(self) -> float:
        return 1.0 - self.match_given_true - self.conflict_given_true

    @model_validator(mode="after")
    def _sums(self) -> FieldRates:
        if self.not_visible < 0 or self.match_given_false + self.not_visible > 1:
            raise ValueError("the field rates do not add up")
        return self


class Latency(_Strict):
    p50: float = Field(gt=0)
    p95: float = Field(gt=0)

    @model_validator(mode="after")
    def _order(self) -> Latency:
        if self.p95 < self.p50:
            raise ValueError("p95 below p50")
        return self


class VlmProfile(_Strict):
    profile_id: str = Field(pattern=r"^[a-z0-9_]{1,64}$")
    label: Literal["modelled", "measured"]
    note: str = ""
    person: PersonRates
    field: FieldRates
    latency_ms: Latency


def load_profile(name_or_path: str | Path = DEFAULT_PROFILE) -> VlmProfile:
    """A profile by name (``sd_sim/vlm_profiles/<name>.json``) or from a file."""
    path = Path(name_or_path)
    if path.suffix == ".json" and path.is_file():
        text = path.read_text(encoding="utf-8")
    else:
        text = files("sd_sim").joinpath(f"vlm_profiles/{name_or_path}.json").read_text("utf-8")
    return VlmProfile.model_validate(json.loads(text))


@dataclass(frozen=True)
class Truth:
    person: bool  # a real person (anyone)
    target: bool  # the missing person


class _Request(Protocol):
    @property
    def candidate_id(self) -> str: ...
    @property
    def profile(self) -> SearchProfile | None: ...
    @property
    def profile_version(self) -> int: ...
    @property
    def crops(self) -> Any: ...


class OracleMatcher:
    engine: Literal["oracle"] = "oracle"

    def __init__(
        self,
        truth: Callable[[Any], Truth],
        profile: VlmProfile | None = None,
        *,
        seed: int | None = None,
        sleep: bool = False,
        environ: dict[str, str] | None = None,
    ) -> None:
        env = os.environ if environ is None else environ
        if env.get("SD_SIM") != "1":
            raise RuntimeError("OracleMatcher används bara i simulatorn (SD_SIM=1)")
        self.profile = profile or load_profile()
        self.model_id = f"oracle/{self.profile.profile_id}"
        self.truth = truth
        self.rng = random.Random(seed)  # noqa: S311 - a simulation, not a secret
        self.sleep = sleep  # wait the drawn latency (scenarios); tests do not

    def draw_person(self, truth: Truth) -> Literal["yes", "no", "unsure"]:
        rates = self.profile.person
        yes, unsure = (
            (rates.tp_yes, rates.tp_unsure) if truth.person else (rates.fp_yes, rates.fp_unsure)
        )
        u = self.rng.random()
        return "yes" if u < yes else "unsure" if u < yes + unsure else "no"

    def draw_field(self, truth: Truth) -> FieldVerdict:
        rates = self.profile.field
        u = self.rng.random()
        if truth.target:
            if u < rates.match_given_true:
                return "match"
            return (
                "conflict"
                if u < rates.match_given_true + rates.conflict_given_true
                else ("not_visible")
            )
        if u < rates.match_given_false:
            return "match"
        return "not_visible" if u < rates.match_given_false + rates.not_visible else "conflict"

    def draw_latency_ms(self) -> float:
        lat = self.profile.latency_ms
        sigma = (math.log(lat.p95) - math.log(lat.p50)) / Z95
        return self.rng.lognormvariate(math.log(lat.p50), sigma)

    async def assess(self, req: _Request) -> CandidateAssessment:
        truth = self.truth(req)
        person = self.draw_person(truth)
        keys = [] if req.profile is None else req.profile.field_keys
        fields = {k: self.draw_field(truth) for k in keys}
        latency = self.draw_latency_ms()
        if self.sleep:
            await asyncio.sleep(latency / 1000)
        person_p = {"yes": 0.9, "unsure": 0.5, "no": 0.1}[person]
        seen = [v for v in fields.values() if v != "not_visible"]
        overall = sum(v == "match" for v in seen) / len(seen) if seen else person_p
        return CandidateAssessment.model_validate(
            {
                "candidate_id": req.candidate_id,
                "profile_version": req.profile_version,
                "engine": self.engine,
                "model_id": self.model_id,
                "person": person,
                "person_p": person_p,
                "fields": fields,
                "overall": round(overall, 4),
                "explanation_sv": f"Simulerad bedömning ({self.profile.profile_id}).",
                "crops_used": [c.detection_id for c in list(req.crops)[:3]],
                "latency_ms": round(latency),
            }
        )
