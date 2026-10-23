"""The simulator's stand-in for the cloud model follows its VLM profile."""

from __future__ import annotations

import asyncio
from collections import Counter
from dataclasses import dataclass, field
from typing import Any

import pytest

from sd_schemas.search import SearchProfile
from sd_sim.vlm_oracle import OracleMatcher, Truth, VlmProfile, load_profile

SIM = {"SD_SIM": "1"}
DRAWS = 5000
TOL = 0.03  # +-3 percentage points
MISSION = "0192f4a1-7c2e-7d10-9a3b-5e1f0c2d8a21"


@dataclass(frozen=True)
class Req:
    candidate_id: str
    profile: SearchProfile | None = None
    crops: list[Any] = field(default_factory=list[Any])

    @property
    def profile_version(self) -> int:
        return 0 if self.profile is None else self.profile.profile_version


def _profile() -> SearchProfile:
    return SearchProfile.model_validate(
        {
            "profile_id": "0192f4a1-7c2e-7d10-9a3b-5e1f0c2d8a22",
            "profile_version": 1,
            "mission_id": MISSION,
            "subject_category": "hiker",
            "garments": [{"part": "upper", "colors": ["red"], "certainty": "certain"}],
            # The public CMAC test field (ArduPilot's SITL home), never a private place.
            "last_seen": {
                "lat": -35.363261,
                "lon": 149.16523,
                "at_utc": "2026-10-01T11:00:00+00:00",
            },
            "approved_by": "operator:demo",
            "approved_at": "2026-10-01T12:00:00+00:00",
        }
    )


def _share(counts: Counter[str], key: str) -> float:
    return counts[key] / sum(counts.values())


@pytest.mark.parametrize("truth", [Truth(True, True), Truth(True, False), Truth(False, False)])
def test_oracle_follows_the_profile(truth: Truth) -> None:
    prof = load_profile()
    oracle = OracleMatcher(lambda _: truth, prof, seed=7, environ=SIM)
    person = Counter(oracle.draw_person(truth) for _ in range(DRAWS))
    fields = Counter(oracle.draw_field(truth) for _ in range(DRAWS))
    p, f = prof.person, prof.field
    if truth.person:
        assert abs(_share(person, "yes") - p.tp_yes) <= TOL
        assert abs(_share(person, "unsure") - p.tp_unsure) <= TOL
    else:
        assert abs(_share(person, "yes") - p.fp_yes) <= TOL
        assert abs(_share(person, "unsure") - p.fp_unsure) <= TOL
    if truth.target:
        assert abs(_share(fields, "match") - f.match_given_true) <= TOL
        assert abs(_share(fields, "conflict") - f.conflict_given_true) <= TOL
    else:
        assert abs(_share(fields, "match") - f.match_given_false) <= TOL
    assert abs(_share(fields, "not_visible") - f.not_visible) <= TOL


def test_latency_follows_p50_and_p95() -> None:
    prof = load_profile()
    oracle = OracleMatcher(lambda _: Truth(True, True), prof, seed=3, environ=SIM)
    draws = sorted(oracle.draw_latency_ms() for _ in range(DRAWS))
    assert abs(draws[DRAWS // 2] / prof.latency_ms.p50 - 1) < 0.1
    assert abs(draws[int(DRAWS * 0.95)] / prof.latency_ms.p95 - 1) < 0.15


def test_refuses_outside_the_simulator() -> None:
    with pytest.raises(RuntimeError, match="SD_SIM=1"):
        OracleMatcher(lambda _: Truth(True, True), environ={})


def test_start_profile_is_labelled_modelled() -> None:
    prof = load_profile("vlm_oracle_v0")
    assert prof.profile_id == "vlm_oracle_v0" and prof.label == "modelled"


def test_bad_profile_is_refused() -> None:
    with pytest.raises(ValueError, match=r"add up|above 1"):
        VlmProfile.model_validate(
            load_profile().model_dump()
            | {"person": {"tp_yes": 0.9, "tp_unsure": 0.2, "fp_yes": 0, "fp_unsure": 0}}
        )


def test_assessment_is_marked_oracle() -> None:
    oracle = OracleMatcher(lambda _: Truth(True, True), seed=1, environ=SIM)
    req = Req("0192f4a1-7c2e-7d10-9a3b-5e1f0c2d8a23", _profile())
    a = asyncio.run(oracle.assess(req))
    assert a.engine == "oracle" and a.model_id == "oracle/vlm_oracle_v0"
    assert set(a.fields) == {"garment:0", "state"}
    no_profile = asyncio.run(oracle.assess(Req("0192f4a1-7c2e-7d10-9a3b-5e1f0c2d8a23")))
    assert no_profile.fields == {}
