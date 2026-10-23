"""The search-profile evaluation: the scoring, and the recorded cloud answers replayed."""

from __future__ import annotations

import datetime as dt
import json
from pathlib import Path
from typing import Any

from sd_brain.evals.profiles import (
    FIELD_ACCURACY_MIN,
    INVENTED_CERTAIN_MAX,
    load_cases,
    load_profile_golden,
    replay_for,
    run_profiles,
    score_profile,
)
from sd_brain.simeval.golden import refuse_secrets
from sd_schemas.search import SearchProfileDraft

SUITE = Path(__file__).resolve().parents[1] / "eval" / "profiles"
SEEN = {"lat": 1.0, "lon": 2.0, "at_utc": dt.datetime(2026, 1, 1, tzinfo=dt.UTC).isoformat()}
CASE: dict[str, Any] = {
    "id": "T",
    "expect": {
        "garments": [
            {"part": ["outer", "upper"], "colors": ["red"]},
            {"part": "lower", "colors": ["blue"], "unsure": True},
        ],
        "items": [{"kind": "backpack", "colors": ["yellow"]}],
    },
}


def _draft(**kw: Any) -> SearchProfileDraft:
    return SearchProfileDraft.model_validate({"subject_category": "hiker", "last_seen": SEEN, **kw})


def _g(part: str, color: str, certainty: str = "certain") -> dict[str, Any]:
    return {"part": part, "colors": [color], "certainty": certainty}


def test_a_right_profile_scores_every_field() -> None:
    d = _draft(
        garments=[_g("upper", "red"), _g("lower", "blue", "unsure")],
        items=[{"kind": "backpack", "colors": ["yellow"], "certainty": "certain"}],
    )
    s = score_profile(CASE, d)
    assert (s.correct, s.fields, s.invented_certain, s.overconfident) == (6, 6, 0, 0)


def test_invented_and_wrong_colours_marked_certain_count() -> None:
    d = _draft(
        garments=[_g("outer", "green"), _g("lower", "blue"), _g("headwear", "black")],
        hair={"color": "red", "length": "long", "certainty": "certain"},
    )
    s = score_profile(CASE, d)
    # A wrong colour, a cap and hair nobody mentioned, all marked certain.
    assert s.invented_certain == 3
    assert s.overconfident == 1  # the guessed trousers came back certain
    assert s.correct == 3 and "item:0" in s.misses


def test_an_invented_unsure_field_is_not_counted() -> None:
    d = _draft(garments=[_g("outer", "red"), _g("headwear", "black", "unsure")])
    assert score_profile(CASE, d).invented_certain == 0


def test_no_proposal_scores_nothing() -> None:
    s = score_profile(CASE, None)
    assert not s.proposed and s.correct == 0 and s.fields == 6


def test_suite_is_twenty_fictional_cases_with_a_recorded_answer() -> None:
    cases = load_cases(SUITE / "cases.yaml")
    ids = [c["id"] for c in cases]
    assert len(cases) == 20 and len(set(ids)) == 20
    golden = load_profile_golden(SUITE / "golden.jsonl")
    assert set(golden) <= set(ids)
    rows = [r["reply"] for rs in golden.values() for r in rs]
    refuse_secrets(rows, {})  # no key or personal data in what the cloud answered


async def test_replay_meets_the_bar() -> None:
    """The cloud model's recorded answers, replayed: what CI checks on every change."""
    cases = load_cases(SUITE / "cases.yaml")
    golden = load_profile_golden(SUITE / "golden.jsonl")
    report = await run_profiles(cases, replay_for(golden))
    summary = json.dumps({k: v for k, v in report.items() if k != "per_case"})
    assert report["field_accuracy"] >= FIELD_ACCURACY_MIN, summary
    assert report["invented_certain"] <= INVENTED_CERTAIN_MAX, summary
    assert report["passed"]
