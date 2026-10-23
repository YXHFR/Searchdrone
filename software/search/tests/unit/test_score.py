"""The match score, the person probability and the queue priority."""

from __future__ import annotations

import pytest
from search_helpers import profile  # pyright: ignore[reportMissingImports]

from sd_schemas.search import PRIORITY_MEDIUM, FieldVerdict, priority_class
from sd_search.priority import PRIORITY_MIN, priority
from sd_search.score import field_weight, match_score, person_probability

PROFILE = profile(
    garments=[
        {"part": "upper", "colors": ["red"], "certainty": "certain"},
        {"part": "lower", "colors": ["blue"], "certainty": "likely"},
        {"part": "footwear", "colors": ["black"], "certainty": "unsure"},
    ],
    items=[{"kind": "backpack", "colors": ["yellow"], "certainty": "likely"}],
    hair={"color": "gray", "length": "short", "certainty": "certain"},
)


def test_score_weights_follow_the_part_and_the_certainty() -> None:
    assert field_weight(PROFILE, "garment:0") == 1.0
    assert field_weight(PROFILE, "garment:1") == pytest.approx(0.4 * 0.7)
    assert field_weight(PROFILE, "garment:2") == pytest.approx(0.1 * 0.4)
    assert field_weight(PROFILE, "item:0") == pytest.approx(1.0 * 0.7)
    assert field_weight(PROFILE, "hair") == pytest.approx(0.7)
    assert field_weight(PROFILE, "state") == pytest.approx(0.2 * 0.7)
    assert field_weight(PROFILE, "garment:7") == 0.0


def test_score_not_visible_never_penalizes() -> None:
    seen: dict[str, FieldVerdict] = {"garment:0": "match"}
    assert match_score(PROFILE, seen) == 1.0
    hidden = ("garment:1", "garment:2", "item:0", "hair", "state")
    more: dict[str, FieldVerdict] = dict(seen)
    for key in hidden:
        more[key] = "not_visible"
    assert match_score(PROFILE, more) == 1.0
    assert match_score(PROFILE, dict.fromkeys(PROFILE.field_keys, "not_visible")) is None
    assert match_score(PROFILE, {}) is None
    # And a priority with nothing visible is the neutral one, never lower.
    assert priority(0.6, None, multi_frame=True) == priority(0.6, 0.5, multi_frame=True)


def test_score_conflict_weighs_against() -> None:
    fields: dict[str, FieldVerdict] = {"garment:0": "conflict", "item:0": "match"}
    assert match_score(PROFILE, fields) == pytest.approx(0.7 / 1.7, abs=1e-4)
    assert match_score(PROFILE, {"garment:0": "conflict"}) == 0.0


def test_score_person_probability() -> None:
    assert person_probability(0.4, 0.7) == 0.7
    assert person_probability(0.4, None) == 0.4


def test_priority_formula() -> None:
    # sqrt(0.64) * (0.35 + 0.65) * 1.0 * (0.7 + 0.15) * 1.0
    assert priority(0.64, 1.0, multi_frame=True) == pytest.approx(0.8 * 0.85, abs=1e-4)
    single = priority(0.64, 1.0, multi_frame=False)
    assert single == pytest.approx(0.8 * 0.85 * 0.75, abs=1e-4)
    weak = priority(0.64, 1.0, multi_frame=True, refined=0.2)
    assert weak == pytest.approx(0.8 * 0.85 * 0.8, abs=1e-4)
    assert priority(0.64, 1.0, multi_frame=True, poa_rank=1.0) == pytest.approx(0.8, abs=1e-4)


def test_priority_is_never_zero_and_classes() -> None:
    assert priority(0.0, 0.0, multi_frame=False, poa_rank=0.0, refined=0.0) == PRIORITY_MIN
    low = priority(0.01, 0.2, multi_frame=False)
    assert 0 < low < PRIORITY_MEDIUM and priority_class(low) == "low"
    assert priority_class(priority(1.0, 1.0, multi_frame=True, poa_rank=1.0)) == "high"
