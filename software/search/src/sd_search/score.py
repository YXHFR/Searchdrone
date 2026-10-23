"""The match score of a candidate against the search profile, and its person probability.

``match = sum(w * c * s) / sum(w * c)`` over the fields that are not ``not_visible``
(``s`` is 1 for ``match`` and 0 for ``conflict``), with ``w`` the field's weight and ``c``
the factor of the operator's certainty. ``None`` when nothing could be seen: no evidence is
not evidence against (a field that is not visible never lowers the score).

The image model's verdicts win over the code's, except that the code's ``match`` stays
when the model says ``not_visible`` (the colours were there in the pixels).
"""

from __future__ import annotations

from collections.abc import Mapping
from typing import Final

from sd_schemas.search import Certainty, FieldVerdict, SearchProfile

__all__ = [
    "CERTAINTY_FACTOR",
    "PART_WEIGHT",
    "field_weight",
    "match_score",
    "merge_fields",
    "person_probability",
]

PART_WEIGHT: Final[dict[str, float]] = {
    "outer": 1.0,
    "upper": 1.0,
    "full_body": 1.0,
    "lower": 0.4,
    "headwear": 0.7,
    "footwear": 0.1,
}
ITEM_WEIGHT: Final[dict[str, float]] = {
    "backpack": 1.0,
    "bag": 1.0,
    "walker": 0.6,
    "bicycle": 0.6,
    "dog": 0.6,
    "umbrella": 0.6,
    "cane": 0.3,
    "other": 0.3,
}
HAIR_WEIGHT: Final = 0.7
STATE_WEIGHT: Final = 0.2
CERTAINTY_FACTOR: Final[dict[Certainty, float]] = {"certain": 1.0, "likely": 0.7, "unsure": 0.4}
STATE_CERTAINTY: Final[Certainty] = "likely"  # the likely state carries no certainty of its own


def field_weight(profile: SearchProfile, key: str) -> float:
    """``w * c`` of one field key of the profile (0 for a key the profile lacks)."""
    kind, _, index = key.partition(":")
    if kind == "garment" and int(index) < len(profile.garments):
        g = profile.garments[int(index)]
        return PART_WEIGHT[g.part] * CERTAINTY_FACTOR[g.certainty]
    if kind == "item" and int(index) < len(profile.items):
        item = profile.items[int(index)]
        return ITEM_WEIGHT[item.kind] * CERTAINTY_FACTOR[item.certainty]
    if kind == "hair" and profile.hair is not None:
        return HAIR_WEIGHT * CERTAINTY_FACTOR[profile.hair.certainty]
    if kind == "state":
        return STATE_WEIGHT * CERTAINTY_FACTOR[STATE_CERTAINTY]
    return 0.0


def match_score(profile: SearchProfile, fields: Mapping[str, FieldVerdict]) -> float | None:
    num = den = 0.0
    for key, verdict in fields.items():
        if verdict == "not_visible":
            continue
        w = field_weight(profile, key)
        den += w
        num += w if verdict == "match" else 0.0
    return None if den == 0 else round(num / den, 4)


def merge_fields(
    code: Mapping[str, FieldVerdict], model: Mapping[str, FieldVerdict] | None
) -> dict[str, FieldVerdict]:
    """The model's verdict per field, except a code ``match`` the model could not see."""
    out: dict[str, FieldVerdict] = dict(code)
    for key, verdict in (model or {}).items():
        if verdict == "not_visible" and out.get(key) == "match":
            continue
        out[key] = verdict
    return out


def person_probability(
    model_person_p: float | None, best_detector_score: float, refined: float | None
) -> float:
    """The image model's value if there is one, else the best detector evidence."""
    if model_person_p is not None:
        return model_person_p
    return max(best_detector_score, refined if refined is not None else 0.0)
