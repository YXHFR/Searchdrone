"""What the AI sees of the candidate queue: the ``candidate`` event and the "Kandidater"
line in ``Situation``.

The ground station builds both from its cards; the simulator exam builds them from its
own cards with the same functions, so the AI meets one path in both.

Read-only, and built from enum-like values only (priority class, profile part, colour
and item words from the schema's vocabulary, verdicts, counts, distance): never from a
profile's notes or descriptions, a person's decision note or anything else typed by a
person. The AI sees the candidates; only a person confirms a find.
"""

from __future__ import annotations

import math
from collections.abc import Iterable, Mapping
from typing import Any, Final

from sd_schemas.colors import COLOR_TABLE, ColorName
from sd_schemas.geo import local_xy
from sd_schemas.search import (
    PRIORITY_MEDIUM,
    AssessmentEngine,
    CandidateCard,
    CandidateState,
    FieldVerdict,
    PriorityClass,
    SearchProfile,
)

__all__ = [
    "LINE_CANDIDATES_MAX",
    "candidate_event_body",
    "candidates_line",
    "queue_counts",
]

LINE_CANDIDATES_MAX: Final = 3
_S = CandidateState
_OPEN: Final = (_S.READY, _S.ASSESSING, _S.INSPECTING)
_CLASS: Final[Mapping[PriorityClass, str]] = {"high": "HÖG", "medium": "medel", "low": "låg"}
_VERDICT: Final[Mapping[FieldVerdict, str]] = {
    "match": "passar",
    "conflict": "krockar",
    "not_visible": "okänt",
}
_ENGINE: Final[Mapping[AssessmentEngine, str]] = {
    "code": "kod",
    "fast": "moln snabb",
    "strong": "moln",
    "oracle": "orakel",  # the simulator's own labels
}
# The panel's words (profile.parts / profile.itemKinds), lower case.
_PART: Final = {
    "outer": "ytterplagg",
    "upper": "överdel",
    "lower": "underdel",
    "headwear": "huvudbonad",
    "footwear": "skor",
    "full_body": "helkropp",
}
_ITEM: Final = {
    "backpack": "ryggsäck",
    "bag": "väska",
    "cane": "käpp",
    "walker": "rollator",
    "umbrella": "paraply",
    "bicycle": "cykel",
    "dog": "hund",
    "other": "föremål",
}
_STATE: Final = {"walking": "går", "sitting": "sitter", "lying": "ligger"}
_COMPASS: Final = ("N", "NO", "O", "SO", "S", "SV", "V", "NV")
_EVENT_KEYS: Final = (
    "candidate_id",
    "seq",
    "priority_class",
    "priority",
    "state",
    "lat",
    "lon",
    "sigma95_m",
    "n_seen",
    "n_in_view",
    "engine",
)


def candidate_event_body(card: Mapping[str, Any]) -> dict[str, Any] | None:
    """The body of ``event.candidate`` for a new or updated card (its JSON form), or None.

    Only a high candidate waiting for a person wakes the AI; it sees the rest in the
    situation's candidates line. It never decides one: only a person confirms."""
    if card.get("priority_class") != "high" or card.get("state") not in ("ready", "assessing"):
        return None
    body = {k: card.get(k) for k in _EVENT_KEYS}
    body["detection_id"] = card.get("thumb_detection_id")
    body["multi_frame"] = int(card.get("n_seen") or 0) >= 2
    return body


def queue_counts(cards: Iterable[CandidateCard]) -> dict[str, int]:
    """The queue's counts: to review (medium or high), low, confirmed and rejected;
    merged candidates are not counted."""
    counts = {"review": 0, "low": 0, "confirmed": 0, "rejected": 0}
    for c in cards:
        if c.state in (CandidateState.CONFIRMED, CandidateState.REJECTED):
            counts[c.state.value] += 1
        elif c.state is not CandidateState.MERGED:
            counts["review" if c.priority >= PRIORITY_MEDIUM else "low"] += 1
    return counts


def _color(colors: Iterable[ColorName]) -> str | None:
    first = next((c for c in colors if c is not ColorName.UNKNOWN), None)
    return None if first is None else COLOR_TABLE[first].sv


def _field_word(profile: SearchProfile, key: str) -> str | None:
    """The profile field an assessment key names, in a few fixed words ("överdel röd")."""
    kind, _, index = key.partition(":")
    if kind == "garment" and index.isdigit() and int(index) < len(profile.garments):
        g = profile.garments[int(index)]
        words = [_PART[g.part], _color(g.colors)]
    elif kind == "item" and index.isdigit() and int(index) < len(profile.items):
        it = profile.items[int(index)]
        words = [_ITEM[it.kind], _color(it.colors)]
    elif kind == "hair" and profile.hair is not None:
        words = ["hår", _color([profile.hair.color])]
    elif kind == "state":
        state = next((x for x in profile.likely_state if x in _STATE), None)
        words = ["tillstånd", None if state is None else _STATE[state]]
    else:
        return None
    return " ".join(w for w in words if w)


def _where(card: CandidateCard, drone: tuple[float, float] | None) -> str | None:
    if drone is None:
        return None
    east, north = local_xy(drone[0], drone[1], card.lat, card.lon)
    deg = math.degrees(math.atan2(east, north)) % 360
    return f"{round(math.hypot(east, north))} m {_COMPASS[round(deg / 45) % 8]}"


def _one(
    card: CandidateCard,
    profiles: Mapping[int, SearchProfile],
    drone: tuple[float, float] | None,
) -> str:
    items: list[str] = []
    # The keys name the fields of the profile version that was assessed, not the newest.
    profile = profiles.get(card.assessed_profile_version or 0)
    if profile is not None:
        fields = [
            f"{word} {_VERDICT[verdict]}"
            for key, verdict in card.fields.items()
            if (word := _field_word(profile, key))
        ]
        if fields:
            items.append(", ".join(fields))
    items.append(f"{card.n_seen}/{card.n_in_view} bilder")
    if card.engine is not None:
        items.append(_ENGINE[card.engine])
    out = f"K{card.seq} {_CLASS[card.priority_class]} ({'; '.join(items)})"
    where = _where(card, drone)
    return out if where is None else f"{out} {where}"


def candidates_line(
    cards: list[CandidateCard],
    profiles: Mapping[int, SearchProfile],
    drone: tuple[float, float] | None,
) -> str | None:
    """The open candidates, best first, then the queue's counts: "Kandidater: K3 HÖG
    (överdel röd passar, ryggsäck blå passar; 2/3 bilder; moln snabb) 120 m NO | ... |
    att granska 4, låg 11". None for a mission without candidates."""
    if not cards:
        return None
    counts = queue_counts(cards)
    shown = sorted((c for c in cards if c.state in _OPEN), key=lambda c: (-c.priority, c.seq))
    parts = [_one(c, profiles, drone) for c in shown[:LINE_CANDIDATES_MAX]]
    parts.append(f"att granska {counts['review']}, låg {counts['low']}")
    return "Kandidater: " + " | ".join(parts)
