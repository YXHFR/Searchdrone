"""The AI's two candidate lines in ``Situation``: the open candidates and the confirmed
finds.

Read-only, and built from enum-like values only (priority class, profile part, colour
and item words from the schema's vocabulary, verdicts, counts, distance): never from a
profile's notes or descriptions, a person's decision note or anything else typed by a
person. The AI sees the candidates; only a person confirms a find.
"""

from __future__ import annotations

import datetime as dt
import math
from collections.abc import Iterable, Mapping
from typing import TYPE_CHECKING, Final

from sd_ground.logging import get_logger
from sd_safety.constants import PERSON_KEEPOUT_M
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

if TYPE_CHECKING:
    from sd_ground.mav.thread import MavSnapshot
    from sd_ground.services.missions import MissionFlow

__all__ = [
    "LINE_CANDIDATES_MAX",
    "candidate_lines",
    "candidates_line",
    "drone_position",
    "finds_line",
]

log = get_logger("sd_ground.candidate_lines")

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


def drone_position(mav: MavSnapshot | None) -> tuple[float, float] | None:
    """The drone's (lat, lon) when it is connected with a 3D fix, else None."""
    if mav is None or not mav.connected or mav.telemetry.gps_fix_type < 3:
        return None
    return mav.telemetry.lat, mav.telemetry.lon


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
    # The same counts as the queue's (``candidates.cards``).
    review = low = 0
    for c in cards:
        if c.state not in (_S.CONFIRMED, _S.REJECTED, _S.MERGED):
            if c.priority >= PRIORITY_MEDIUM:
                review += 1
            else:
                low += 1
    shown = sorted((c for c in cards if c.state in _OPEN), key=lambda c: (-c.priority, c.seq))
    parts = [_one(c, profiles, drone) for c in shown[:LINE_CANDIDATES_MAX]]
    parts.append(f"att granska {review}, låg {low}")
    return "Kandidater: " + " | ".join(parts)


def finds_line(cards: list[CandidateCard], confirmed_at: Mapping[str, dt.datetime]) -> str | None:
    """The confirmed finds: "Fynd: K3 BEKRÄFTAD 14:32, cylinder 34 m" each, "Fynd: inga
    bekräftade" without one, or None for a mission without candidates. The cylinder is
    the keep-out around a person (PERSON_KEEPOUT_M + the position's sigma95_m)."""
    if not cards:
        return None
    found = sorted((c for c in cards if c.state is _S.CONFIRMED), key=lambda c: c.seq)
    if not found:
        return "Fynd: inga bekräftade"
    parts: list[str] = []
    for c in found:
        when = confirmed_at.get(str(c.candidate_id))
        # UTC, like every other clock in the AI's text (the chat lines' times).
        clock = "" if when is None else f" {when.astimezone(dt.UTC):%H:%M}"
        parts.append(
            f"K{c.seq} BEKRÄFTAD{clock}, cylinder {round(PERSON_KEEPOUT_M + c.sigma95_m)} m"
        )
    return "Fynd: " + " | ".join(parts)


async def candidate_lines(
    flow: MissionFlow, mission_id: str, mav: MavSnapshot | None
) -> tuple[str | None, str | None]:
    """(candidates_line, finds_line) for the situation. A fault here never takes the rest
    of the situation with it: it is logged and both lines are None."""
    from sd_ground.services import candidates as candidates_service

    try:
        cards = await candidates_service.card_list(flow, mission_id)
        if not cards:
            return None, None
        profiles = {
            p.profile_version: p
            for p in (
                SearchProfile.model_validate(r["body"])
                for r in await flow.repo.search_profile_versions(mission_id)
            )
            if p.approved
        }
        r = candidates_service.repo(flow)
        confirmed_at: dict[str, dt.datetime] = {}
        for c in cards:
            if c.state is _S.CONFIRMED:
                confirms = [
                    d for d in await r.decisions(str(c.candidate_id)) if d["decision"] == "confirm"
                ]
                if confirms:
                    confirmed_at[str(c.candidate_id)] = confirms[-1]["created_at"]
        drone = drone_position(mav)
        return candidates_line(cards, profiles, drone), finds_line(cards, confirmed_at)
    except Exception:
        log.exception("candidates.situation_lines_failed", mission_id=mission_id)
        return None, None
