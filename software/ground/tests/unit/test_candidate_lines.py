"""The AI's candidate and find lines in the situation: fixed words, never free text."""

from __future__ import annotations

import asyncio
import datetime as dt
from typing import Any

from sd_ground.services.candidate_lines import candidate_lines, finds_line
from sd_schemas.candidate_ai import candidates_line
from sd_schemas.geo import plane_offset
from sd_schemas.search import CandidateCard, SearchProfile, priority_class

T = dt.datetime(2026, 10, 17, 12, 0, tzinfo=dt.UTC)
DRONE = (-35.3630, 149.1650)
MISSION = "0192f49e-0b3a-7c55-8e2f-1a2b3c4d5e6f"


def card(seq: int, *, priority: float = 0.7, state: str = "ready", **over: Any) -> CandidateCard:
    east, north = over.pop("at", (0.0, 0.0))
    lat, lon = plane_offset(DRONE[0], DRONE[1], east, north)
    return CandidateCard.model_validate(
        {
            "candidate_id": f"0192f49e-2000-7000-8000-{seq:012d}",
            "rev": 1,
            "seq": seq,
            "state": state,
            "priority": priority,
            "priority_class": priority_class(priority),
            "lat": lat,
            "lon": lon,
            "sigma95_m": 4.4,
            "n_seen": 2,
            "n_in_view": 3,
            "best_detector_score": 0.8,
            "first_seen_utc": T,
            "last_seen_utc": T,
            **over,
        }
    )


def profile(version: int = 1) -> SearchProfile:
    return SearchProfile.model_validate(
        {
            "profile_id": "0192f49e-1000-7000-8000-000000000009",
            "profile_version": version,
            "mission_id": MISSION,
            "subject_category": "dementia",
            "garments": [
                {
                    "part": "upper",
                    "kind": "Fritext-jacka",
                    "colors": ["red", "black"],
                    "certainty": "certain",
                    "source": "Påhittad källa",
                }
            ],
            "items": [{"kind": "backpack", "colors": ["blue"], "certainty": "likely"}],
            "last_seen": {"lat": -35.3620, "lon": 149.1640, "at_utc": T},
            "visual_notes": "Påhittad anteckning",
            "free_text": "Påhittad fritext",
            "approved_by": "operator:demo",
            "approved_at": T,
        }
    )


def test_the_best_open_candidates_first_with_fields_distance_and_counts() -> None:
    cards = [
        card(
            3,
            priority=0.8,
            at=(85.0, 85.0),
            engine="fast",
            fields={"garment:0": "match", "item:0": "match"},
            assessed_profile_version=1,
        ),
        card(
            5,
            priority=0.45,
            at=(0.0, -40.0),
            engine="code",
            fields={"garment:0": "conflict"},
            assessed_profile_version=1,
            n_seen=1,
            n_in_view=1,
        ),
        card(1, priority=0.9, state="confirmed"),
        card(2, priority=0.95, state="merged"),
        card(4, priority=0.2, state="new"),
        card(6, priority=0.1, state="assessing"),
        card(
            7,
            priority=0.35,
            state="inspecting",
            at=(10.0, 0.0),
            fields={"hair": "match", "state": "match"},
            assessed_profile_version=1,
        ),
        card(8, priority=0.32),
    ]
    line = candidates_line(cards, {1: profile()}, DRONE)
    assert line == (
        "Kandidater: K3 HÖG (överdel röd passar, ryggsäck blå passar; 2/3 bilder; moln snabb)"
        " 120 m NO | K5 medel (överdel röd krockar; 1/1 bilder; kod) 40 m S"
        " | K7 medel (tillstånd passar; 2/3 bilder) 10 m O | att granska 4, låg 2"
    )
    # Nothing a person typed reaches the line.
    assert "Påhittad" not in line and "Fritext" not in line


def test_no_distance_without_the_drone_and_no_fields_without_the_assessed_profile() -> None:
    cards = [card(2, fields={"garment:0": "not_visible"}, assessed_profile_version=2)]
    assert candidates_line(cards, {1: profile()}, None) == (
        "Kandidater: K2 HÖG (2/3 bilder) | att granska 1, låg 0"
    )
    cards = [card(2, fields={"garment:0": "not_visible"}, assessed_profile_version=1)]
    assert candidates_line(cards, {1: profile()}, None) == (
        "Kandidater: K2 HÖG (överdel röd okänt; 2/3 bilder) | att granska 1, låg 0"
    )


def test_finds() -> None:
    assert candidates_line([], {}, DRONE) is None
    assert finds_line([], {}) is None
    assert finds_line([card(1)], {}) == "Fynd: inga bekräftade"
    cards = [card(3, state="confirmed"), card(1, state="confirmed", sigma95_m=12.6)]
    at = {str(cards[0].candidate_id): T.replace(hour=14, minute=32)}
    assert finds_line(cards, at) == (
        "Fynd: K1 BEKRÄFTAD, cylinder 43 m | K3 BEKRÄFTAD 14:32, cylinder 34 m"
    )


def test_a_fault_never_takes_the_situation_down() -> None:
    assert asyncio.run(candidate_lines(object(), MISSION, None)) == (None, None)  # type: ignore[arg-type]
