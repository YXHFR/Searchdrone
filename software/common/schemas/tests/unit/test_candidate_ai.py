"""What the AI sees of the candidate queue: the event body and the queue's counts, shared by
the ground station and the simulator exam."""

from __future__ import annotations

import datetime as dt
from typing import Any

from sd_schemas.candidate_ai import candidate_event_body, candidates_line, queue_counts
from sd_schemas.search import CandidateCard, priority_class

T = dt.datetime(2026, 10, 17, 12, 0, tzinfo=dt.UTC)


def card(seq: int, priority: float, state: str = "ready", **over: Any) -> CandidateCard:
    return CandidateCard.model_validate(
        {
            "candidate_id": f"0192f49e-2000-7000-8000-{seq:012d}",
            "rev": 1,
            "seq": seq,
            "state": state,
            "priority": priority,
            "priority_class": priority_class(priority),
            "lat": -35.3615,
            "lon": 149.1652,
            "sigma95_m": 4.1,
            "n_seen": 1,
            "n_in_view": 2,
            "best_detector_score": 0.8,
            "thumb_detection_id": "0192f49e-3000-7000-8000-000000000009",
            "first_seen_utc": T,
            "last_seen_utc": T,
            **over,
        }
    )


def test_only_a_high_candidate_waiting_for_a_person_has_an_event_body() -> None:
    body = candidate_event_body(card(3, 0.72).model_dump(mode="json"))
    assert body is not None
    assert body["seq"] == 3 and body["priority_class"] == "high" and body["state"] == "ready"
    assert body["detection_id"] == "0192f49e-3000-7000-8000-000000000009"
    assert body["multi_frame"] is False
    two = candidate_event_body(card(3, 0.72, "assessing", n_seen=2).model_dump(mode="json"))
    assert two is not None and two["multi_frame"] is True
    assert candidate_event_body(card(3, 0.4).model_dump(mode="json")) is None
    for state in ("new", "inspecting", "confirmed", "rejected", "merged"):
        assert candidate_event_body(card(3, 0.72, state).model_dump(mode="json")) is None


def test_queue_counts_and_the_line_leave_out_decided_and_merged() -> None:
    cards = [
        card(1, 0.7),
        card(2, 0.4),
        card(3, 0.1),
        card(4, 0.9, "confirmed"),
        card(5, 0.9, "rejected"),
        card(6, 0.9, "merged"),
    ]
    assert queue_counts(cards) == {"review": 2, "low": 1, "confirmed": 1, "rejected": 1}
    assert candidates_line(cards, {}, None) == (
        "Kandidater: K1 HÖG (1/2 bilder) | K2 medel (1/2 bilder) | K3 låg (1/2 bilder)"
        " | att granska 2, låg 1"
    )
    assert candidates_line([], {}, None) is None
    # With the drone's position: how far and which way, from the drone.
    drone = (-35.3615 - 40 / 111_320, 149.1652)
    assert candidates_line([card(1, 0.7)], {}, drone) == (
        "Kandidater: K1 HÖG (1/2 bilder) 40 m N | att granska 1, låg 0"
    )
