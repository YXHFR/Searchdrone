"""A new mission reaches the AI as ``event.mission`` with ``created``, so an ``sd-brain``
that was already running takes the mission on (its loop only switches on ``created``)."""

from __future__ import annotations

import datetime as dt

from sd_ground.api.commander import to_commander
from sd_ground.bus import BusEvent


def _event(data: dict[str, object]) -> BusEvent:
    return BusEvent(
        7,
        dt.datetime(2026, 10, 17, tzinfo=dt.UTC),
        "mission",
        "mission.state_changed",
        "sim-01",
        "00000000-0000-4000-8000-000000000001",
        data,
    )


def test_new_mission_is_marked_created() -> None:
    out = to_commander(_event({"from": None, "to": "DRAFT", "created": True, "state_seq": 1}))
    assert out is not None
    assert out["type"] == "event.mission"
    assert out["body"]["created"] is True
    assert out["mission_id"] == "00000000-0000-4000-8000-000000000001"


def test_a_state_change_is_not_a_new_mission() -> None:
    out = to_commander(_event({"from": "DRAFT", "to": "CHARTER_PROPOSED", "state_seq": 2}))
    assert out is not None
    assert out["body"]["created"] is False
