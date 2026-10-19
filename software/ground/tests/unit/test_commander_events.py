"""What the commander's event stream says about the pilot's buttons."""

from __future__ import annotations

import datetime as dt
from typing import Any

from sd_ground.api.commander import to_commander
from sd_ground.bus import BusEvent

NOW = dt.datetime(2026, 10, 17, 12, 0, tzinfo=dt.UTC)


def event(type_: str, data: dict[str, Any]) -> BusEvent:
    return BusEvent(
        seq=7, ts=NOW, topic="vehicle", type=type_, vehicle_id="sim-01", mission_id=None, data=data
    )


def outcome(action: str, result: str = "dispatched") -> dict[str, Any]:
    return {
        "action_id": "a1",
        "action": action,
        "via": "button",
        "by": "operator:pilot",
        "result": result,
    }


def test_the_pilots_resume_wakes_the_ai_as_an_override() -> None:
    out = to_commander(event("vehicle.action", outcome("resume")))
    assert out is not None
    assert out["type"] == "event.operator_override"
    assert out["body"] == {
        "active": False,
        "action": "resume",
        "by": "operator:pilot",
        "via": "button",
    }
    # Later updates of the same press do not wake it again.
    assert to_commander(event("vehicle.action", outcome("resume", "acked"))) is None


def test_other_actions_stay_vehicle_actions() -> None:
    out = to_commander(event("vehicle.action", outcome("stop")))
    assert out is not None and out["type"] == "event.vehicle_action"
    assert out["body"]["action"] == "stop"


def test_a_lifted_latch_is_not_a_resume() -> None:
    """A latch also goes at landing: "no latch" alone never wakes the AI; the pilot's
    Resume does, and it must not be hidden behind a later "no latch" in the same turn."""
    lifted = {"active": False, "action": None, "set_by": None, "via": "button"}
    assert to_commander(event("latch.updated", lifted)) is None
    stop = {"active": True, "action": "stop", "set_by": "operator:pilot", "via": "button"}
    out = to_commander(event("latch.updated", stop))
    assert out is not None and out["type"] == "event.operator_override"
    assert out["body"]["action"] == "stop"


def test_a_new_find_reaches_the_ai_and_updates_do_not() -> None:
    find = {
        "find_id": "f1",
        "label": "red_block",
        "name_sv": "rött block",
        "kind": "target",
        "source": "color",
        "score": 0.7,
        "lat": -35.3615,
        "lon": 149.1652,
        "sigma95_m": 5.2,
        "count": 1,
        "thumb_detection_id": "f1",
        "synthetic": True,
    }
    out = to_commander(event("find.new", find))
    assert out is not None and out["type"] == "event.detection"
    assert out["body"] == {
        k: v for k, v in find.items() if k not in ("count", "thumb_detection_id")
    }
    assert to_commander(event("find.updated", find)) is None
    assert to_commander(event("detection.new", find)) is None
