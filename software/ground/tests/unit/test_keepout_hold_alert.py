"""The operator's alarm for a keep-out hover (``executor.keepout_hold`` from the drone):
red over a person, orange short of a cylinder; the ground's Swedish text matches the
panel's (``keepouts.HOLD*`` in sv.json) and both read the same fields."""

from __future__ import annotations

import json
import uuid
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import pytest

from sd_ground.services.missions import MissionFlow, keepout_hold_text
from sd_schemas.api import KeepoutHoldField
from sd_schemas.mission import MissionState

SV = Path(__file__).resolve().parents[3] / "panel" / "src" / "i18n" / "sv.json"


def _panel() -> dict[str, str]:
    return json.loads(SV.read_text(encoding="utf-8"))["keepouts"]


def test_short_of_a_cylinder() -> None:
    assert keepout_hold_text({"inside": False}) == _panel()["HOLD"]


def test_over_a_person_climbing() -> None:
    text = keepout_hold_text({"inside": True, "climb_to_m": 63.4})
    assert text == _panel()["HOLD_CLIMB"].replace("{{alt}}", "63")
    assert "FLYG HEM" in text and "RC" in text


def test_over_a_person_unable_to_climb() -> None:
    assert keepout_hold_text({"inside": True}) == _panel()["HOLD_INSIDE"]


def test_half_metres_round_as_on_the_panel() -> None:
    """The panel's Math.round: 62.5 -> 63 (Python's own round would give 62)."""
    text = keepout_hold_text({"inside": True, "climb_to_m": 62.5})
    assert text == _panel()["HOLD_CLIMB"].replace("{{alt}}", "63")


def test_the_panel_reads_the_same_fields() -> None:
    gnss = (SV.parents[1] / "safety" / "gnss.ts").read_text(encoding="utf-8")
    for name in KeepoutHoldField:  # typed there as the generated KeepoutHoldField
        assert f'KeepoutHoldField = "{name.value}"' in gnss


class _Flow:
    def __init__(self) -> None:
        self.alerts: list[tuple[str, str, str, dict[str, Any]]] = []

    def alert(self, code: str, severity: str, message_sv: str, **kw: Any) -> None:
        self.alerts.append((code, severity, message_sv, kw["details"]))


def _alert(data: dict[str, Any], status: MissionState = MissionState.ACTIVE) -> _Flow:
    flow = _Flow()
    mission = SimpleNamespace(id=uuid.uuid4(), status=status)
    MissionFlow._keepout_hold_alert(flow, "sd-01", mission, data)  # type: ignore[arg-type]  # pyright: ignore[reportPrivateUsage]
    return flow


def test_short_of_a_cylinder_is_orange() -> None:
    flow = _alert({"command_id": "c", "skill": "goto", "inside": False})
    assert [(c, s) for c, s, _, _ in flow.alerts] == [("KEEPOUT_HOLD", "warning")]
    assert flow.alerts[0][3] == {"inside": False}


def test_over_a_person_is_red_with_the_climb() -> None:
    flow = _alert({"inside": True, "climb_to_m": 63.4})
    code, severity, text, details = flow.alerts[0]
    assert (code, severity) == ("KEEPOUT_HOLD", "critical")
    assert details == {"inside": True, "climb_to_m": 63.4} and "63 m" in text


@pytest.mark.parametrize("climb", ["63", True, None, float("nan"), [63]])
def test_a_climb_that_is_not_a_number_is_left_out(climb: Any) -> None:
    """Red all the same: the drone is over a person; the text says it does not climb."""
    flow = _alert({"inside": True, "climb_to_m": climb})
    _, severity, text, details = flow.alerts[0]
    assert severity == "critical" and details == {"inside": True}
    assert text == _panel()["HOLD_INSIDE"]


def test_a_hover_held_back_past_its_mission_is_only_logged() -> None:
    """Held back by a lost link and delivered after its mission closed: that hover is
    over, so no alarm (the event itself is stored as usual)."""
    assert _alert({"inside": True}, MissionState.CLOSED).alerts == []
