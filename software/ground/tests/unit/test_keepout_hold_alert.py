"""The operator's alarm for a keep-out hover (``executor.keepout_hold`` from the drone):
the ground's Swedish text matches the panel's (``keepouts.HOLD*`` in sv.json)."""

from __future__ import annotations

import json
from pathlib import Path

from sd_ground.services.missions import keepout_hold_text

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
