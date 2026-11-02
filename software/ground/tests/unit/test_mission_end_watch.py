"""The alarm for a search mission's built-in end flying without the onboard computer: the
flight controller in AUTO on the hover at the end (``MISSION_CURRENT``) for longer
than the onboard computer needs to switch to LOITER. Its Swedish text matches the panel's
(``onboard.MISSION_FALLBACK`` in sv.json)."""

from __future__ import annotations

import dataclasses
import json
from pathlib import Path
from typing import Any

from sd_ground.mav.mission_end_watch import (
    FALLBACK_GRACE_MS,
    MISSION_FALLBACK_SV,
    MissionEndWatch,
)
from sd_mav.mission import survey_mission
from sd_mav.modes import CopterMode
from sd_mav.vehicle import FcTelemetry
from sd_schemas.base import LatLonAlt

SV = Path(__file__).resolve().parents[3] / "panel" / "src" / "i18n" / "sv.json"
HOME = LatLonAlt(lat=-35.36, lon=149.16, alt_amsl_m=584.0)
# A search mission of 12 points as the onboard computer uploads it: 15 items, 0 is home.
ITEMS = survey_mission(HOME, [(-35.36 + i * 1e-4, 149.16) for i in range(12)], 50.0, 8.0)
TOTAL = len(ITEMS) - 1  # MISSION_CURRENT.total leaves home out: the last item has seq == total
END_SEQ = TOTAL  # the hover at the last point
LAST_POINT_SEQ = TOTAL - 1


class Sink:
    def __init__(self) -> None:
        self.events: list[tuple[str, str, dict[str, Any]]] = []

    def __call__(self, kind: str, severity: str, payload: dict[str, Any]) -> None:
        self.events.append((kind, severity, payload))


def tel(seq: int | None, total: int | None = TOTAL) -> FcTelemetry:
    return dataclasses.replace(FcTelemetry(), mission_seq_current=seq, mission_total=total)


def run(watch: MissionEndWatch, steps: list[tuple[int, FcTelemetry, CopterMode | None]]) -> None:
    for now, t, mode in steps:
        watch.tick(t, mode, now, in_flight=True)


def test_the_end_item_is_the_hover_with_no_end() -> None:
    assert ITEMS[END_SEQ].command == 17  # NAV_LOITER_UNLIM
    assert ITEMS[LAST_POINT_SEQ].command == 16  # the last search point


def test_alarm_after_the_grace_on_the_hover_in_auto() -> None:
    sink = Sink()
    w = MissionEndWatch(sink)
    run(w, [(0, tel(END_SEQ), CopterMode.AUTO)])
    run(w, [(FALLBACK_GRACE_MS - 1, tel(END_SEQ), CopterMode.AUTO)])
    assert sink.events == []
    run(w, [(FALLBACK_GRACE_MS, tel(END_SEQ), CopterMode.AUTO)])
    assert len(sink.events) == 1
    kind, severity, payload = sink.events[0]
    assert (kind, severity) == ("alarm.MISSION_FALLBACK", "critical")
    assert payload["code"] == "MISSION_FALLBACK"
    assert payload["message_sv"] == MISSION_FALLBACK_SV
    # Once per episode: hovering on, no second alarm.
    run(w, [(FALLBACK_GRACE_MS + 30_000, tel(END_SEQ), CopterMode.AUTO)])
    assert len(sink.events) == 1


def test_the_onboard_computer_taking_over_in_time_raises_nothing() -> None:
    """The usual end of a search: the last point reached, the hover item begins, and
    the onboard computer switches to LOITER a moment later."""
    sink = Sink()
    w = MissionEndWatch(sink)
    run(
        w,
        [
            (0, tel(LAST_POINT_SEQ), CopterMode.AUTO),
            (1_000, tel(END_SEQ), CopterMode.AUTO),
            (2_000, tel(END_SEQ), CopterMode.LOITER),
            (60_000, tel(END_SEQ), CopterMode.LOITER),
        ],
    )
    assert sink.events == []


def test_search_points_rtl_mode_and_no_total_raise_nothing() -> None:
    sink = Sink()
    w = MissionEndWatch(sink)
    for now in range(0, 60_000, 1_000):
        w.tick(tel(LAST_POINT_SEQ), CopterMode.AUTO, now, in_flight=True)  # a search point
        w.tick(tel(END_SEQ), CopterMode.RTL, now, in_flight=True)  # the mode RTL is FLY HOME
        w.tick(tel(END_SEQ, None), CopterMode.AUTO, now, in_flight=True)  # no total sent
        w.tick(tel(END_SEQ), None, now, in_flight=True)  # no fresh mode
    assert sink.events == []


def test_on_the_ground_nothing_and_a_new_flight_alarms_again() -> None:
    sink = Sink()
    w = MissionEndWatch(sink)
    for now in range(0, 30_000, 1_000):
        w.tick(tel(END_SEQ), CopterMode.AUTO, now, in_flight=False)
    assert sink.events == []
    run(w, [(0, tel(END_SEQ), CopterMode.AUTO), (FALLBACK_GRACE_MS, tel(END_SEQ), CopterMode.AUTO)])
    assert len(sink.events) == 1
    w.reset()  # landed
    run(w, [(0, tel(END_SEQ), CopterMode.AUTO), (FALLBACK_GRACE_MS, tel(END_SEQ), CopterMode.AUTO)])
    assert len(sink.events) == 2


def test_the_text_is_the_panels() -> None:
    panel = json.loads(SV.read_text(encoding="utf-8"))["onboard"]["MISSION_FALLBACK"]
    assert panel == MISSION_FALLBACK_SV
