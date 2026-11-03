"""The alarm for a search mission's built-in end flying on its own.

Every search mission ends with a 30 s hover at the last point and a return home
(``sd_mav.mission.survey_mission``), so the flight ends safely even if the onboard
computer stops. When all goes well the flight controller never gets there: the onboard
computer switches to LOITER at the last search point. If the flight controller stays in
AUTO on those last two items, the onboard computer has stopped steering, and the
return home that follows flies straight, also through the keep-out cylinders around
persons, without anyone having decided it.

``MissionEndWatch`` follows ``MISSION_CURRENT`` on the ground's own MAVLink channel, so it
works whatever state the onboard computer is in. It only raises the alarm (once per
episode); it never sends anything to the flight controller.
"""

from __future__ import annotations

from typing import Any, Final

from sd_ground.mav.gnss_watch import EventSink, alarm
from sd_mav.modes import CopterMode
from sd_mav.vehicle import FcTelemetry
from sd_schemas.api import AlertCode

__all__ = ["FALLBACK_GRACE_MS", "MISSION_FALLBACK_SV", "MissionEndWatch", "fallback_alarm"]

# The onboard computer switches to LOITER within a second or two of the last search point;
# this long on the end items in AUTO means it did not. The hover there lasts 30 s.
FALLBACK_GRACE_MS: Final = 10_000
_END_ITEMS: Final = 2  # the hover at the last point and the return home

MISSION_FALLBACK_SV: Final = (
    "Ombordsdatorn styr inte drönaren längre. Flygkontrollen avslutar uppdraget själv: den "
    "svävar en kort stund och flyger sedan rakt hem, även genom förbudszoner runt personer. "
    "Tryck STOPP eller ta över med RC om vägen hem inte är säker."
)


def fallback_alarm(seq: int, total: int) -> tuple[str, str, dict[str, Any]]:
    return alarm(
        str(AlertCode.MISSION_FALLBACK),
        "critical",
        MISSION_FALLBACK_SV,
        mission_seq=seq,
        mission_total=total,
        flying_home=seq >= total,
    )


def on_end_items(t: FcTelemetry, fc_mode: CopterMode | None) -> bool:
    """AUTO on the mission's hover-then-home items (the last item has ``seq == total``)."""
    seq, total = t.mission_seq_current, t.mission_total
    return (
        fc_mode is CopterMode.AUTO
        and seq is not None
        and total is not None
        and total > _END_ITEMS
        and seq > total - _END_ITEMS
    )


class MissionEndWatch:
    """Once per MAVLink tick: the alarm after ``FALLBACK_GRACE_MS`` on the end items."""

    def __init__(self, sink: EventSink) -> None:
        self._sink = sink
        self._since_ms: int | None = None
        self.alarm_active = False

    def reset(self) -> None:
        self._since_ms = None
        self.alarm_active = False

    def tick(
        self, t: FcTelemetry, fc_mode: CopterMode | None, now_ms: int, *, in_flight: bool
    ) -> None:
        seq, total = t.mission_seq_current, t.mission_total
        if not in_flight or seq is None or total is None or not on_end_items(t, fc_mode):
            self.reset()
            return
        if self._since_ms is None:
            self._since_ms = now_ms
        if self.alarm_active or now_ms - self._since_ms < FALLBACK_GRACE_MS:
            return
        self.alarm_active = True
        self._sink(*fallback_alarm(seq, total))
