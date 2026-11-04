"""The alarm for a search mission's built-in end flying on its own.

Every search mission ends with a hover at the last point that has no end
(``sd_mav.mission.survey_mission``): the mission never flies home, so nothing flies
straight through the keep-out cylinders around persons without someone deciding it. When
all goes well the flight controller never gets there: the onboard computer switches to
LOITER at the last search point. If the flight controller stays in AUTO on that last
item, the onboard computer has stopped steering, and the drone hovers there until the
operator presses FLY HOME or takes over, or a failsafe brings it home (the flight
controller's own for a low battery, a lost link or the fence, or the ground's battery and
energy watch). FLY HOME tries SMART_RTL first, but plain RTL flies straight, through the
keep-out cylinders too; the alarm text says so.

``MissionEndWatch`` follows ``MISSION_CURRENT`` on the ground's own MAVLink channel, so it
works whatever state the onboard computer is in. It only raises the alarm (once per
episode); it never sends anything to the flight controller. When the onboard computer
falls silent mid-search, the companion watchdog stops the search at once and raises the
same alarm (``fallback_alarm``).
"""

from __future__ import annotations

from typing import Any, Final

from sd_ground.mav.gnss_watch import EventSink, alarm
from sd_mav.modes import CopterMode
from sd_mav.vehicle import FcTelemetry
from sd_schemas.api import AlertCode

__all__ = ["FALLBACK_GRACE_MS", "MISSION_FALLBACK_SV", "MissionEndWatch", "fallback_alarm"]

# The onboard computer switches to LOITER within a second or two of the last search point;
# this long on the end item in AUTO means it did not.
FALLBACK_GRACE_MS: Final = 10_000

MISSION_FALLBACK_SV: Final = (
    "Ombordsdatorn styr inte drönaren längre. Sökningen går inte vidare och drönaren svävar "
    "där den är. Den flyger hem när du trycker FLYG HEM eller när ett skydd slår till (till "
    "exempel lågt batteri eller förlorad länk). FLYG HEM kan flyga rakt genom förbudszoner "
    "runt personer. Ta över med RC om du vill styra själv."
)


def fallback_alarm(seq: int | None, total: int | None) -> tuple[str, str, dict[str, Any]]:
    """Here, and from the companion watchdog when it stops a search mid-way in AUTO."""
    return alarm(
        str(AlertCode.MISSION_FALLBACK),
        "critical",
        MISSION_FALLBACK_SV,
        mission_seq=seq,
        mission_total=total,
    )


def on_end_item(t: FcTelemetry, fc_mode: CopterMode | None) -> bool:
    """AUTO on the mission's hover at the end (the last item, ``seq == total``)."""
    seq, total = t.mission_seq_current, t.mission_total
    return fc_mode is CopterMode.AUTO and seq is not None and total is not None and seq >= total > 1


class MissionEndWatch:
    """Once per MAVLink tick: the alarm after ``FALLBACK_GRACE_MS`` on the end item."""

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
        if not in_flight or seq is None or total is None or not on_end_item(t, fc_mode):
            self.reset()
            return
        if self._since_ms is None:
            self._since_ms = now_ms
        if self.alarm_active or now_ms - self._since_ms < FALLBACK_GRACE_MS:
            return
        self.alarm_active = True
        self._sink(*fallback_alarm(seq, total))
