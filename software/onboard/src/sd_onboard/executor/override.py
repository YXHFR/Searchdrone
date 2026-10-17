"""STOP and FLY HOME onboard, the stop latch and the pilot's takeover.

STOP and FLY HOME reach the flight controller as MAVLink straight from the ground station,
past the AI, the gate and this process. Onboard, the ``OperatorOverrideNotice`` that
comes with them makes the executor let go at once: the running skill is cancelled
without sending anything and the executor goes to HELD. Only an explicit
``OperatorRelease`` (the pilot's Resume click) hands control back; no chat text can.

``OVERRIDE_KINDS`` is the one table of what each kind does onboard. The fallback path
for when the ground's MAVLink does not reach the flight controller (sending the same
mode steps from onboard) is ``override_fallback.py``, the only user of the fallback
send origin.
"""

from __future__ import annotations

from dataclasses import dataclass
from types import MappingProxyType
from typing import Final

from sd_schemas.executor import HeldBy
from sd_schemas.link import LatchRef, ManualTakeover, OperatorOverrideNotice
from sd_schemas.mission import LatchAction

__all__ = ["OVERRIDE_KINDS", "KindRule", "OverrideState"]


@dataclass(frozen=True, slots=True)
class KindRule:
    held_by: HeldBy
    sets_stop_latch: bool
    battery_rtl: bool  # Resume is refused until LANDED
    steps: str  # the ground's mode chain: "stop" (BRAKE, LOITER, LAND) or "home" (SMART_RTL, ...)


OVERRIDE_KINDS: Final = MappingProxyType(
    {
        "stop": KindRule("operator_stop", True, False, "stop"),
        "return_home": KindRule("operator_home", False, False, "home"),
        "battery_rtl": KindRule("system_battery", False, True, "home"),
        "watchdog_stop": KindRule("system_watchdog", True, False, "stop"),
        "watchdog_rtl": KindRule("system_watchdog", False, False, "home"),
    }
)


class OverrideState:
    """The latches onboard and the newest action number seen from the ground."""

    def __init__(self) -> None:
        self.stop_latch = False
        self.battery_rtl_active = False
        self.last_action_seq = -1
        self.last_kind: LatchAction | None = None
        self.manual = False  # the pilot steers by hand (Ta över)

    def is_new(self, action_seq: int) -> bool:
        """A notice, release or takeover older than the newest one seen is ignored."""
        return action_seq > self.last_action_seq

    def apply_notice(self, notice: OperatorOverrideNotice) -> KindRule | None:
        """The rule for a new notice (and the latches set), or None for an old one."""
        if not self.is_new(notice.action_seq):
            return None
        self.last_action_seq = notice.action_seq
        rule = OVERRIDE_KINDS[notice.kind]
        self.last_kind = notice.kind
        if rule.sets_stop_latch:
            self.stop_latch = True
        if rule.battery_rtl:
            self.battery_rtl_active = True
        return rule

    def apply_takeover(self, takeover: ManualTakeover) -> bool:
        if not self.is_new(takeover.action_seq):
            return False
        self.last_action_seq = takeover.action_seq
        self.manual = takeover.active
        return True

    def accept_release_seq(self, action_seq: int) -> bool:
        """A release names the newest action; an older one is stale."""
        if action_seq < self.last_action_seq:
            return False
        self.last_action_seq = action_seq
        return True

    def released(self) -> None:
        self.stop_latch = False
        self.last_kind = None

    def landed(self) -> None:
        self.stop_latch = False
        self.battery_rtl_active = False
        self.last_kind = None
        self.manual = False

    def latch_ref(self) -> LatchRef | None:
        """What the Hello reports as the latch onboard."""
        if self.last_kind is None:
            return None
        return LatchRef(action=self.last_kind, action_seq=self.last_action_seq)

    def held_by_for_mode_change(self) -> HeldBy:
        """Who took over when a mode change arrives just after a notice."""
        if self.last_kind is None:
            return "external"
        return OVERRIDE_KINDS[self.last_kind].held_by
