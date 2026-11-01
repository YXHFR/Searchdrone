"""The link-loss rule (``LinkLossRule``): what the drone does when the ground goes quiet.

Pure like ``LimitMonitor``: no I/O, the clock comes in with the input. The executor calls
it every tick after ``LimitMonitor.check()`` and applies the actions it returns (the same
``LimitAction`` kinds, plus ``find`` and ``event``). Times count from the last inbound
SD-link message; the phases are ``sd_schemas.executor.LinkLossPhase``.

* **L-LINK** (no confirmed find nearby): the running step goes on (``GRACE``), hold at
  23 s (``HOLD``), home at 53 s (``HOME``).
* **L-FIND** (a confirmed find within 100 m, ``sd_safety.link_loss.choose_mode``): fly to
  the edge of its keep-out cylinder + 5 m (``FIND_MOVE``), hover there from at the latest
  15 s (``FIND_HOVER``), home at 123 s.
* **L-GCS**: the flight controller lost the ground station's MAVLink but stayed in LOITER
  (``FS_OPTIONS`` bit 4, the text "GCS Failsafe - Continuing Pilot Control"): home at once,
  as the flight controller itself would have done, except during the find hover.

Anyone else taking over (a person, the RC transmitter, a flight-controller failsafe,
another limit's return home) ends the rule's part (``YIELDED``) until the link is back.

The keep-out climb (``climbing_out``: straight up out of a person's cylinder) is not cut
short: L-LINK holds once it has ended (in the keep-out hold) and goes home from there when
that is due, and L-FIND's hover starts up there, with its clock, also past the find move's
deadline. A climb that has not ended by then does not hold home back for good: L-LINK goes
home regardless at ``HOME_CLIMBING_MS`` (53 s plus ``KEEPOUT_CLIMB_LINK_ALLOWANCE_S``, 113 s)
and L-FIND at its usual 123 s.
"""

from __future__ import annotations

import math
from collections.abc import Sequence
from dataclasses import dataclass
from typing import Final

from sd_onboard.executor.limits import LimitAction
from sd_onboard.executor.state import MOVING
from sd_safety.constants import GOTO_ARRIVE_RADIUS_M
from sd_safety.link_loss import (
    DETECT_MS,
    FIND_HOVER_UNTIL_MS,
    FIND_MOVE_DEADLINE_MS,
    HOLD_MS,
    HOME_CLIMBING_MS,
    HOME_MS,
    FindPlan,
    choose_mode,
)
from sd_schemas.charter import MissionCharter
from sd_schemas.executor import ExecutorState, LinkLossPhase
from sd_schemas.link import Keepout
from sd_schemas.vehicle import VehicleState

__all__ = ["INTERNAL_FIND_GOTO", "LinkLossInput", "LinkLossRule"]

P = LinkLossPhase
S = ExecutorState
INTERNAL_FIND_GOTO: Final = "internal_find_goto"
# The phases in which the rule steers (or will); anything else taking over yields.
_ACTIVE: Final = frozenset({P.GRACE, P.HOLD, P.FIND_MOVE, P.FIND_HOVER})
_FIND: Final = frozenset({P.FIND_MOVE, P.FIND_HOVER})


@dataclass(frozen=True, slots=True)
class LinkLossInput:
    now_ms: int
    state: ExecutorState
    vehicle: VehicleState
    charter: MissionCharter
    link_up: bool
    link_last_inbound_ms: int | None  # last message from the ground; None = never
    nav_ok: bool  # 3D fix, EKF fine, no limit monitor and no spoofing guard active
    keepouts: Sequence[Keepout]
    gcs_continue: bool  # "GCS Failsafe - Continuing Pilot Control" since the last tick
    active_skill: str | None
    # Why the executor is no longer steering, when that happened this tick or earlier:
    # "battery", "limit", "gnss", "human" or "fc_failsafe".
    taken_by: str | None = None
    # The keep-out climb is under way: it ends in the keep-out hold, which nothing cuts short.
    climbing_out: bool = False


class LinkLossRule:
    def __init__(self) -> None:
        self.phase = P.NONE
        self.plan: FindPlan | None = None
        self.hover_since_ms: int | None = None
        self.detected_ms: int | None = None
        self._down = False

    def reset(self) -> None:
        """After landing (LL-T15)."""
        self.__init__()

    @property
    def link_hold_active(self) -> bool:
        return self.phase == P.HOLD

    @property
    def find_active(self) -> bool:
        return self.phase in _FIND

    def summary(self) -> dict[str, object] | None:
        """For executor_state.json and the log; never read back to steer."""
        if self.phase == P.NONE:
            return None
        plan = self.plan
        return {
            "phase": str(self.phase),
            "keepout_id": str(plan.keepout_id) if plan else None,
            "hover_point": [plan.hover_lat, plan.hover_lon] if plan else None,
            "detected_mono_ms": self.detected_ms,
        }

    def check(self, x: LinkLossInput) -> list[LimitAction]:
        """This tick's actions, in order. Several phases may pass in one tick (a link that
        was never heard counts as lost since forever)."""
        if x.link_up:
            return self._link_back(x)
        self._down = True
        out: list[LimitAction] = []
        out += self._yield_if_taken(x)
        if x.gcs_continue:
            out += self._gcs(x)
        last = x.link_last_inbound_ms
        down_ms = math.inf if last is None else x.now_ms - last
        for _ in range(4):
            step = self._step(x, down_ms)
            if step is None:
                break
            out += step
        return out

    # --- the phases -------------------------------------------------------------------

    def _step(self, x: LinkLossInput, down_ms: float) -> list[LimitAction] | None:
        """One phase change, or None when the phase stays."""
        match self.phase:
            case P.NONE if down_ms >= DETECT_MS:
                self.detected_ms = x.now_ms
                if x.state not in MOVING:
                    self.phase = P.YIELDED  # someone else steers already
                    return []
                return self._detect(x)
            case P.GRACE if down_ms >= HOLD_MS and (
                not x.climbing_out or down_ms >= HOME_CLIMBING_MS
            ):
                self.phase = P.HOLD
                return [LimitAction("L-LINK", "hold", "link_loss", "link_loss_hold", {})]
            case P.HOLD if down_ms >= HOME_MS:
                self.phase = P.HOME
                return [LimitAction("L-LINK", "rtl", "link_loss", "link_loss_rtl", {})]
            case P.FIND_MOVE if x.state == S.HOLD and x.active_skill != INTERNAL_FIND_GOTO:
                # Arrived (T27), or another limit held it on the way (LL-T11).
                self._hover(x.now_ms)
                return []
            case P.FIND_MOVE if down_ms >= FIND_MOVE_DEADLINE_MS and (
                not x.climbing_out or down_ms >= FIND_HOVER_UNTIL_MS
            ):
                self._hover(x.now_ms)
                return [LimitAction("L-FIND", "hold", "find_hold", "find_move_deadline", {})]
            case P.FIND_HOVER if down_ms >= FIND_HOVER_UNTIL_MS:
                self.phase = P.HOME
                data = self._ended(x.now_ms, "timeout")
                return [LimitAction("L-FIND", "rtl", "find_hold", "find_hold_ended", data)]
            case _:
                return None

    def _detect(self, x: LinkLossInput) -> list[LimitAction]:
        v = x.vehicle
        plan = choose_mode(
            state=x.state,
            lat=v.lat,
            lon=v.lon,
            alt_rel_home_m=v.alt_rel_home_m,
            nav_ok=x.nav_ok,
            keepouts=x.keepouts,
            charter=x.charter,
        )
        if not isinstance(plan, FindPlan):
            self.phase = P.GRACE
            return [LimitAction("L-LINK", "event", "link_loss", "link_lost", {})]
        self.plan = plan
        self.phase = P.FIND_MOVE
        if plan.distance_m <= GOTO_ARRIVE_RADIUS_M:
            self._hover(x.now_ms)
        data: dict[str, float | str | None] = {
            "keepout_id": str(plan.keepout_id),
            "candidate_id": str(plan.candidate_id),
            "lat": plan.find_lat,
            "lon": plan.find_lon,
            "hover_lat": plan.hover_lat,
            "hover_lon": plan.hover_lon,
            "alt_m": round(plan.alt_m, 1),
            "yaw_deg": round(plan.yaw_deg, 1),
            "battery_pct": v.battery_pct,
        }
        return [LimitAction("L-FIND", "find", "find_hold", "find_hold_started", data)]

    def _hover(self, now_ms: int) -> None:
        self.phase = P.FIND_HOVER
        self.hover_since_ms = now_ms

    def _gcs(self, x: LinkLossInput) -> list[LimitAction]:
        if x.state not in MOVING:
            return []
        if self.phase in _FIND:
            return [LimitAction("L-GCS", "event", "find_hold", "gcs_lost_ignored_find", {})]
        if self.phase in (P.NONE, P.GRACE, P.HOLD):
            self.phase = P.HOME
            if self.detected_ms is None:
                self.detected_ms = x.now_ms
            return [LimitAction("L-GCS", "rtl", "link_loss", "gcs_lost_rtl", {})]
        return []

    def _yield_if_taken(self, x: LinkLossInput) -> list[LimitAction]:
        if self.phase not in _ACTIVE or x.state in MOVING:
            return []
        was_find = self.phase in _FIND
        self.phase = P.YIELDED
        if not was_find:
            return []
        reason = x.taken_by or ("human" if x.state == S.HELD else "fc_failsafe")
        data = self._ended(x.now_ms, reason)
        return [LimitAction("L-FIND", "event", "find_hold", "find_hold_ended", data)]

    def _link_back(self, x: LinkLossInput) -> list[LimitAction]:
        """LL-T14: the phase is NONE again; what the drone does stays as it is."""
        out: list[LimitAction] = []
        if self._down and self.phase in _FIND:
            data = self._ended(x.now_ms, "link_restored")
            out.append(LimitAction("L-FIND", "event", "find_hold", "find_hold_ended", data))
        if x.gcs_continue:
            out += self._gcs(x)  # LL12: only the ground's MAVLink is gone (SD-link up)
        self._down = False
        if self.phase != P.HOME or not x.gcs_continue:
            self.phase = P.NONE
            self.plan = None
            self.hover_since_ms = None
            self.detected_ms = None
        return out

    def _ended(self, now_ms: int, reason: str) -> dict[str, float | str | None]:
        since = self.hover_since_ms
        plan = self.plan
        return {
            "keepout_id": str(plan.keepout_id) if plan else None,
            "candidate_id": str(plan.candidate_id) if plan else None,
            "reason": reason,
            "hovered_s": round((now_ms - since) / 1000, 1) if since is not None else 0.0,
        }
