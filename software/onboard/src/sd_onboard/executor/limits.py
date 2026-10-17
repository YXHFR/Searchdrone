"""The limit monitors (``LimitMonitor``): the last line of defence in the air.

They run every tick while the executor steers (TAKEOFF, TRANSIT, SURVEY, HOLD), and work
even without a link to the ground. Each monitor acts on a rising edge only: when its
condition (with its own time, e.g. "for 2 s") becomes true while it is not active. It
resets after the condition has been false for ``LIMIT_CLEAR_S``. While active it fires no
new action or event; the order refusals it implies last as long as it is active.

Order when several fire in the same tick: L-BATT, L-RANGE2, L-AREA, L-GPS, L-BATT0, L-ALT,
L-RANGE, L-SPEED, L-LINK, L-AI. The first return-home wins; a later hold is then ignored.
(L-FC, the silent flight controller, is handled by the executor in every steered state.)
"""

from __future__ import annotations

import math
from collections import deque
from dataclasses import dataclass, field
from typing import Final, Literal

from sd_safety.constants import (
    ALT_OVERSHOOT_M,
    BATTERY_UNKNOWN_RTL_S,
    BATTERY_UNKNOWN_S,
    LIMIT_CLEAR_S,
    LINK_LOSS_DETECT_S,
    PILOT_POS_MAX_JUMP_MS,
    PILOT_POS_STALE_S,
    RANGE_HARD_MARGIN_M,
    SDLINK_GRACE_S,
    SDLINK_HOLD_S,
    SPEED_OVERSHOOT_MS,
)
from sd_safety.targets import Plane, distance_m, polygon_from_geojson
from sd_schemas.base import LatLon
from sd_schemas.charter import MissionCharter
from sd_schemas.executor import ExecutorState
from sd_schemas.vehicle import VehicleState

__all__ = [
    "INTERNAL_ALT_GOTO",
    "LimitAction",
    "LimitInput",
    "LimitMonitor",
    "PilotTracker",
    "inside_charter",
]

S = ExecutorState
INTERNAL_ALT_GOTO: Final = "internal_alt_goto"
_AREA_FOR_S: Final = 2.0
_GPS_FOR_S: Final = 2.0
_GPS_GOOD_FOR_S: Final = 5.0
_ALT_FOR_S: Final = 1.0
_SPEED_FOR_S: Final = 3.0
_RANGE_TREND_S: Final = 2.0
_AI_STATES: Final = frozenset({S.TRANSIT, S.SURVEY, S.HOLD})
_AI_HOLD_FROM: Final = frozenset({S.TRANSIT, S.SURVEY})

# "after_step": nothing now; the running step finishes, then the vehicle holds.
Kind = Literal["hold", "rtl", "alt", "after_step"]


@dataclass(frozen=True, slots=True)
class LimitAction:
    limit: str  # "L-BATT", ...
    kind: Kind
    hold_reason: str  # for a hold: "limit", "link_loss" or "ai_silent"
    event: str  # OnboardEvent type without the "executor." prefix
    data: dict[str, float | str | None] = field(default_factory=dict)


@dataclass(frozen=True, slots=True)
class LimitInput:
    now_ms: int
    state: ExecutorState
    vehicle: VehicleState
    charter: MissionCharter
    pilot: LatLon  # the effective pilot position
    active_skill: str | None
    inbound_order_active: bool  # an accepted order towards the pilot is running (L-RANGE)
    link_last_inbound_ms: int | None  # last message from the ground; None = never
    link_up: bool
    commander_connected: bool | None  # the last CommanderStatus; None = none yet
    commander_status_ms: int | None  # when the last CommanderStatus arrived


def inside_charter(lat: float, lon: float, charter: MissionCharter) -> bool:
    """The point lies in the charter's area grown by its fence buffer (the gate's geometry)."""
    plane = Plane(charter.home.lat, charter.home.lon)
    area = polygon_from_geojson(charter.area, plane)
    xy = plane.xy(lat, lon)
    return area.contains(xy) or area.boundary_distance(xy) <= charter.geofence_buffer_m


class _Edge:
    """One monitor's latch: fires once when its condition has held long enough."""

    def __init__(self, hold_for_s: float = 0.0) -> None:
        self.hold_for_ms = int(hold_for_s * 1000)
        self.true_since: int | None = None
        self.false_since: int | None = None
        self.active = False

    def update(self, condition: bool, now_ms: int) -> bool:
        """True exactly on the rising edge."""
        if condition:
            self.false_since = None
            if self.true_since is None:
                self.true_since = now_ms
            if not self.active and now_ms - self.true_since >= self.hold_for_ms:
                self.active = True
                return True
            return False
        self.true_since = None
        if self.active:
            if self.false_since is None:
                self.false_since = now_ms
            if now_ms - self.false_since >= LIMIT_CLEAR_S * 1000:
                self.active = False
                self.false_since = None
        return False

    def reset(self) -> None:
        self.true_since = self.false_since = None
        self.active = False


class PilotTracker:
    """The effective pilot position: a fresh PilotPosition, else the charter's."""

    def __init__(self) -> None:
        self.position: LatLon | None = None
        self.received_ms: int | None = None
        self.stale_reported = False

    def update(self, lat: float, lon: float, now_ms: int) -> bool:
        """Take a new position; False (ignored) when it jumps faster than 50 m/s."""
        if self.position is not None and self.received_ms is not None:
            dt = max(1e-3, (now_ms - self.received_ms) / 1000)
            jump = distance_m(self.position.lat, self.position.lon, lat, lon)
            if jump / dt > PILOT_POS_MAX_JUMP_MS:
                return False
        self.position = LatLon(lat=lat, lon=lon)
        self.received_ms = now_ms
        self.stale_reported = False
        return True

    def effective(self, charter: MissionCharter, now_ms: int) -> tuple[LatLon, bool]:
        """(position, became_stale_now)."""
        fresh = (
            self.received_ms is not None and now_ms - self.received_ms <= PILOT_POS_STALE_S * 1000
        )
        if fresh and self.position is not None:
            return self.position, False
        became = self.position is not None and not self.stale_reported
        if became:
            self.stale_reported = True
        return charter.pilot_position, became


class LimitMonitor:
    def __init__(self) -> None:
        self.batt = _Edge()
        self.batt0 = _Edge(BATTERY_UNKNOWN_S)
        self.batt0_rtl = _Edge(BATTERY_UNKNOWN_S + BATTERY_UNKNOWN_RTL_S)
        self.range2 = _Edge()
        self.area = _Edge(_AREA_FOR_S)
        self.gps = _Edge(_GPS_FOR_S)
        self.alt = _Edge(_ALT_FOR_S)
        self.range = _Edge()
        self.speed = _Edge(_SPEED_FOR_S)
        self.link_hold = _Edge()
        self.link_rtl = _Edge()
        self.ai = _Edge()
        self._gps_good_since: int | None = None
        self._range_track: deque[tuple[int, float]] = deque()
        self._range_rehold = False
        self._ai_timer_from_ms: int | None = None
        self._link_was_down = False

    # --- what the executor asks -------------------------------------------------------

    @property
    def battery_rtl_active(self) -> bool:
        return self.batt.active

    @property
    def range_active(self) -> bool:
        return self.range.active

    def nav_degraded(self, now_ms: int) -> bool:
        if self.gps.active:
            return True
        return self._gps_good_since is not None and now_ms - self._gps_good_since < (
            _GPS_GOOD_FOR_S * 1000
        )

    def active(self) -> list[str]:
        names = {
            "L-BATT": self.batt,
            "L-BATT0": self.batt0,
            "L-RANGE2": self.range2,
            "L-AREA": self.area,
            "L-GPS": self.gps,
            "L-ALT": self.alt,
            "L-RANGE": self.range,
            "L-SPEED": self.speed,
            "L-LINK": self.link_hold,
            "L-AI": self.ai,
        }
        return [name for name, edge in names.items() if edge.active]

    def reset(self) -> None:
        """After landing: a new flight starts with every monitor quiet."""
        self.__init__()

    # --- one tick ---------------------------------------------------------------------

    def check(self, x: LimitInput) -> list[LimitAction]:
        """The actions of this tick, in priority order (only rising edges)."""
        now, v, c = x.now_ms, x.vehicle, x.charter
        out: list[LimitAction] = []
        pct = v.battery_pct
        if self.batt.update(pct is not None and pct <= c.battery_rtl_pct, now):
            out.append(
                LimitAction("L-BATT", "rtl", "limit", "limit_battery_rtl", {"battery_pct": pct})
            )
        dist = distance_m(x.pilot.lat, x.pilot.lon, v.lat, v.lon)
        if not math.isfinite(dist):
            dist = math.inf
        hard = c.max_range_from_pilot_m + RANGE_HARD_MARGIN_M
        if self.range2.update(dist > hard, now):
            out.append(
                LimitAction("L-RANGE2", "rtl", "limit", "limit_range_rtl", {"distance_m": _r(dist)})
            )
        if self.area.update(not inside_charter(v.lat, v.lon, c), now):
            out.append(LimitAction("L-AREA", "rtl", "limit", "limit_area", {}))
        bad_nav = v.gps_fix_type < 3 or not v.ekf_ok
        was_gps = self.gps.active
        if self.gps.update(bad_nav, now):
            out.append(LimitAction("L-GPS", "hold", "limit", "limit_nav", {"fix": v.gps_fix_type}))
        if was_gps and not self.gps.active:
            self._gps_good_since = now
        if bad_nav:
            self._gps_good_since = None
        if self.batt0.update(pct is None, now):
            out.append(LimitAction("L-BATT0", "hold", "limit", "limit_battery_unknown", {}))
        if self.batt0_rtl.update(pct is None, now):
            out.append(LimitAction("L-BATT0", "rtl", "limit", "limit_battery_unknown_rtl", {}))
        too_high = v.alt_rel_home_m > c.alt_max_agl_m + ALT_OVERSHOOT_M
        if self.alt.update(too_high and x.active_skill != INTERNAL_ALT_GOTO, now):
            out.append(
                LimitAction(
                    "L-ALT", "alt", "limit", "limit_altitude", {"alt_m": _r(v.alt_rel_home_m)}
                )
            )
        out += self._range(x, dist, now)
        fast = v.groundspeed_ms > c.max_speed_ms + SPEED_OVERSHOOT_MS
        if self.speed.update(fast, now):
            out.append(
                LimitAction(
                    "L-SPEED", "hold", "limit", "limit_speed", {"speed_ms": _r(v.groundspeed_ms)}
                )
            )
        out += self._link(x, now)
        out += self._ai(x, now)
        return _first_rtl_wins(out)

    def _range(self, x: LimitInput, dist: float, now: int) -> list[LimitAction]:
        outside = dist > x.charter.max_range_from_pilot_m
        self._range_track.append((now, dist))
        while self._range_track and now - self._range_track[0][0] > _RANGE_TREND_S * 1000:
            self._range_track.popleft()
        if not outside:
            self.range.update(False, now)
            return []
        oldest_ms, oldest = self._range_track[0]
        spanned = now - oldest_ms >= _RANGE_TREND_S * 1000 - 150
        growing = spanned and dist > oldest
        if not self.range.active:
            self.range.update(True, now)
            return [
                LimitAction("L-RANGE", "hold", "limit", "limit_range", {"distance_m": _r(dist)})
            ]
        # Active: an inbound order may bring it back; it holds again (once per order) if
        # the distance grows for 2 s.
        if not x.inbound_order_active:
            self._range_rehold = False
            return []
        if growing and not self._range_rehold:
            self._range_rehold = True
            return [
                LimitAction("L-RANGE", "hold", "limit", "limit_range", {"distance_m": _r(dist)})
            ]
        return []

    def _link(self, x: LimitInput, now: int) -> list[LimitAction]:
        if x.link_up:
            if self._link_was_down:
                self.link_hold.reset()
                self.link_rtl.reset()
            self._link_was_down = False
            return []
        self._link_was_down = True
        last = x.link_last_inbound_ms
        down_ms = math.inf if last is None else now - last
        out: list[LimitAction] = []
        grace = (LINK_LOSS_DETECT_S + SDLINK_GRACE_S) * 1000
        if not self.link_hold.active and down_ms >= grace:
            self.link_hold.active = True
            out.append(LimitAction("L-LINK", "hold", "link_loss", "link_lost", {}))
        if not self.link_rtl.active and down_ms >= grace + SDLINK_HOLD_S * 1000:
            self.link_rtl.active = True
            out.append(LimitAction("L-LINK", "rtl", "link_loss", "link_loss_rtl", {}))
        return out

    def _ai(self, x: LimitInput, now: int) -> list[LimitAction]:
        """L-AI, only while SD-link is up; the timer starts again when the link returns."""
        if not x.link_up:
            self._ai_timer_from_ms = None
            self.ai.update(False, now)
            return []
        if self._ai_timer_from_ms is None:
            self._ai_timer_from_ms = now
        heard = x.commander_status_ms
        since = max(self._ai_timer_from_ms, heard) if heard is not None else self._ai_timer_from_ms
        silent_for = now - since
        timeout = x.charter.ai_silent_timeout_s * 1000
        silent = x.commander_connected is False or silent_for >= timeout
        if x.state not in _AI_STATES:
            return []
        if not self.ai.update(silent, now):
            return []
        configured = x.charter.on_ai_silent
        if configured == "hold" and x.state in _AI_HOLD_FROM:
            return [LimitAction("L-AI", "hold", "ai_silent", "ai_silent", {"on_ai_silent": "hold"})]
        # continue_plan, and return_home too (the owner's rule: the AI going quiet finishes
        # the current step and then holds; it never sends the vehicle home by itself).
        data: dict[str, float | str | None] = {
            "on_ai_silent": configured,
            "applied": "continue_plan",
        }
        return [LimitAction("L-AI", "after_step", "ai_silent", "ai_silent", data)]

    @property
    def ai_silent(self) -> bool:
        return self.ai.active


def _first_rtl_wins(actions: list[LimitAction]) -> list[LimitAction]:
    """Everything up to the first return home; what comes after it is dropped."""
    first_rtl = next((i for i, a in enumerate(actions) if a.kind == "rtl"), None)
    return actions if first_rtl is None else actions[: first_rtl + 1]


def _r(value: float) -> float | None:
    return round(value, 1) if math.isfinite(value) else None
