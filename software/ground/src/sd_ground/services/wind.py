"""The ground station's "too windy" warning, computed from the flight controller's telemetry.

A warning only: nothing on board or on the ground acts on it, and STOP, FLY HOME and the
flight controller's own failsafes are unchanged. The panel shows it before takeoff and in
flight (``Telemetry.wind``), and the event log gets ``vehicle.wind_warning`` when it
starts. Three checks, each one a reason in ``WindCheck.reasons``:

* **Tilt margin** (``TILT_MARGIN``): a quadcopter leans into the wind to hold its place,
  so the tilt is a wind gauge that needs no calibration. The mean tilt over the last
  ``TILT_WINDOW_S`` in the air must stay ``TILT_MARGIN_DEG`` below the flight controller's
  lean limit (``ATC_ANGLE_MAX``).
* **Energy home** (``ENERGY_HOME``): flying home now in a straight line at ``RTL_SPEED_MS``
  at the mean current of the last ``TILT_WINDOW_S``, times ``ENERGY_MARGIN``, must leave
  the flight controller's low-battery reserve (``BATT_LOW_MAH``). The measured current
  already holds the wind's cost where the drone is; the higher airspeed against the wind
  on the way home is covered only by the margin, which is not yet calibrated.
* **Wind limit** (``WIND_LIMIT``): the flight controller's own wind estimate (MAVLink
  WIND) over the owner's limit for the current stage. ArduCopter only sends WIND when
  ``EK3_DRAG_*`` are set, which they are not until the drag is measured in flight, so
  this check is silent until then.

None of the thresholds is measured yet, so ``WindCheck.calibrated`` is false.
"""

from __future__ import annotations

import math
from collections import deque
from dataclasses import dataclass
from types import MappingProxyType
from typing import Final, Literal

from sd_mav.vehicle import FcTelemetry
from sd_safety.constants import FC_ANGLE_MAX_DEG, FC_SAFETY_EXPECTED
from sd_safety.targets import distance_m
from sd_schemas.api import WindCheck, WindReason

__all__ = [
    "ENERGY_MARGIN",
    "STAGES",
    "TILT_MARGIN_DEG",
    "TILT_WINDOW_S",
    "WIND_FRESH_S",
    "WIND_WINDOW_S",
    "WindStage",
    "WindWatch",
    "home_mah",
    "tilt_deg",
]

WindStage = Literal["first_flights", "until_calibrated"]
# The owner's wind limits on flight height, (mean, gusts) in m/s, raised only after the
# drag is measured. The first flights have no separate gust limit.
STAGES: Final = MappingProxyType(
    {"first_flights": (4.0, None), "until_calibrated": (8.0, 12.0)},
)
TILT_WINDOW_S: Final = 10.0
TILT_MARGIN_DEG: Final = 10.0
ENERGY_MARGIN: Final = 1.3
WIND_WINDOW_S: Final = 60.0
WIND_FRESH_S: Final = 3.0  # WIND is requested at 1 Hz
_FULL_WINDOW_S: Final = TILT_WINDOW_S - 1.0  # ATTITUDE at 4 Hz: allow one late sample
_IN_AIR: Final = "in_air"

RTL_SPEED_MS: Final = float(FC_SAFETY_EXPECTED["RTL_SPEED_MS"])
CAPACITY_MAH: Final = float(FC_SAFETY_EXPECTED["BATT_CAPACITY"])
RESERVE_MAH: Final = float(FC_SAFETY_EXPECTED["BATT_LOW_MAH"])


def tilt_deg(roll_deg: float, pitch_deg: float) -> float:
    """The angle between the drone's vertical axis and straight up."""
    c = math.cos(math.radians(roll_deg)) * math.cos(math.radians(pitch_deg))
    return math.degrees(math.acos(max(-1.0, min(1.0, c))))


def home_mah(distance_m: float, current_a: float, speed_ms: float = RTL_SPEED_MS) -> float:
    """mAh to fly ``distance_m`` home at ``speed_ms`` drawing ``current_a``, margin included."""
    return distance_m / speed_ms * current_a / 3.6 * ENERGY_MARGIN


@dataclass(frozen=True, slots=True)
class _Tilt:
    mono_ms: int
    tilt_deg: float
    current_a: float | None


@dataclass(frozen=True, slots=True)
class _Wind:
    mono_ms: int
    speed_ms: float
    from_deg: float


class WindWatch:
    """Keeps the last samples and computes ``WindCheck``. Not thread safe on its own."""

    def __init__(self, stage: WindStage = "first_flights") -> None:
        self.limit_ms, self.gust_limit_ms = STAGES[stage]
        self.tilt_warn_deg = FC_ANGLE_MAX_DEG - TILT_MARGIN_DEG
        self._tilt: deque[_Tilt] = deque()
        self._wind: deque[_Wind] = deque()

    def observe(self, t: FcTelemetry, now_mono_ms: int) -> None:
        """Take the newest ATTITUDE and WIND from ``t``; each sample is kept once."""
        if not (t.armed and t.landed_state == _IN_AIR):
            self._tilt.clear()  # the window starts again at the next takeoff
        elif t.attitude_mono_ms is not None and (
            not self._tilt or t.attitude_mono_ms > self._tilt[-1].mono_ms
        ):
            tilt = tilt_deg(t.roll_deg, t.pitch_deg)
            if math.isfinite(tilt):
                self._tilt.append(_Tilt(t.attitude_mono_ms, tilt, t.battery_current_a))
        if t.wind_mono_ms is not None and (
            not self._wind or t.wind_mono_ms > self._wind[-1].mono_ms
        ):
            self._wind.append(_Wind(t.wind_mono_ms, t.wind_speed_ms, t.wind_from_deg))
        while self._tilt and now_mono_ms - self._tilt[0].mono_ms > TILT_WINDOW_S * 1000:
            self._tilt.popleft()
        while self._wind and now_mono_ms - self._wind[0].mono_ms > WIND_WINDOW_S * 1000:
            self._wind.popleft()

    def check(self, t: FcTelemetry, now_mono_ms: int) -> WindCheck:
        reasons: list[WindReason] = []
        tilt_avg = None
        if self._tilt and self._tilt[-1].mono_ms - self._tilt[0].mono_ms >= _FULL_WINDOW_S * 1000:
            tilt_avg = sum(s.tilt_deg for s in self._tilt) / len(self._tilt)
            if tilt_avg > self.tilt_warn_deg:
                reasons.append("TILT_MARGIN")

        cost, left = None, None
        if t.battery_pct is not None:
            left = t.battery_pct / 100.0 * CAPACITY_MAH
        currents = [s.current_a for s in self._tilt if s.current_a is not None]
        way = None if t.home is None else distance_m(t.home.lat, t.home.lon, t.lat, t.lon)
        if currents and way is not None and math.isfinite(way):
            cost = home_mah(way, sum(currents) / len(currents))
            if left is not None and cost + RESERVE_MAH > left:
                reasons.append("ENERGY_HOME")

        mean = gust = from_deg = None
        if self._wind and now_mono_ms - self._wind[-1].mono_ms <= WIND_FRESH_S * 1000:
            mean = sum(s.speed_ms for s in self._wind) / len(self._wind)
            gust = max(s.speed_ms for s in self._wind)
            from_deg = self._wind[-1].from_deg
            over_gust = self.gust_limit_ms is not None and gust > self.gust_limit_ms
            if mean > self.limit_ms or over_gust:
                reasons.append("WIND_LIMIT")

        return WindCheck(
            warning=bool(reasons),
            reasons=reasons,
            tilt_avg_deg=tilt_avg,
            tilt_warn_deg=self.tilt_warn_deg,
            wind_ms=mean,
            wind_gust_ms=gust,
            wind_from_deg=from_deg,
            wind_limit_ms=self.limit_ms,
            wind_gust_limit_ms=self.gust_limit_ms,
            home_mah=cost,
            left_mah=left,
            reserve_mah=RESERVE_MAH,
        )
