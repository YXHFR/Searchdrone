"""The ground station's "too windy" warning, computed from the flight controller's telemetry.

The panel shows it before takeoff and in flight (``Telemetry.wind``), and the event log
gets ``vehicle.wind_warning`` when it starts. ``TILT_MARGIN`` and ``WIND_LIMIT`` are
warnings only. ``ENERGY_HOME`` also starts the battery guard's FLY HOME
(``sd_ground.mav.thread.GroundMav``, ``battery_rtl``) in addition to its percentage
level, so the drone can turn home earlier than before but never later. STOP, FLY HOME
and the flight controller's own failsafes are unchanged. Three checks, each one a
reason in ``WindCheck.reasons``:

* **Tilt margin** (``TILT_MARGIN``): a quadcopter leans into the wind to hold its place,
  so the tilt is a wind gauge that needs no calibration. The mean tilt over the last
  ``TILT_WINDOW_S`` in the air must stay ``TILT_MARGIN_DEG`` below the flight controller's
  lean limit (``ATC_ANGLE_MAX``).
* **Energy home** (``ENERGY_HOME``): the energy to fly home now in a straight line must
  leave the flight controller's low-battery reserve (``BATT_LOW_MAH``). Up to three
  estimates, and the strictest one decides:

  - *as before*: ``RTL_SPEED_MS`` at the mean current of the last ``TILT_WINDOW_S``,
    times ``ENERGY_MARGIN``, with the constants in ``sd_safety.constants``, so the
    warning never comes later than it did before;
  - the same with the flight controller's own RTL speed, capacity and reserve;
  - *with the wind* (only with a fresh WIND): the ground speed home is the airspeed
    minus the headwind component, with the crosswind crabbed out
    (``home_ground_speed``); the current on the way is the hover current scaled to the
    lean that airspeed needs (``home_leg_mah``), times ``ENERGY_MARGIN``. When the wind
    is too strong to get home at all the check fails at once.

  The drag that turns airspeed into lean is measured while the drone holds position in
  a wind (``tan(tilt) = k * wind**2``). Until it is measured, the drone is assumed to fly
  home at ``RTL_SPEED_MS`` through the air at full lean (``ATC_ANGLE_MAX``): slow and
  expensive, so early.
* **Wind limit** (``WIND_LIMIT``): the flight controller's own wind estimate (MAVLink
  WIND) over the owner's limit for the current stage. ArduCopter only sends WIND when
  ``EK3_DRAG_*`` are set, which they are not until the drag is measured in flight, so
  this check, and the energy estimate with the wind, are silent until then.

The lean limit, the RTL speed and the battery numbers are the flight controller's own
(``WIND_PARAMS``, read on each connection, ``FcParams``); a parameter that could not be
read keeps the constant from ``sd_safety.constants`` and is logged
(``wind.param_fallback``). None of the thresholds is measured yet, so
``WindCheck.calibrated`` is false.
"""

from __future__ import annotations

import math
from collections import deque
from collections.abc import Mapping
from dataclasses import dataclass, field
from types import MappingProxyType
from typing import Final, Literal

from sd_ground.logging import get_logger
from sd_mav.vehicle import FcTelemetry
from sd_safety.constants import FC_ANGLE_MAX_DEG, FC_SAFETY_EXPECTED
from sd_safety.targets import Plane, distance_m
from sd_schemas.api import WindCheck, WindReason

__all__ = [
    "ENERGY_MARGIN",
    "STAGES",
    "TILT_MARGIN_DEG",
    "TILT_WINDOW_S",
    "WIND_FRESH_S",
    "WIND_PARAMS",
    "WIND_WINDOW_S",
    "FcParams",
    "WindStage",
    "WindWatch",
    "fc_params",
    "home_ground_speed",
    "home_leg_mah",
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

# The drag is measured only while the drone holds position (every sample of a full tilt
# window slower than this over the ground) in at least this much wind and lean.
HOLD_GROUNDSPEED_MS: Final = 1.0
DRAG_MIN_WIND_MS: Final = 3.0
DRAG_MIN_TILT_DEG: Final = 2.0
# Slower than this over the ground on the way home counts as not getting home.
MIN_HOME_GROUND_MS: Final = 1.0
_NEAR_HOME_M: Final = 1.0  # closer than this the way home has no direction: all headwind

RTL_SPEED_MS: Final = float(FC_SAFETY_EXPECTED["RTL_SPEED_MS"])
CAPACITY_MAH: Final = float(FC_SAFETY_EXPECTED["BATT_CAPACITY"])
RESERVE_MAH: Final = float(FC_SAFETY_EXPECTED["BATT_LOW_MAH"])

# Read from the flight controller on each connection. ``WP_SPD`` is what RTL flies at
# when ``RTL_SPEED_MS`` is 0 (Copter 4.7 names; WPNAV_SPEED before).
WIND_PARAMS: Final = ("ATC_ANGLE_MAX", "RTL_SPEED_MS", "WP_SPD", "BATT_CAPACITY", "BATT_LOW_MAH")
_ANGLE_RANGE_DEG: Final = (10.0, 80.0)  # ATC_ANGLE_MAX's range in ArduCopter

log = get_logger("sd_ground.wind")


@dataclass(frozen=True, slots=True)
class FcParams:
    """The flight controller's numbers the checks use. ``fallback`` names each one that
    could not be read and is the constant from ``sd_safety.constants`` instead."""

    angle_max_deg: float = FC_ANGLE_MAX_DEG
    rtl_speed_ms: float = RTL_SPEED_MS
    capacity_mah: float = CAPACITY_MAH
    reserve_mah: float = RESERVE_MAH
    fallback: tuple[str, ...] = field(
        default=("ATC_ANGLE_MAX", "RTL_SPEED_MS", "BATT_CAPACITY", "BATT_LOW_MAH")
    )


def _usable(v: float | None, low: float = 0.0, high: float = math.inf) -> float | None:
    """``v`` when the flight controller answered with a value in (low, high]."""
    return v if v is not None and math.isfinite(v) and low < v <= high else None


def fc_params(values: Mapping[str, float | None]) -> FcParams:
    """``FcParams`` from what the flight controller answered (``mav.fence.read_params``).
    A parameter without a usable answer keeps its constant, with ``wind.param_fallback``
    in the log: ``BATT_LOW_MAH`` 0 (no mAh failsafe) or an ``ATC_ANGLE_MAX`` outside
    ArduCopter's range count as unusable too."""
    defaults = FcParams()
    lo, hi = _ANGLE_RANGE_DEG
    angle = _usable(values.get("ATC_ANGLE_MAX"), lo - 1e-6, hi)
    rtl = _usable(values.get("RTL_SPEED_MS"))
    if rtl is None and values.get("RTL_SPEED_MS") == 0.0:
        rtl = _usable(values.get("WP_SPD"))  # RTL_SPEED_MS 0: RTL flies at WP_SPD
    read = {
        "ATC_ANGLE_MAX": (angle, defaults.angle_max_deg),
        "RTL_SPEED_MS": (rtl, defaults.rtl_speed_ms),
        "BATT_CAPACITY": (_usable(values.get("BATT_CAPACITY")), defaults.capacity_mah),
        "BATT_LOW_MAH": (_usable(values.get("BATT_LOW_MAH")), defaults.reserve_mah),
    }
    fallback = tuple(name for name, (v, _) in read.items() if v is None)
    for name in fallback:
        log.warning(
            "wind.param_fallback",
            param=name,
            read=values.get(name),
            used=read[name][1],
            detail="the flight controller's value could not be read; the constant is used",
        )
    used = {name: (d if v is None else v) for name, (v, d) in read.items()}
    log.info("wind.params", **used, fallback=list(fallback))
    return FcParams(
        angle_max_deg=used["ATC_ANGLE_MAX"],
        rtl_speed_ms=used["RTL_SPEED_MS"],
        capacity_mah=used["BATT_CAPACITY"],
        reserve_mah=used["BATT_LOW_MAH"],
        fallback=fallback,
    )


def tilt_deg(roll_deg: float, pitch_deg: float) -> float:
    """The angle between the drone's vertical axis and straight up."""
    c = math.cos(math.radians(roll_deg)) * math.cos(math.radians(pitch_deg))
    return math.degrees(math.acos(max(-1.0, min(1.0, c))))


def home_mah(distance_m: float, current_a: float, speed_ms: float = RTL_SPEED_MS) -> float:
    """mAh to fly ``distance_m`` home at ``speed_ms`` drawing ``current_a``, margin
    included: the estimate without the wind."""
    return distance_m / speed_ms * current_a / 3.6 * ENERGY_MARGIN


def home_ground_speed(
    rtl_speed_ms: float, max_airspeed_ms: float, headwind_ms: float, crosswind_ms: float
) -> float | None:
    """Ground speed on the way home, or None when the drone does not get home.

    RTL holds ``rtl_speed_ms`` over the ground while the airspeed that needs,
    ``hypot(rtl + headwind, crosswind)``, stays within ``max_airspeed_ms``. Beyond that it
    flies at ``max_airspeed_ms``, crabs out the crosswind and makes good what is left after
    the headwind: ``sqrt(max_air**2 - cross**2) - headwind``. A tailwind never makes it
    faster than ``rtl_speed_ms``."""
    if math.hypot(rtl_speed_ms + headwind_ms, crosswind_ms) <= max_airspeed_ms:
        return rtl_speed_ms
    if crosswind_ms >= max_airspeed_ms:
        return None
    ground = math.sqrt(max_airspeed_ms**2 - crosswind_ms**2) - headwind_ms
    ground = min(ground, rtl_speed_ms)
    return ground if ground >= MIN_HOME_GROUND_MS else None


def home_leg_mah(
    distance_m: float,
    *,
    current_a: float,
    tilt_now_deg: float,
    headwind_ms: float,
    crosswind_ms: float,
    params: FcParams,
    drag_k: float | None,
) -> float | None:
    """mAh to fly ``distance_m`` home against the wind, margin included; None when the
    drone does not get home.

    ``drag_k`` (``tan(tilt) = k * airspeed**2``) gives the highest airspeed (at
    ``ATC_ANGLE_MAX``) and the lean on the way. The hover current is the measured current
    taken back to level flight, and the current on the way that scaled to the lean:
    the thrust grows with ``1 / cos(tilt)`` and the induced power with the thrust to the
    power 1.5 (momentum theory, without translational lift, so on the safe side). Without
    ``drag_k`` the airspeed is ``RTL_SPEED_MS`` at full lean."""
    max_tan = math.tan(math.radians(params.angle_max_deg))
    max_air = params.rtl_speed_ms if drag_k is None else math.sqrt(max_tan / drag_k)
    ground = home_ground_speed(params.rtl_speed_ms, max_air, headwind_ms, crosswind_ms)
    if ground is None:
        return None
    airspeed = math.hypot(ground + headwind_ms, crosswind_ms)
    lean_tan = max_tan if drag_k is None else min(max_tan, drag_k * airspeed**2)
    hover_a = current_a * math.cos(math.radians(tilt_now_deg)) ** 1.5
    leg_a = hover_a / math.cos(math.atan(lean_tan)) ** 1.5
    return distance_m / ground * leg_a / 3.6 * ENERGY_MARGIN


@dataclass(frozen=True, slots=True)
class _Tilt:
    mono_ms: int
    tilt_deg: float
    current_a: float | None
    groundspeed_ms: float


@dataclass(frozen=True, slots=True)
class _Wind:
    mono_ms: int
    speed_ms: float
    from_deg: float


@dataclass(frozen=True, slots=True)
class _Energy:
    cost_mah: float | None  # None: does not get home
    left_mah: float | None
    reserve_mah: float

    @property
    def short(self) -> bool:
        return self.cost_mah is None or (
            self.left_mah is not None and self.cost_mah + self.reserve_mah > self.left_mah
        )

    @property
    def spare_mah(self) -> float:
        """What is left above the reserve after the way home (-inf: does not get home)."""
        if self.cost_mah is None:
            return -math.inf
        if self.left_mah is None:
            return math.inf
        return self.left_mah - self.cost_mah - self.reserve_mah


def _wind_on_the_way(
    home: Plane, lat: float, lon: float, speed_ms: float, from_deg: float
) -> tuple[float, float]:
    """(headwind, crosswind) in m/s on the straight line from the drone to home."""
    east, north = home.xy(lat, lon)
    way = math.hypot(east, north)
    if not math.isfinite(way) or way < _NEAR_HOME_M:
        return speed_ms, 0.0
    home_e, home_n = -east / way, -north / way
    from_e, from_n = math.sin(math.radians(from_deg)), math.cos(math.radians(from_deg))
    # The wind comes from where the drone flies to: a headwind.
    head = speed_ms * (home_e * from_e + home_n * from_n)
    cross = speed_ms * abs(home_e * from_n - home_n * from_e)
    return head, cross


class WindWatch:
    """Keeps the last samples and computes ``WindCheck``. Not thread safe on its own."""

    def __init__(self, stage: WindStage = "first_flights", params: FcParams | None = None) -> None:
        self.limit_ms, self.gust_limit_ms = STAGES[stage]
        self.params = FcParams() if params is None else params
        self._tilt: deque[_Tilt] = deque()
        self._wind: deque[_Wind] = deque()
        self.drag_k: float | None = None  # tan(tilt) / airspeed**2, measured this flight

    @property
    def tilt_warn_deg(self) -> float:
        return self.params.angle_max_deg - TILT_MARGIN_DEG

    def set_params(self, params: FcParams) -> None:
        """The flight controller's values (read on each connection)."""
        self.params = params

    def observe(self, t: FcTelemetry, now_mono_ms: int) -> None:
        """Take the newest ATTITUDE and WIND from ``t``; each sample is kept once."""
        if not (t.armed and t.landed_state == _IN_AIR):
            self._tilt.clear()  # the window starts again at the next takeoff
            self.drag_k = None
        elif t.attitude_mono_ms is not None and (
            not self._tilt or t.attitude_mono_ms > self._tilt[-1].mono_ms
        ):
            tilt = tilt_deg(t.roll_deg, t.pitch_deg)
            if math.isfinite(tilt):
                self._tilt.append(
                    _Tilt(t.attitude_mono_ms, tilt, t.battery_current_a, t.groundspeed_ms)
                )
        if t.wind_mono_ms is not None and (
            not self._wind or t.wind_mono_ms > self._wind[-1].mono_ms
        ):
            self._wind.append(_Wind(t.wind_mono_ms, t.wind_speed_ms, t.wind_from_deg))
        while self._tilt and now_mono_ms - self._tilt[0].mono_ms > TILT_WINDOW_S * 1000:
            self._tilt.popleft()
        while self._wind and now_mono_ms - self._wind[0].mono_ms > WIND_WINDOW_S * 1000:
            self._wind.popleft()

    def _full_tilt_window(self) -> bool:
        return bool(self._tilt) and (
            self._tilt[-1].mono_ms - self._tilt[0].mono_ms >= _FULL_WINDOW_S * 1000
        )

    def _measure_drag(self, tilt_avg: float | None, wind_ms: float | None) -> None:
        """While the drone holds position the lean is the wind's drag alone. The largest
        drag seen this flight is kept: more drag means a slower, steeper way home."""
        if tilt_avg is None or wind_ms is None:
            return
        if wind_ms < DRAG_MIN_WIND_MS or tilt_avg < DRAG_MIN_TILT_DEG:
            return
        if any(s.groundspeed_ms >= HOLD_GROUNDSPEED_MS for s in self._tilt):
            return
        k = math.tan(math.radians(tilt_avg)) / wind_ms**2
        if math.isfinite(k) and (self.drag_k is None or k > self.drag_k):
            self.drag_k = k

    def _energy(
        self, t: FcTelemetry, tilt_now: float, wind: tuple[float, float] | None
    ) -> _Energy | None:
        """The strictest of: the estimate as before (the constants), the same with the
        flight controller's own numbers, and (with a fresh WIND) the one with the wind."""
        currents = [s.current_a for s in self._tilt if s.current_a is not None]
        if not currents or t.home is None:
            return None
        way = distance_m(t.home.lat, t.home.lon, t.lat, t.lon)
        if not math.isfinite(way):
            return None
        current = sum(currents) / len(currents)
        pct = None if t.battery_pct is None else t.battery_pct / 100.0
        p = self.params
        left = None if pct is None else pct * p.capacity_mah
        estimates = [
            _Energy(
                home_mah(way, current), None if pct is None else pct * CAPACITY_MAH, RESERVE_MAH
            ),
            _Energy(home_mah(way, current, p.rtl_speed_ms), left, p.reserve_mah),
        ]
        if wind is not None:
            head, cross = _wind_on_the_way(Plane(t.home.lat, t.home.lon), t.lat, t.lon, *wind)
            cost = home_leg_mah(
                way,
                current_a=current,
                tilt_now_deg=tilt_now,
                headwind_ms=head,
                crosswind_ms=cross,
                params=p,
                drag_k=self.drag_k,
            )
            estimates.append(_Energy(cost, left, p.reserve_mah))
        return min(estimates, key=lambda e: e.spare_mah)

    def check(self, t: FcTelemetry, now_mono_ms: int) -> WindCheck:
        reasons: list[WindReason] = []
        tilt_avg = None
        if self._full_tilt_window():
            tilt_avg = sum(s.tilt_deg for s in self._tilt) / len(self._tilt)
            if tilt_avg > self.tilt_warn_deg:
                reasons.append("TILT_MARGIN")

        mean = gust = from_deg = None
        if self._wind and now_mono_ms - self._wind[-1].mono_ms <= WIND_FRESH_S * 1000:
            mean = sum(s.speed_ms for s in self._wind) / len(self._wind)
            gust = max(s.speed_ms for s in self._wind)
            from_deg = self._wind[-1].from_deg
        self._measure_drag(tilt_avg, mean)

        wind = None
        if mean is not None and from_deg is not None:
            wind = (max(mean, self._wind[-1].speed_ms), from_deg)
        # The lean the measured current belongs to (the same samples, full window or not).
        tilt_now = sum(s.tilt_deg for s in self._tilt) / len(self._tilt) if self._tilt else 0.0
        energy = self._energy(t, tilt_now, wind)
        if energy is not None and energy.short:
            reasons.append("ENERGY_HOME")

        if mean is not None and gust is not None:
            over_gust = self.gust_limit_ms is not None and gust > self.gust_limit_ms
            if mean > self.limit_ms or over_gust:
                reasons.append("WIND_LIMIT")

        left = None
        if t.battery_pct is not None:
            left = t.battery_pct / 100.0 * self.params.capacity_mah
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
            home_mah=None if energy is None else energy.cost_mah,
            left_mah=left if energy is None else energy.left_mah,
            reserve_mah=self.params.reserve_mah if energy is None else energy.reserve_mah,
        )
