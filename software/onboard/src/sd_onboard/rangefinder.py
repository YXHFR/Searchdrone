"""The downward laser, read through the flight controller (``DISTANCE_SENSOR``).

``LaserReader`` judges every reading the flight controller reports and keeps what the
height floor needs (``sd_schemas.rangefinder.LaserState``). It never steers and never
writes: the closer look (``InspectHandler``) and the floor (``sd_safety.floor``) read
``state(now)``. Without a laser (``[rangefinder] model = "none"``) there is no reader.

A reading is used only when all of these hold:

* the flight controller's signal quality does not say "invalid" (1);
* the distance lies inside the flight controller's range (``RNGFND1_MIN``/``_MAX``) and
  the tested range (``[rangefinder] min_tested_m``/``max_tested_m``); a laser that sees
  nothing reports a value past its maximum;
* the drone's height (``GLOBAL_POSITION_INT``) and attitude (``ATTITUDE``) are at most
  ``LASER_GAP_S`` old;
* the drone tilts less than ``LASER_TILT_MAX_DEG``; the distance is then multiplied by
  the cosine of the tilt, so it is the height above the surface, not the slanted beam;
* the distance has not stayed exactly the same while the drone moved
  ``LASER_STUCK_ALT_M`` up or down (a stuck laser).

The laser is confirmed after ``LASER_CONFIRM_S`` of used readings in a row, no two more
than ``LASER_GAP_S`` apart, the newest at most ``LASER_GAP_S`` old. Any reading that is
not used breaks the run at once; so does silence. The surface is the drone's height
above sea level minus the corrected distance, and the confirmed surface is the highest
one of the last ``LASER_CONFIRM_S`` (the nearest treetop), never an average.
"""

from __future__ import annotations

import math
from collections import deque
from typing import Final

from sd_mav.vehicle import FcTelemetry
from sd_onboard.config import RangefinderConfig
from sd_schemas.rangefinder import (
    LASER_CONFIRM_S,
    LASER_GAP_S,
    LASER_STUCK_ALT_M,
    LASER_TILT_MAX_DEG,
    LaserState,
)

__all__ = ["LaserReader"]

_CONFIRM_MS: Final = round(LASER_CONFIRM_S * 1000)
_GAP_MS: Final = round(LASER_GAP_S * 1000)
_QUALITY_INVALID: Final = 1  # MAVLink DISTANCE_SENSOR.signal_quality: "invalid signal"
# The tilt limit as the cosine of the angle between the beam and straight down.
_COS_TILT_MAX: Final = math.cos(math.radians(LASER_TILT_MAX_DEG))


def _fresh(stamp_ms: int | None, now_ms: int) -> bool:
    return stamp_ms is not None and 0 <= now_ms - stamp_ms <= _GAP_MS


def _cos_tilt(roll_deg: float, pitch_deg: float) -> float:
    """The cosine of the angle between the drone's down axis (the beam) and straight
    down."""
    return math.cos(math.radians(roll_deg)) * math.cos(math.radians(pitch_deg))


class LaserReader:
    """Every ``DISTANCE_SENSOR`` reading in, ``LaserState`` out. ``observe`` is a
    ``Vehicle.subscribe`` callback; it skips telemetry without a new reading."""

    def __init__(self, config: RangefinderConfig) -> None:
        if not config.fitted:
            raise ValueError("no laser is fitted ([rangefinder] model = 'none')")
        self._cfg = config
        self._seen_ms: int | None = None  # the newest reading looked at
        self._epoch: int | None = None  # the flight controller's connection it came over
        self._run_start_ms: int | None = None  # the first used reading of this run
        self._last_used_ms: int | None = None
        self._surfaces: deque[tuple[int, float]] = deque()  # (time, surface AMSL) this run
        self._newest_m: float | None = None  # the newest reading, corrected; None if unused
        self._same: tuple[float, float] | None = None  # (distance, height when it began)

    def observe(self, t: FcTelemetry) -> None:
        ms = t.rangefinder_mono_ms
        if ms is None or ms == self._seen_ms:
            return
        self._seen_ms = ms
        if t.fc_link_epoch != self._epoch:  # a new connection or a reboot: start over
            self._epoch = t.fc_link_epoch
            self._break()
            self._same = None
        distance = self._judge(t, ms)
        self._newest_m = distance
        if distance is None:
            self._break()
            return
        if self._last_used_ms is not None and ms - self._last_used_ms > _GAP_MS:
            self._break()
        if self._run_start_ms is None:
            self._run_start_ms = ms
        self._last_used_ms = ms
        self._surfaces.append((ms, t.alt_amsl_m - distance))
        while self._surfaces[0][0] < ms - _CONFIRM_MS:
            self._surfaces.popleft()

    def state(self, now_mono_ms: int) -> LaserState:
        """What the laser says at ``now_mono_ms`` (the vehicle's monotonic clock)."""
        last, start = self._last_used_ms, self._run_start_ms
        fresh = _fresh(last, now_mono_ms)
        confirmed = fresh and last is not None and start is not None and last - start >= _CONFIRM_MS
        return LaserState(
            confirmed=confirmed,
            surface_amsl_m=max(s for _, s in self._surfaces) if confirmed else None,
            min_range_m=self._cfg.min_tested_m,
            distance_m=self._newest_m if fresh else None,
        )

    def _break(self) -> None:
        self._run_start_ms = None
        self._last_used_ms = None
        self._surfaces.clear()

    def _judge(self, t: FcTelemetry, ms: int) -> float | None:
        """The reading's height above the surface, or ``None`` when it is not used."""
        raw = t.rangefinder_m
        stuck = self._stuck(raw, t.alt_amsl_m)
        low = max(self._cfg.min_tested_m, t.rangefinder_min_m)
        high = min(self._cfg.max_tested_m, t.rangefinder_max_m)
        if (
            stuck
            or t.rangefinder_quality == _QUALITY_INVALID
            or not low <= raw <= high
            or not _fresh(t.position_mono_ms, ms)
            or not _fresh(t.attitude_mono_ms, ms)
        ):
            return None
        cos_tilt = _cos_tilt(t.roll_deg, t.pitch_deg)
        if cos_tilt <= _COS_TILT_MAX:  # tilting LASER_TILT_MAX_DEG or more
            return None
        return raw * cos_tilt

    def _stuck(self, raw: float, alt_amsl_m: float) -> bool:
        if self._same is None or self._same[0] != raw:
            self._same = (raw, alt_amsl_m)
            return False
        return abs(alt_amsl_m - self._same[1]) >= LASER_STUCK_ALT_M
