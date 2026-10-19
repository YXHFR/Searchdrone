"""GPS spoofing and GPS health from the flight controller's own messages.

Pure classes, the clock comes in as an argument; the onboard executor (L-SPOOF) and the
ground station (its own MAVLink, for the panel's alarms) run the same code.

:class:`SpoofDetector` compares the GPS receiver's raw position (``GPS_RAW_INT``) with the
EKF's (``GLOBAL_POSITION_INT``, whose height comes from the barometer with
``EK3_SRC1_POSZ 1``). Only samples with a 3D fix count; a lost or jammed fix is the dead-
reckoning failsafe's business, not spoofing.

* **S1 jump**: horizontal distance raw GPS - EKF > ``SPOOF_JUMP_M`` for ``SPOOF_JUMP_FOR_S``.
* **S2 height**: ``d = gps_alt_msl - ekf_alt_msl``; ``d0`` is the median of ``d`` over the
  ``SPOOF_ALT_BASELINE_S`` after arming. ``|d - d0| > SPOOF_ALT_M`` for ``SPOOF_ALT_FOR_S``.

It fires once per flight. A slow spoof that moves only horizontally and that the EKF
follows smoothly (below about 1 m/s) is not caught: that needs an independent position
source, which v1 does not have.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

from sd_safety.constants import (
    GPS_LOST_FOR_S,
    GPS_OK_FOR_S,
    GPS_WEAK_FOR_S,
    GPS_WEAK_HDOP,
    GPS_WEAK_SATS,
    SPOOF_ALT_BASELINE_S,
    SPOOF_ALT_FOR_S,
    SPOOF_ALT_M,
    SPOOF_JUMP_FOR_S,
    SPOOF_JUMP_M,
)
from sd_safety.targets import distance_m

__all__ = ["GnssSample", "GpsHealth", "GpsStatus", "SpoofDetector", "SpoofFinding"]


@dataclass(frozen=True, slots=True)
class GnssSample:
    """One GPS_RAW_INT next to the EKF's position at the same moment."""

    mono_ms: int
    fix_type: int
    satellites: int
    hdop: float | None
    gps_lat: float
    gps_lon: float
    gps_alt_msl_m: float
    ekf_lat: float
    ekf_lon: float
    ekf_alt_msl_m: float


@dataclass(frozen=True, slots=True)
class SpoofFinding:
    kind: Literal["jump", "altitude"]
    value_m: float  # the distance (S1) or the height difference from takeoff (S2)


def _median(values: list[float]) -> float:
    ordered = sorted(values)
    mid = len(ordered) // 2
    return ordered[mid] if len(ordered) % 2 else (ordered[mid - 1] + ordered[mid]) / 2


class _Run:
    """How long a condition has held, sample by sample."""

    def __init__(self, for_s: float) -> None:
        self.for_ms = round(for_s * 1000)
        self.since: int | None = None

    def update(self, condition: bool, now_ms: int) -> bool:
        if not condition:
            self.since = None
            return False
        if self.since is None:
            self.since = now_ms
        return now_ms - self.since >= self.for_ms


class SpoofDetector:
    def __init__(self) -> None:
        self._jump = _Run(SPOOF_JUMP_FOR_S)
        self._alt = _Run(SPOOF_ALT_FOR_S)
        self._armed_ms: int | None = None
        self._baseline: list[float] = []
        self._d0: float | None = None
        self._last_ms: int | None = None
        self.finding: SpoofFinding | None = None

    def reset(self) -> None:
        """A new flight (after landing)."""
        self.__init__()

    def update(self, s: GnssSample, *, armed: bool) -> SpoofFinding | None:
        """Feed one sample; the finding on the sample that first proves spoofing, once."""
        if s.mono_ms == self._last_ms:
            return None  # the same GPS_RAW_INT again
        self._last_ms = s.mono_ms
        if not armed:
            self._armed_ms = None
            self._baseline.clear()
            self._d0 = None
            self._jump.update(False, s.mono_ms)
            self._alt.update(False, s.mono_ms)
            return None
        if self._armed_ms is None:
            self._armed_ms = s.mono_ms
        if s.fix_type < 3:
            self._jump.update(False, s.mono_ms)
            self._alt.update(False, s.mono_ms)
            return None
        d = s.gps_alt_msl_m - s.ekf_alt_msl_m
        if self._d0 is None:
            if s.mono_ms - self._armed_ms <= SPOOF_ALT_BASELINE_S * 1000:
                self._baseline.append(d)
            elif self._baseline:
                self._d0 = _median(self._baseline)
        jump = distance_m(s.gps_lat, s.gps_lon, s.ekf_lat, s.ekf_lon)
        found: SpoofFinding | None = None
        if self._jump.update(jump > SPOOF_JUMP_M, s.mono_ms):
            found = SpoofFinding("jump", round(jump, 1))
        off = None if self._d0 is None else abs(d - self._d0)
        if self._alt.update(off is not None and off > SPOOF_ALT_M, s.mono_ms) and found is None:
            found = SpoofFinding("altitude", round(off or 0.0, 1))
        if found is None or self.finding is not None:
            return None
        self.finding = found
        return found


GpsStatus = Literal["ok", "weak", "lost"]


class GpsHealth:
    """``lost`` after ``fix < 3`` for ``GPS_LOST_FOR_S``; ``weak`` after too few satellites
    or a too large HDOP for ``GPS_WEAK_FOR_S``; back to ``ok`` after a good fix and
    neither for ``GPS_OK_FOR_S``. ``update`` returns the new status when it changes."""

    def __init__(self) -> None:
        self.status: GpsStatus = "ok"
        self._lost = _Run(GPS_LOST_FOR_S)
        self._weak = _Run(GPS_WEAK_FOR_S)
        self._good = _Run(GPS_OK_FOR_S)

    def update(
        self, *, fix_type: int, satellites: int, hdop: float | None, now_ms: int
    ) -> GpsStatus | None:
        no_fix = fix_type < 3
        weak = not no_fix and (
            satellites < GPS_WEAK_SATS or (hdop is not None and hdop > GPS_WEAK_HDOP)
        )
        lost = self._lost.update(no_fix, now_ms)
        weak_long = self._weak.update(weak, now_ms)
        good = self._good.update(not no_fix and not weak, now_ms)
        new: GpsStatus = self.status
        if lost:
            new = "lost"
        elif weak_long and self.status != "lost":
            new = "weak"
        elif good:
            new = "ok"
        if new == self.status:
            return None
        self.status = new
        return new
