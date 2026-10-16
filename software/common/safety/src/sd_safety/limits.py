"""Per-mission limits, the outer cap they may be raised to, and the flight controller's fence.

Every new mission starts with :data:`DEFAULT_LIMITS` (120 m, 15 m/s, 500 m, home at
15 %). A pilot may make them stricter, or raise them by click up to the outer cap
(:class:`~sd_schemas.safety.LimitCaps`) that the ground station reads from a local
settings file. The AI can never change limits or caps.

The ``ABSOLUTE_MAX_*`` constants are the standard values and the lowest any cap can be;
the two ``ABSOLUTE_MIN_*`` floors can never be loosened by any setting. Nothing here is
read from configuration.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Final, Literal

from sd_safety.constants import (
    FC_FENCE_ALT_MAX_RANGE_M,
    FC_FENCE_AREA_MARGIN_M,
    FC_FENCE_RADIUS_RANGE_M,
    FC_LOIT_SPEED_RANGE_MS,
    FENCE_ALT_MARGIN_M,
    FENCE_MARGIN_M,
    FENCE_RADIUS_MARGIN_M,
)
from sd_schemas.charter import MissionCharter
from sd_schemas.safety import (
    DEFAULT_MAX_ALT_AGL_M,
    DEFAULT_MAX_RANGE_FROM_PILOT_M,
    DEFAULT_MAX_SPEED_MS,
    MIN_ALT_AGL_M,
    MIN_BATTERY_RTL_PCT,
    LimitCaps,
    SafetyLimits,
)

__all__ = [
    "ABSOLUTE_MAX_ALT_AGL_M",
    "ABSOLUTE_MAX_RANGE_FROM_PILOT_M",
    "ABSOLUTE_MAX_SPEED_MS",
    "ABSOLUTE_MIN_ALT_AGL_M",
    "ABSOLUTE_MIN_BATTERY_RTL_PCT",
    "DEFAULT_CAPS",
    "DEFAULT_LIMITS",
    "LimitViolation",
    "charter_violations",
    "check_limits",
    "fc_area_max_from_home_m",
    "fc_fence_params",
]

DEFAULT_LIMITS: Final = SafetyLimits()
DEFAULT_CAPS: Final = LimitCaps()  # used when the local settings file sets no cap

# Standard values; also the lowest an outer cap may be (owner decision).
ABSOLUTE_MAX_ALT_AGL_M: Final = DEFAULT_MAX_ALT_AGL_M
ABSOLUTE_MAX_SPEED_MS: Final = DEFAULT_MAX_SPEED_MS
ABSOLUTE_MAX_RANGE_FROM_PILOT_M: Final = DEFAULT_MAX_RANGE_FROM_PILOT_M
# Floors that no setting can loosen.
ABSOLUTE_MIN_BATTERY_RTL_PCT: Final = MIN_BATTERY_RTL_PCT
ABSOLUTE_MIN_ALT_AGL_M: Final = MIN_ALT_AGL_M


@dataclass(frozen=True, slots=True)
class LimitViolation:
    """One field of a limits change that cannot be accepted."""

    field: str
    code: Literal["LIMIT_ABOVE_ABSOLUTE", "VALIDATION"]
    value: float
    limit: float


def fc_fence_params(limits: SafetyLimits) -> dict[str, float]:
    """The flight controller's fence for ``limits``: just outside them, never inside."""
    return {
        "FENCE_ALT_MAX": limits.max_alt_agl_m + FENCE_ALT_MARGIN_M,
        "FENCE_RADIUS": limits.max_range_from_pilot_m + FENCE_RADIUS_MARGIN_M,
        "LOIT_SPEED_MS": limits.max_speed_ms,
    }


def fc_area_max_from_home_m(limits: SafetyLimits) -> float:
    """How far from home any corner of a charter area may be, so that the area with the
    fence margins fits inside the flight controller's circle (515 m with the defaults)."""
    return fc_fence_params(limits)["FENCE_RADIUS"] - FENCE_MARGIN_M - FC_FENCE_AREA_MARGIN_M


def check_limits(limits: SafetyLimits, caps: LimitCaps) -> list[LimitViolation]:
    """Everything wrong with ``limits`` under ``caps``; empty when they can be used.

    Looser than the cap (or a floor) gives ``LIMIT_ABOVE_ABSOLUTE``. A height limit at or
    below the lowest height, or a fence the flight controller cannot hold, gives
    ``VALIDATION``.
    """
    out: list[LimitViolation] = []

    def above(field: str, value: float, cap: float) -> None:
        if value > cap:
            out.append(LimitViolation(field, "LIMIT_ABOVE_ABSOLUTE", value, cap))

    above("max_alt_agl_m", limits.max_alt_agl_m, caps.max_alt_agl_m)
    above("max_speed_ms", limits.max_speed_ms, caps.max_speed_ms)
    above("max_range_from_pilot_m", limits.max_range_from_pilot_m, caps.max_range_from_pilot_m)
    floor = max(caps.min_battery_rtl_pct, ABSOLUTE_MIN_BATTERY_RTL_PCT)
    if limits.battery_rtl_pct < floor:
        out.append(
            LimitViolation("battery_rtl_pct", "LIMIT_ABOVE_ABSOLUTE", limits.battery_rtl_pct, floor)
        )
    lowest = max(caps.min_alt_agl_m, ABSOLUTE_MIN_ALT_AGL_M)
    if limits.max_alt_agl_m <= lowest:
        out.append(LimitViolation("max_alt_agl_m", "VALIDATION", limits.max_alt_agl_m, lowest))
    ranges = {
        "FENCE_ALT_MAX": FC_FENCE_ALT_MAX_RANGE_M,
        "FENCE_RADIUS": FC_FENCE_RADIUS_RANGE_M,
        "LOIT_SPEED_MS": FC_LOIT_SPEED_RANGE_MS,
    }
    for param, value in fc_fence_params(limits).items():
        low, high = ranges[param]
        if not low <= value <= high:
            out.append(LimitViolation(param, "VALIDATION", value, high if value > high else low))
    return out


def charter_violations(charter: MissionCharter, limits: SafetyLimits) -> list[str]:
    """The charter fields that are looser than ``limits`` (empty when it fits)."""
    out: list[str] = []
    if charter.alt_max_agl_m > limits.max_alt_agl_m:
        out.append("alt_max_agl_m")
    if charter.max_speed_ms > limits.max_speed_ms:
        out.append("max_speed_ms")
    if charter.max_range_from_pilot_m > limits.max_range_from_pilot_m:
        out.append("max_range_from_pilot_m")
    if charter.battery_rtl_pct < limits.battery_rtl_pct:
        out.append("battery_rtl_pct")
    return out
