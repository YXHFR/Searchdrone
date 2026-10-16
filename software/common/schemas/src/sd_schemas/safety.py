"""Mission limits and the outer cap they may be raised to.

``SafetyLimits()`` gives the standard values every new mission starts with. A pilot may
make them stricter, or raise them (by click, per mission) up to the outer cap
(``LimitCaps``), which the ground station reads from a local settings file. The owner
decides the cap freely in that file; only the floors below can never be loosened. The
AI can never change limits or caps. The gate (``sd_safety``) enforces the limits; this
module describes them and rejects values that can never be valid.
"""

from __future__ import annotations

from pydantic import Field, model_validator

from sd_schemas.base import SdModel

__all__ = [
    "DEFAULT_BATTERY_RTL_PCT",
    "DEFAULT_CAP_ALT_AGL_M",
    "DEFAULT_CAP_RANGE_FROM_PILOT_M",
    "DEFAULT_CAP_SPEED_MS",
    "DEFAULT_MAX_ALT_AGL_M",
    "DEFAULT_MAX_RANGE_FROM_PILOT_M",
    "DEFAULT_MAX_SPEED_MS",
    "MIN_ALT_AGL_M",
    "MIN_BATTERY_RTL_PCT",
    "LimitCaps",
    "SafetyLimits",
]

# Standard values for a new mission (owner decision: 120 m, 15 m/s, 500 m, home at 15 %).
DEFAULT_MAX_ALT_AGL_M = 120.0
DEFAULT_MAX_SPEED_MS = 15.0
DEFAULT_MAX_RANGE_FROM_PILOT_M = 500.0
DEFAULT_BATTERY_RTL_PCT = 15.0

# The outer cap when the local settings file does not set one (owner decision).
DEFAULT_CAP_ALT_AGL_M = 300.0
DEFAULT_CAP_SPEED_MS = 25.0
DEFAULT_CAP_RANGE_FROM_PILOT_M = 2000.0

# Floors that no setting can loosen.
MIN_ALT_AGL_M = 5.0
MIN_BATTERY_RTL_PCT = 15.0


class SafetyLimits(SdModel):
    """Per-mission limits (outer layer). Heights are metres above the home point."""

    max_alt_agl_m: float = Field(default=DEFAULT_MAX_ALT_AGL_M, gt=MIN_ALT_AGL_M)
    max_speed_ms: float = Field(default=DEFAULT_MAX_SPEED_MS, gt=0)
    max_range_from_pilot_m: float = Field(default=DEFAULT_MAX_RANGE_FROM_PILOT_M, gt=0)
    battery_rtl_pct: float = Field(  # fly home at this level
        default=DEFAULT_BATTERY_RTL_PCT, ge=MIN_BATTERY_RTL_PCT, le=90
    )


class LimitCaps(SdModel):
    """The outer cap a pilot may raise ``SafetyLimits`` to, per mission and by click only.

    Read by the ground station from its local settings file; the owner sets it freely.
    A cap is never below the standard values, and the battery floor and lowest height
    can never be loosened.
    """

    max_alt_agl_m: float = Field(default=DEFAULT_CAP_ALT_AGL_M, ge=DEFAULT_MAX_ALT_AGL_M)
    max_speed_ms: float = Field(default=DEFAULT_CAP_SPEED_MS, ge=DEFAULT_MAX_SPEED_MS)
    max_range_from_pilot_m: float = Field(
        default=DEFAULT_CAP_RANGE_FROM_PILOT_M,
        ge=DEFAULT_MAX_RANGE_FROM_PILOT_M,
    )
    min_battery_rtl_pct: float = Field(
        default=MIN_BATTERY_RTL_PCT, ge=MIN_BATTERY_RTL_PCT, le=DEFAULT_BATTERY_RTL_PCT
    )
    min_alt_agl_m: float = Field(default=MIN_ALT_AGL_M, ge=MIN_ALT_AGL_M, lt=DEFAULT_MAX_ALT_AGL_M)

    def allows(self, limits: SafetyLimits) -> bool:
        """Whether ``limits`` lie within this cap."""
        return (
            limits.max_alt_agl_m <= self.max_alt_agl_m
            and limits.max_speed_ms <= self.max_speed_ms
            and limits.max_range_from_pilot_m <= self.max_range_from_pilot_m
            and limits.battery_rtl_pct >= self.min_battery_rtl_pct
        )

    @model_validator(mode="after")
    def _standard_values_fit(self) -> LimitCaps:
        if not self.allows(SafetyLimits()):
            raise ValueError("the cap must allow the standard limits")
        return self
