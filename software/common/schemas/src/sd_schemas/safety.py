"""Mission limits and the outer cap they may be raised to.

``SafetyLimits()`` gives the standard values every new mission starts with. A pilot may
make them stricter, or raise them up to the outer cap (``LimitCaps``), which is read
from a local settings file on the ground station. The AI can never change either.
The gate (``sd_safety``) enforces both; this module only describes them.
"""

from __future__ import annotations

from pydantic import Field, model_validator

from sd_schemas.base import SdModel

__all__ = [
    "DEFAULT_BATTERY_RTL_PCT",
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

# Floors that no setting can loosen. sd_safety.limits repeats them as ABSOLUTE_MIN_*,
# and a test there checks that the numbers match (sd_schemas cannot import sd_safety).
MIN_ALT_AGL_M = 5.0
MIN_BATTERY_RTL_PCT = 15.0


class SafetyLimits(SdModel):
    """Per-mission limits (outer layer). Heights are metres above the home point."""

    max_alt_agl_m: float = Field(default=DEFAULT_MAX_ALT_AGL_M, gt=MIN_ALT_AGL_M)
    max_speed_ms: float = Field(default=DEFAULT_MAX_SPEED_MS, gt=0)
    max_range_from_pilot_m: float = Field(default=DEFAULT_MAX_RANGE_FROM_PILOT_M, gt=0)
    battery_rtl_pct: float = Field(default=DEFAULT_BATTERY_RTL_PCT, le=90)  # fly home at this level


class LimitCaps(SdModel):
    """The outer cap a pilot may raise ``SafetyLimits`` to, per mission and by click only.

    Read by the ground station from its local settings file. The defaults are the
    owner's chosen outer cap. A cap can never be set below the standard values, and
    the battery floor and lowest height can never be loosened.
    """

    max_alt_agl_m: float = Field(default=300.0, ge=DEFAULT_MAX_ALT_AGL_M)
    max_speed_ms: float = Field(default=25.0, ge=DEFAULT_MAX_SPEED_MS)
    max_range_from_pilot_m: float = Field(default=2000.0, ge=DEFAULT_MAX_RANGE_FROM_PILOT_M)
    min_battery_rtl_pct: float = Field(default=MIN_BATTERY_RTL_PCT, ge=MIN_BATTERY_RTL_PCT, le=90)
    min_alt_agl_m: float = Field(default=MIN_ALT_AGL_M, ge=MIN_ALT_AGL_M)

    @model_validator(mode="after")
    def _standard_values_fit(self) -> LimitCaps:
        if self.min_battery_rtl_pct > DEFAULT_BATTERY_RTL_PCT:
            raise ValueError("min_battery_rtl_pct must allow the standard battery_rtl_pct")
        if self.min_alt_agl_m >= DEFAULT_MAX_ALT_AGL_M:
            raise ValueError("min_alt_agl_m must be below the standard max_alt_agl_m")
        return self
