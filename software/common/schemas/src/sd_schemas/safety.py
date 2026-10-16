"""Mission limits and the outer cap they may be raised to.

``SafetyLimits()`` gives the standard values every new mission starts with. A pilot may
make them stricter, or raise them (by click, per mission) up to the outer cap
(``LimitCaps``), which the ground station reads from a local settings file. The AI can
never change either. No setting can go past the hard ceilings below; raising those
needs a code change with a flight-safety review. The gate (``sd_safety``) enforces the
limits; this module describes them and rejects values that can never be valid.
"""

from __future__ import annotations

from pydantic import Field, model_validator

from sd_schemas.base import SdModel

__all__ = [
    "DEFAULT_BATTERY_RTL_PCT",
    "DEFAULT_MAX_ALT_AGL_M",
    "DEFAULT_MAX_RANGE_FROM_PILOT_M",
    "DEFAULT_MAX_SPEED_MS",
    "HARD_MAX_ALT_AGL_M",
    "HARD_MAX_RANGE_FROM_PILOT_M",
    "HARD_MAX_SPEED_MS",
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

# Hard ceilings: the owner's outer cap. The local settings file may lower the cap,
# never raise it past these.
HARD_MAX_ALT_AGL_M = 300.0
HARD_MAX_SPEED_MS = 25.0
HARD_MAX_RANGE_FROM_PILOT_M = 2000.0

# Floors that no setting can loosen.
MIN_ALT_AGL_M = 5.0
MIN_BATTERY_RTL_PCT = 15.0


class SafetyLimits(SdModel):
    """Per-mission limits (outer layer). Heights are metres above the home point."""

    max_alt_agl_m: float = Field(
        default=DEFAULT_MAX_ALT_AGL_M, gt=MIN_ALT_AGL_M, le=HARD_MAX_ALT_AGL_M
    )
    max_speed_ms: float = Field(default=DEFAULT_MAX_SPEED_MS, gt=0, le=HARD_MAX_SPEED_MS)
    max_range_from_pilot_m: float = Field(
        default=DEFAULT_MAX_RANGE_FROM_PILOT_M, gt=0, le=HARD_MAX_RANGE_FROM_PILOT_M
    )
    battery_rtl_pct: float = Field(  # fly home at this level
        default=DEFAULT_BATTERY_RTL_PCT, ge=MIN_BATTERY_RTL_PCT, le=90
    )


class LimitCaps(SdModel):
    """The outer cap a pilot may raise ``SafetyLimits`` to, per mission and by click only.

    Read by the ground station from its local settings file. A cap lies between the
    standard values and the hard ceilings; the battery floor and lowest height can
    never be loosened.
    """

    max_alt_agl_m: float = Field(
        default=HARD_MAX_ALT_AGL_M, ge=DEFAULT_MAX_ALT_AGL_M, le=HARD_MAX_ALT_AGL_M
    )
    max_speed_ms: float = Field(
        default=HARD_MAX_SPEED_MS, ge=DEFAULT_MAX_SPEED_MS, le=HARD_MAX_SPEED_MS
    )
    max_range_from_pilot_m: float = Field(
        default=HARD_MAX_RANGE_FROM_PILOT_M,
        ge=DEFAULT_MAX_RANGE_FROM_PILOT_M,
        le=HARD_MAX_RANGE_FROM_PILOT_M,
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
