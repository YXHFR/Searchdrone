"""The mission charter (the inner limits the AI flies within) and proposals for it."""

from __future__ import annotations

from enum import StrEnum
from typing import Annotated, Literal

from pydantic import AwareDatetime, Field, model_validator

from sd_schemas.base import (
    SKILL_NAME_PATTERN,
    GeoJsonPolygon,
    Id,
    LatLon,
    LatLonAlt,
    SdModel,
    VehicleId,
)
from sd_schemas.safety import (
    DEFAULT_BATTERY_RTL_PCT,
    DEFAULT_MAX_ALT_AGL_M,
    DEFAULT_MAX_RANGE_FROM_PILOT_M,
    DEFAULT_MAX_SPEED_MS,
    MIN_ALT_AGL_M,
    MIN_BATTERY_RTL_PCT,
)

__all__ = [
    "MAX_SECTORS",
    "SECTOR_MAX_AREA_M2",
    "AutonomyMode",
    "CharterProposal",
    "CharterValidationError",
    "CharterValidationResult",
    "MissionCharter",
    "Sector",
]

SECTOR_MAX_AREA_M2 = 250_000.0  # largest area one survey_sector may cover
MAX_SECTORS = 52


class AutonomyMode(StrEnum):
    """Set only by the operator on the charter card; the AI can never change it."""

    APPROVE_EACH = "APPROVE_EACH"  # mode A: every step is approved
    WITHIN_CHARTER = "WITHIN_CHARTER"  # mode B: free within the charter


class Sector(SdModel):
    sector_id: str = Field(pattern=r"^[A-Z]{1,2}$")  # "A", "B", ... stable between versions
    name: str = Field(min_length=1, max_length=40)  # display name, defaults to the sector_id
    area: GeoJsonPolygon  # entirely inside charter.area
    order_idx: int = Field(ge=0)


class MissionCharter(SdModel):
    """An approved (or proposed) charter. Fields marked *server* are set by the ground station."""

    charter_id: Id  # *server*
    mission_id: Id  # *server*
    vehicle_id: VehicleId  # *server*
    version: int = Field(ge=1)  # *server*, 1, 2, 3 ... per mission
    area: GeoJsonPolygon  # the search area = the gate's boundary
    geofence_buffer_m: float = Field(default=20, ge=5, le=50)
    home: LatLonAlt  # *server*: the vehicle's latest HOME_POSITION
    pilot_position: LatLon  # *server*: always = home in v1
    alt_min_agl_m: float = Field(default=15, ge=MIN_ALT_AGL_M)
    alt_max_agl_m: float = Field(default=DEFAULT_MAX_ALT_AGL_M, gt=MIN_ALT_AGL_M)
    default_alt_agl_m: float = Field(default=40, ge=MIN_ALT_AGL_M)
    max_speed_ms: float = Field(default=DEFAULT_MAX_SPEED_MS, gt=0)
    max_range_from_pilot_m: float = Field(default=DEFAULT_MAX_RANGE_FROM_PILOT_M, gt=0)
    battery_rtl_pct: float = Field(default=DEFAULT_BATTERY_RTL_PCT, ge=MIN_BATTERY_RTL_PCT, le=90)
    allowed_skills: list[Annotated[str, Field(pattern=SKILL_NAME_PATTERN)]] = Field(max_length=32)
    on_ai_silent: Literal["continue_plan", "hold", "return_home"] = "continue_plan"
    ai_silent_timeout_s: float = Field(default=10, gt=0, le=600)
    autonomy_mode: AutonomyMode  # *server*, set on approval; required so it is never implied
    sectors: list[Sector] = Field(max_length=MAX_SECTORS)  # *server* when the proposal has none
    fence_hash: str | None  # *server*
    approved_by: str | None  # *server*
    approved_at: AwareDatetime | None  # *server*
    notes: str | None = Field(default=None, max_length=2000)

    @model_validator(mode="after")
    def _altitude_band(self) -> MissionCharter:
        if not self.alt_min_agl_m <= self.default_alt_agl_m <= self.alt_max_agl_m:
            raise ValueError("need alt_min_agl_m <= default_alt_agl_m <= alt_max_agl_m")
        if self.alt_min_agl_m >= self.alt_max_agl_m:
            raise ValueError("alt_min_agl_m must be below alt_max_agl_m")
        return self


class CharterProposal(SdModel):
    """A new or changed charter from the pilot (panel) or the AI.

    It has no field for the autonomy mode, the mission's SafetyLimits, home or the pilot
    position: sending one is rejected (unknown fields are forbidden). Its own charter
    values are checked against the mission limits by the server before any card shows.
    """

    base_version: int | None = None  # None = the first charter
    area: GeoJsonPolygon | None = None
    area_annotation_id: Id | None = None  # a drawn area; exactly one of area / area_annotation_id
    # The band itself (min < default <= max) is checked by the server (ALT_BAND_INVALID).
    alt_min_agl_m: float = Field(default=15, ge=MIN_ALT_AGL_M)
    alt_max_agl_m: float | None = Field(  # None = the mission limit
        default=None, gt=MIN_ALT_AGL_M
    )
    geofence_buffer_m: float = Field(default=20, ge=5, le=50)
    default_alt_agl_m: float = Field(default=40, ge=MIN_ALT_AGL_M)
    max_speed_ms: float | None = Field(default=None, gt=0)
    max_range_from_pilot_m: float | None = Field(default=None, gt=0)
    battery_rtl_pct: float | None = Field(default=None, ge=MIN_BATTERY_RTL_PCT, le=90)
    allowed_skills: list[Annotated[str, Field(pattern=SKILL_NAME_PATTERN)]] | None = Field(
        default=None, max_length=32
    )  # None = the default
    sectors: list[Sector] | None = Field(default=None, max_length=MAX_SECTORS)
    reason_sv: str = Field(max_length=500)

    @model_validator(mode="after")
    def _one_area(self) -> CharterProposal:
        if (self.area is None) == (self.area_annotation_id is None):
            raise ValueError("give exactly one of area and area_annotation_id")
        return self


class CharterValidationError(SdModel):
    code: str = Field(pattern=r"^[A-Z][A-Z0-9_]{1,63}$")  # see reason_codes
    detail_sv: str = Field(max_length=500)
    field: str | None = None
    limit: float | None = None


class CharterValidationResult(SdModel):
    ok: bool
    errors: list[CharterValidationError] = Field(default_factory=list[CharterValidationError])
