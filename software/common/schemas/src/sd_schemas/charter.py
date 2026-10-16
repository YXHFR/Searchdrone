"""The mission charter (the inner limits the AI flies within) and proposals for it."""

from __future__ import annotations

from enum import StrEnum
from typing import Literal

from pydantic import AwareDatetime, Field, model_validator

from sd_schemas.base import GeoJsonPolygon, Id, LatLon, LatLonAlt, SdModel, VehicleId
from sd_schemas.safety import (
    DEFAULT_BATTERY_RTL_PCT,
    DEFAULT_MAX_ALT_AGL_M,
    DEFAULT_MAX_RANGE_FROM_PILOT_M,
    DEFAULT_MAX_SPEED_MS,
)

__all__ = [
    "SECTOR_MAX_AREA_M2",
    "AutonomyMode",
    "CharterProposal",
    "CharterValidationError",
    "CharterValidationResult",
    "MissionCharter",
    "Sector",
]

SECTOR_MAX_AREA_M2 = 250_000.0  # largest area one survey_sector may cover


class AutonomyMode(StrEnum):
    """Set only by the operator on the charter card; the AI can never change it."""

    APPROVE_EACH = "APPROVE_EACH"  # mode A: every step is approved
    WITHIN_CHARTER = "WITHIN_CHARTER"  # mode B: free within the charter


class Sector(SdModel):
    sector_id: str = Field(pattern=r"^[A-Z]{1,2}$")  # "A", "B", ... stable between versions
    name: str  # display name, defaults to the sector_id
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
    alt_min_agl_m: float = 15
    alt_max_agl_m: float = DEFAULT_MAX_ALT_AGL_M
    default_alt_agl_m: float = 40
    max_speed_ms: float = DEFAULT_MAX_SPEED_MS
    max_range_from_pilot_m: float = DEFAULT_MAX_RANGE_FROM_PILOT_M
    battery_rtl_pct: float = DEFAULT_BATTERY_RTL_PCT
    allowed_skills: list[str]
    on_ai_silent: Literal["continue_plan", "hold", "return_home"] = "continue_plan"
    ai_silent_timeout_s: float = Field(default=10, gt=0)
    autonomy_mode: AutonomyMode = AutonomyMode.WITHIN_CHARTER  # *server*, set on approval
    sectors: list[Sector]  # *server* when the proposal has none
    fence_hash: str | None  # *server*
    approved_by: str | None  # *server*
    approved_at: AwareDatetime | None  # *server*
    notes: str | None = None


class CharterProposal(SdModel):
    """A new or changed charter from the pilot (panel) or the AI.

    It deliberately has no field for the autonomy mode, the limits, home or the pilot
    position: sending one is rejected (unknown fields are forbidden).
    """

    base_version: int | None = None  # None = the first charter
    area: GeoJsonPolygon | None = None
    area_annotation_id: Id | None = None  # a drawn area; exactly one of area / area_annotation_id
    alt_min_agl_m: float = 15
    alt_max_agl_m: float | None = None  # None = the mission's max_alt_agl_m
    geofence_buffer_m: float = Field(default=20, ge=5, le=50)
    default_alt_agl_m: float = 40
    max_speed_ms: float | None = None  # None = the limit
    max_range_from_pilot_m: float | None = None  # None = the limit
    battery_rtl_pct: float | None = None  # None = the limit
    allowed_skills: list[str] | None = None  # None = the default
    sectors: list[Sector] | None = None
    reason_sv: str = Field(max_length=500)

    @model_validator(mode="after")
    def _one_area(self) -> CharterProposal:
        if (self.area is None) == (self.area_annotation_id is None):
            raise ValueError("give exactly one of area and area_annotation_id")
        return self


class CharterValidationError(SdModel):
    code: str  # see sd_schemas.reason_codes.CHARTER_VALIDATION_CODES
    detail_sv: str
    field: str | None = None
    limit: float | None = None


class CharterValidationResult(SdModel):
    ok: bool
    errors: list[CharterValidationError] = Field(default_factory=list[CharterValidationError])
