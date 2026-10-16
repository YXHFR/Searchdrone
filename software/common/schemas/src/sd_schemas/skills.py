"""The skill registry: every order the AI (or the operator) can give, with its rules.

Adding a skill means adding one ``SkillSpec`` to ``_SPECS`` (status ``stub``), then a
handler onboard and tests. The gate and the ground station need no change.
"""

from __future__ import annotations

from collections.abc import Iterable, Iterator, Mapping
from types import MappingProxyType
from typing import Any, Literal

from pydantic import ConfigDict, Field, model_validator

from sd_schemas.base import GeoJsonPolygon, Id, LatLon, LatLonAlt, SdModel
from sd_schemas.charter import CharterProposal, MissionCharter
from sd_schemas.executor import ExecutorState

__all__ = [
    "AI_SCHEMA_MAX_ALT_M",
    "AI_SCHEMA_MAX_SPEED_MS",
    "AI_SCHEMA_MIN_ALT_M",
    "FORBIDDEN_SKILL_NAMES",
    "SKILL_REGISTRY",
    "CaptureStillsParams",
    "FindMeParams",
    "FollowMeParams",
    "FollowTrackParams",
    "GateTargets",
    "GotoAiParams",
    "GotoParams",
    "HoldParams",
    "InspectDetectionParams",
    "LandAtParams",
    "MarkLocationAiParams",
    "MarkLocationParams",
    "OrbitParams",
    "ProposeCharterChangeParams",
    "ReportParams",
    "ResumeParams",
    "ReturnHomeParams",
    "SetCameraTiltParams",
    "SkillParams",
    "SkillRegistry",
    "SkillSpec",
    "SkillSpecPublic",
    "StopFollowParams",
    "SurveySectorAiParams",
    "SurveySectorParams",
    "TakeoffParams",
    "TargetContext",
]

# Fixed bounds in the AI-facing schemas. They equal the default outer cap the pilot may
# raise mission limits to; per turn the AI sees the tighter bounds of the active
# charter. The gate, not these numbers, enforces the real limits.
AI_SCHEMA_MIN_ALT_M = 5.0
AI_SCHEMA_MAX_ALT_M = 300.0
AI_SCHEMA_MAX_SPEED_MS = 25.0

# Mirror of sd_safety.policy.FORBIDDEN_SKILL_NAMES (the source); a test there checks
# that the two sets are equal. These names can never be registered as skills.
FORBIDDEN_SKILL_NAMES: frozenset[str] = frozenset(
    {
        "arm",
        "disarm",
        "set_param",
        "set_mode",
        "disable_fence",
        "set_fence",
        "set_autonomy",
        "set_autonomy_mode",
        "change_autonomy",
        "confirm_detection",
        "confirm_candidate",
        "set_velocity",
        "set_attitude",
    }
)

_PIN = r"^P[0-9]{1,3}$"
_SECTOR = r"^[A-Z]{1,2}$"


# --- what the gate checks -------------------------------------------------------------


class TargetContext(SdModel):
    charter: MissionCharter | None
    position: LatLonAlt | None  # the vehicle's current position
    home: LatLonAlt | None


class GateTargets(SdModel):
    """Where an order would take the vehicle. The gate checks every item against the limits.

    ``path`` holds the corner points of the route (current position first). The gate
    samples it every 10 m with ``sd_geo`` before checking.
    """

    points: list[LatLon] = Field(default_factory=list[LatLon])
    areas: list[GeoJsonPolygon] = Field(default_factory=list[GeoJsonPolygon])
    path: list[LatLon] = Field(default_factory=list[LatLon])
    alts_m: list[float] = Field(default_factory=list[float])
    speed_ms: float | None = None


class SkillParams(SdModel):
    """Base of every parameter model. ``gate_targets`` tells the gate where the order goes."""

    def gate_targets(self, ctx: TargetContext) -> GateTargets:
        return GateTargets()


def _here(ctx: TargetContext) -> list[LatLon]:
    if ctx.position is None:
        return []
    return [LatLon(lat=ctx.position.lat, lon=ctx.position.lon)]


def _first_corner(polygon: dict[str, Any]) -> LatLon:
    lon, lat = polygon["coordinates"][0][0]
    return LatLon(lat=lat, lon=lon)


# --- parameter models (what the gate and the executor validate) -----------------------


class TakeoffParams(SkillParams):
    alt_m: float | None = Field(default=None, gt=0)  # None = charter.default_alt_agl_m

    def gate_targets(self, ctx: TargetContext) -> GateTargets:
        alt = self.alt_m
        if alt is None and ctx.charter is not None:
            alt = ctx.charter.default_alt_agl_m
        home = ctx.home or ctx.position
        return GateTargets(
            points=[LatLon(lat=home.lat, lon=home.lon)] if home is not None else [],
            alts_m=[alt] if alt is not None else [],
        )


class GotoParams(SkillParams):
    lat: float = Field(ge=-90, le=90)
    lon: float = Field(ge=-180, le=180)
    alt_m: float = Field(gt=0)  # metres above home; a pure height change = current position
    speed_ms: float | None = Field(default=None, gt=0)  # None = the executor's default

    def gate_targets(self, ctx: TargetContext) -> GateTargets:
        target = LatLon(lat=self.lat, lon=self.lon)
        return GateTargets(
            points=[target],
            path=[*_here(ctx), target],
            alts_m=[self.alt_m],
            speed_ms=self.speed_ms,
        )


class SurveySectorParams(SkillParams):
    """Search a sector. The ground replaces ``sector_id`` with the sector's polygon."""

    polygon: GeoJsonPolygon | None = None
    sector_id: str | None = Field(default=None, pattern=_SECTOR)
    pattern: str = "lawnmower"  # anything else is rejected onboard as not implemented
    alt_m: float | None = Field(default=None, gt=0)
    spacing_m: float | None = Field(default=None, ge=10, le=100)
    speed_ms: float | None = Field(default=None, gt=0)
    heading_deg: float | None = Field(default=None, ge=0, le=180)

    @model_validator(mode="after")
    def _area_given(self) -> SurveySectorParams:
        if self.polygon is None and self.sector_id is None:
            raise ValueError("give polygon or sector_id")
        return self

    def gate_targets(self, ctx: TargetContext) -> GateTargets:
        polygon = self.polygon
        if polygon is None and ctx.charter is not None:
            polygon = next(
                (s.area for s in ctx.charter.sectors if s.sector_id == self.sector_id), None
            )
        alt = self.alt_m
        if alt is None and ctx.charter is not None:
            alt = ctx.charter.default_alt_agl_m
        if polygon is None:
            return GateTargets(alts_m=[alt] if alt is not None else [], speed_ms=self.speed_ms)
        return GateTargets(
            areas=[polygon],
            path=[*_here(ctx), _first_corner(polygon)],
            alts_m=[alt] if alt is not None else [],
            speed_ms=self.speed_ms,
        )


class HoldParams(SkillParams):
    reason: str = Field(default="", max_length=200)


class ResumeParams(SkillParams):
    pass


class ReturnHomeParams(SkillParams):
    reason: str = Field(default="", max_length=200)


class ReportParams(SkillParams):
    priority: Literal["info", "warn", "urgent"] = "info"
    text_sv: str = Field(min_length=1, max_length=300)


class MarkLocationParams(SkillParams):
    lat: float = Field(ge=-90, le=90)
    lon: float = Field(ge=-180, le=180)
    label: str = Field(min_length=1, max_length=40)

    def gate_targets(self, ctx: TargetContext) -> GateTargets:
        return GateTargets(points=[LatLon(lat=self.lat, lon=self.lon)])


class ProposeCharterChangeParams(CharterProposal, SkillParams):
    """Validated as a charter proposal by the ground station, never by the gate."""


class OrbitParams(SkillParams):
    lat: float = Field(ge=-90, le=90)
    lon: float = Field(ge=-180, le=180)
    radius_m: float = Field(ge=20, le=150)
    alt_m: float = Field(gt=0)
    turns: int = Field(default=1, ge=1, le=3)

    def gate_targets(self, ctx: TargetContext) -> GateTargets:
        centre = LatLon(lat=self.lat, lon=self.lon)
        return GateTargets(points=[centre], path=[*_here(ctx), centre], alts_m=[self.alt_m])


class InspectDetectionParams(SkillParams):
    detection_id: Id
    alt_m: float = Field(gt=0)
    standoff_m: float = Field(ge=15)


class CaptureStillsParams(SkillParams):
    count: int = Field(default=1, ge=1, le=20)
    interval_s: float = Field(default=1.0, gt=0)


class SetCameraTiltParams(SkillParams):
    deg: float = Field(default=90, ge=0, le=90)


class LandAtParams(SkillParams):
    lat: float | None = Field(default=None, ge=-90, le=90)
    lon: float | None = Field(default=None, ge=-180, le=180)
    reason: str = Field(default="", max_length=200)

    @model_validator(mode="after")
    def _both_or_neither(self) -> LandAtParams:
        if (self.lat is None) != (self.lon is None):
            raise ValueError("give both lat and lon, or neither")
        return self

    def gate_targets(self, ctx: TargetContext) -> GateTargets:
        if self.lat is None or self.lon is None:
            return GateTargets()
        target = LatLon(lat=self.lat, lon=self.lon)
        return GateTargets(points=[target], path=[*_here(ctx), target])


class FollowTrackParams(SkillParams):
    track_id: Id
    standoff_m: float = Field(ge=15)
    alt_m: float = Field(gt=0)
    max_speed_ms: float = Field(gt=0)


class StopFollowParams(SkillParams):
    then: Literal["hold", "resume"] = "hold"


class FollowMeParams(SkillParams):
    standoff_m: float = Field(ge=10, le=50)
    alt_m: float = Field(gt=0)
    max_speed_ms: float = Field(gt=0, le=8)


class FindMeParams(SkillParams):
    alt_m: float = Field(gt=0)
    hover_offset_m: float = Field(ge=15)


# --- AI-facing parameter models (adapted to the models above by sd_brain) --------------

_AiAlt = Field(default=None, ge=AI_SCHEMA_MIN_ALT_M, le=AI_SCHEMA_MAX_ALT_M)
_AiSpeed = Field(default=None, ge=1, le=AI_SCHEMA_MAX_SPEED_MS)


class GotoAiParams(SdModel):
    pin: str | None = Field(default=None, pattern=_PIN)
    lat: float | None = Field(default=None, ge=-90, le=90)
    lon: float | None = Field(default=None, ge=-180, le=180)
    alt_m: float | None = _AiAlt
    speed_ms: float | None = _AiSpeed

    @model_validator(mode="after")
    def _target(self) -> GotoAiParams:
        has_point = self.lat is not None and self.lon is not None
        if (self.lat is None) != (self.lon is None) or (self.pin is None) == (not has_point):
            raise ValueError("give pin, or lat and lon")
        return self


class SurveySectorAiParams(SdModel):
    sector: str = Field(pattern=_SECTOR)
    alt_m: float | None = _AiAlt
    spacing_m: float | None = Field(default=None, ge=10, le=100)
    speed_ms: float | None = _AiSpeed


class MarkLocationAiParams(SdModel):
    label: str = Field(min_length=1, max_length=40)
    pin: str | None = Field(default=None, pattern=_PIN)
    lat: float | None = Field(default=None, ge=-90, le=90)
    lon: float | None = Field(default=None, ge=-180, le=180)

    @model_validator(mode="after")
    def _target(self) -> MarkLocationAiParams:
        if (self.lat is None) != (self.lon is None):
            raise ValueError("give both lat and lon, or neither")
        if self.pin is not None and self.lat is not None:
            raise ValueError("give pin or lat/lon, not both")
        return self


# --- the registry ----------------------------------------------------------------------


class SkillSpec(SdModel):
    model_config = ConfigDict(extra="forbid", frozen=True, arbitrary_types_allowed=True)

    name: str = Field(pattern=r"^[a-z][a-z0-9_]*$")
    stage: Literal[1, 2, 3]  # 1 = the foundation
    status: Literal["implemented", "stub"]
    runs_on: Literal["onboard", "ground"]
    params_model: type[SkillParams]  # what the gate and the executor validate
    ai_params_model: type[SdModel] | None = None  # what the AI sees, when different
    ai_allowed: bool = True
    requires_approval_in_b: bool = False
    starts_flight: bool = False
    ends_flight: bool = False
    moves_vehicle: bool
    allowed_when: Literal["on_ground", "airborne", "any"]
    allowed_from: frozenset[ExecutorState] = frozenset()  # empty for ground skills and stubs
    ttl_s: int = Field(default=15, gt=0)
    exempt_from_battery_rule: bool = False
    allowed_under_latch: bool = False
    rate_class: Literal["action", "report", "none"]
    description_en: str = Field(min_length=1, max_length=200)
    synonyms_sv: list[str] = Field(min_length=1)
    title_sv: str = Field(min_length=1)

    @model_validator(mode="after")
    def _consistent(self) -> SkillSpec:
        if self.allowed_under_latch and self.moves_vehicle:
            raise ValueError(
                f"{self.name}: a skill that moves the vehicle is never allowed under latch"
            )
        if self.starts_flight and self.allowed_when != "on_ground":
            raise ValueError(f"{self.name}: a skill that starts the flight runs on the ground")
        if self.allowed_from and (self.status == "stub" or self.runs_on == "ground"):
            raise ValueError(f"{self.name}: only implemented onboard skills have allowed_from")
        return self

    def public(self) -> SkillSpecPublic:
        data = self.model_dump(exclude={"params_model", "ai_params_model"})
        data["allowed_from"] = sorted(data["allowed_from"])
        ai = self.ai_params_model
        return SkillSpecPublic(
            **data,
            params_schema=self.params_model.model_json_schema(),
            ai_params_schema=ai.model_json_schema() if ai is not None else None,
        )


class SkillSpecPublic(SdModel):
    """What ``GET /commander/v1/skills`` returns: a ``SkillSpec`` with JSON Schemas."""

    name: str
    stage: Literal[1, 2, 3]
    status: Literal["implemented", "stub"]
    runs_on: Literal["onboard", "ground"]
    ai_allowed: bool
    requires_approval_in_b: bool
    starts_flight: bool
    ends_flight: bool
    moves_vehicle: bool
    allowed_when: Literal["on_ground", "airborne", "any"]
    allowed_from: list[ExecutorState]
    ttl_s: int
    exempt_from_battery_rule: bool
    allowed_under_latch: bool
    rate_class: Literal["action", "report", "none"]
    description_en: str
    synonyms_sv: list[str]
    title_sv: str
    params_schema: dict[str, Any]
    ai_params_schema: dict[str, Any] | None


class SkillRegistry:
    """An immutable, ordered set of skills. Refuses duplicates and forbidden names."""

    def __init__(self, specs: Iterable[SkillSpec]) -> None:
        by_name: dict[str, SkillSpec] = {}
        for spec in specs:
            if spec.name in FORBIDDEN_SKILL_NAMES:
                raise ValueError(f"forbidden skill name: {spec.name}")
            if spec.name in by_name:
                raise ValueError(f"duplicate skill: {spec.name}")
            by_name[spec.name] = spec
        self._specs: Mapping[str, SkillSpec] = MappingProxyType(by_name)

    def __iter__(self) -> Iterator[SkillSpec]:
        return iter(self._specs.values())

    def __len__(self) -> int:
        return len(self._specs)

    def __contains__(self, name: object) -> bool:
        return name in self._specs

    def get(self, name: str) -> SkillSpec:
        """The skill called ``name``; KeyError if there is none."""
        return self._specs[name]

    def names(self) -> frozenset[str]:
        return frozenset(self._specs)

    def as_mapping(self) -> Mapping[str, SkillSpec]:
        """Read-only name -> spec view, for ``GateContext.registry``."""
        return self._specs

    def public(self) -> list[SkillSpecPublic]:
        return [spec.public() for spec in self._specs.values()]


_SPECS: tuple[SkillSpec, ...] = (
    SkillSpec(
        name="takeoff",
        stage=1,
        status="stub",
        runs_on="onboard",
        params_model=TakeoffParams,
        starts_flight=True,
        moves_vehicle=True,
        allowed_when="on_ground",
        rate_class="action",
        description_en="Arm, take off and climb to a height above home, then hover.",
        synonyms_sv=["starta", "lyft", "ta av"],
        title_sv="Starta",
    ),
    SkillSpec(
        name="goto",
        stage=1,
        status="stub",
        runs_on="onboard",
        params_model=GotoParams,
        ai_params_model=GotoAiParams,
        moves_vehicle=True,
        allowed_when="airborne",
        rate_class="action",
        description_en="Fly to a map pin or a point and hover there. Also changes height.",
        synonyms_sv=["flyg till", "åk till", "byt höjd"],
        title_sv="Flyg till punkt",
    ),
    SkillSpec(
        name="survey_sector",
        stage=1,
        status="stub",
        runs_on="onboard",
        params_model=SurveySectorParams,
        ai_params_model=SurveySectorAiParams,
        moves_vehicle=True,
        allowed_when="airborne",
        rate_class="action",
        description_en="Search a sector of the charter in a lawnmower pattern.",
        synonyms_sv=["sök av", "genomsök", "sök området"],
        title_sv="Sök av sektor",
    ),
    SkillSpec(
        name="hold",
        stage=1,
        status="stub",
        runs_on="onboard",
        params_model=HoldParams,
        moves_vehicle=False,
        allowed_when="airborne",
        exempt_from_battery_rule=True,
        rate_class="action",
        description_en="Stop and hover until resume.",
        synonyms_sv=["vänta", "stanna", "sväva"],
        title_sv="Vänta",
    ),
    SkillSpec(
        name="resume",
        stage=1,
        status="stub",
        runs_on="onboard",
        params_model=ResumeParams,
        moves_vehicle=True,
        allowed_when="airborne",
        rate_class="action",
        description_en="Continue the step that was interrupted by hold.",
        synonyms_sv=["fortsätt", "återuppta"],
        title_sv="Fortsätt steget",
    ),
    SkillSpec(
        name="return_home",
        stage=1,
        status="stub",
        runs_on="onboard",
        params_model=ReturnHomeParams,
        ends_flight=True,
        moves_vehicle=True,
        allowed_when="airborne",
        exempt_from_battery_rule=True,
        rate_class="action",
        description_en="Fly home and land.",
        synonyms_sv=["flyg hem", "landa hemma", "kom tillbaka"],
        title_sv="Flyg hem",
    ),
    SkillSpec(
        name="report",
        stage=1,
        status="stub",
        runs_on="ground",
        params_model=ReportParams,
        moves_vehicle=False,
        allowed_when="any",
        exempt_from_battery_rule=True,
        allowed_under_latch=True,
        rate_class="report",
        description_en="Send the operator an important message with a priority.",
        synonyms_sv=["rapportera", "meddela"],
        title_sv="Rapport",
    ),
    SkillSpec(
        name="mark_location",
        stage=1,
        status="stub",
        runs_on="ground",
        params_model=MarkLocationParams,
        ai_params_model=MarkLocationAiParams,
        moves_vehicle=False,
        allowed_when="any",
        exempt_from_battery_rule=True,
        allowed_under_latch=True,
        rate_class="report",
        description_en="Put a marker on the map at a pin, a point or the drone's position.",
        synonyms_sv=["markera", "sätt en markering"],
        title_sv="Markera plats",
    ),
    SkillSpec(
        name="propose_charter_change",
        stage=1,
        status="stub",
        runs_on="ground",
        params_model=ProposeCharterChangeParams,
        requires_approval_in_b=True,
        moves_vehicle=False,
        allowed_when="any",
        ttl_s=15,
        exempt_from_battery_rule=True,
        rate_class="report",
        description_en="Propose a new charter or a change; the operator must approve it.",
        synonyms_sv=["ny ram", "ändra området", "föreslå ram"],
        title_sv="Föreslå uppdragsram",
    ),
    SkillSpec(
        name="orbit",
        stage=1,
        status="stub",
        runs_on="onboard",
        params_model=OrbitParams,
        moves_vehicle=True,
        allowed_when="airborne",
        rate_class="action",
        description_en="Circle a point. Not available in v1.",
        synonyms_sv=["cirkla", "flyg runt"],
        title_sv="Cirkla",
    ),
    SkillSpec(
        name="set_camera_tilt",
        stage=1,
        status="stub",
        runs_on="onboard",
        params_model=SetCameraTiltParams,
        moves_vehicle=False,
        allowed_when="any",
        exempt_from_battery_rule=True,
        rate_class="action",
        description_en="Tilt the camera. Not available in v1.",
        synonyms_sv=["vinkla kameran", "luta kameran"],
        title_sv="Vinkla kameran",
    ),
    SkillSpec(
        name="capture_stills",
        stage=1,
        status="stub",
        runs_on="onboard",
        params_model=CaptureStillsParams,
        moves_vehicle=False,
        allowed_when="any",
        exempt_from_battery_rule=True,
        rate_class="action",
        description_en="Take still photos. Not available in v1.",
        synonyms_sv=["ta bilder", "fotografera"],
        title_sv="Ta bilder",
    ),
    SkillSpec(
        name="land_at",
        stage=1,
        status="stub",
        runs_on="onboard",
        params_model=LandAtParams,
        requires_approval_in_b=True,
        ends_flight=True,
        moves_vehicle=True,
        allowed_when="airborne",
        exempt_from_battery_rule=True,
        rate_class="action",
        description_en="Land at another point. Not available in v1.",
        synonyms_sv=["landa vid", "landa där"],
        title_sv="Landa vid punkt",
    ),
    SkillSpec(
        name="inspect_detection",
        stage=2,
        status="stub",
        runs_on="onboard",
        params_model=InspectDetectionParams,
        moves_vehicle=True,
        allowed_when="airborne",
        rate_class="action",
        description_en="Fly closer to a detection to look at it. Stage 2.",
        synonyms_sv=["titta närmare", "inspektera"],
        title_sv="Titta närmare",
    ),
    SkillSpec(
        name="follow_track",
        stage=2,
        status="stub",
        runs_on="onboard",
        params_model=FollowTrackParams,
        requires_approval_in_b=True,
        moves_vehicle=True,
        allowed_when="airborne",
        rate_class="action",
        description_en="Follow a tracked object at a distance. Stage 2.",
        synonyms_sv=["följ"],
        title_sv="Följ objekt",
    ),
    SkillSpec(
        name="stop_follow",
        stage=2,
        status="stub",
        runs_on="onboard",
        params_model=StopFollowParams,
        moves_vehicle=False,
        allowed_when="airborne",
        exempt_from_battery_rule=True,
        rate_class="action",
        description_en="Stop following. Stage 2.",
        synonyms_sv=["sluta följa"],
        title_sv="Sluta följa",
    ),
    SkillSpec(
        name="follow_me",
        stage=2,
        status="stub",
        runs_on="onboard",
        params_model=FollowMeParams,
        requires_approval_in_b=True,
        moves_vehicle=True,
        allowed_when="airborne",
        rate_class="action",
        description_en="Follow the operator at a distance. Stage 2.",
        synonyms_sv=["följ mig"],
        title_sv="Följ mig",
    ),
    SkillSpec(
        name="find_me",
        stage=2,
        status="stub",
        runs_on="onboard",
        params_model=FindMeParams,
        requires_approval_in_b=True,
        moves_vehicle=True,
        allowed_when="airborne",
        rate_class="action",
        description_en="Fly to the operator and hover beside them. Stage 2.",
        synonyms_sv=["hitta mig", "kom hit"],
        title_sv="Hitta mig",
    ),
)

SKILL_REGISTRY = SkillRegistry(_SPECS)
"""All 18 skills. Every one is a stub until its executor and tests exist."""
