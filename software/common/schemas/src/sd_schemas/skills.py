"""The skill registry: every order the AI (or the operator) can give, with its rules.

Adding a skill means adding one ``SkillSpec`` to ``_SPECS`` (status ``stub``), then a
handler onboard and tests. The gate and the ground station need no change.
"""

from __future__ import annotations

from collections.abc import Iterable, Iterator, Mapping
from types import MappingProxyType
from typing import Any, Literal

from pydantic import Field, model_validator

from sd_schemas.base import SKILL_NAME_PATTERN, GeoJsonPolygon, Id, LatLon, LatLonAlt, SdModel
from sd_schemas.charter import CharterProposal, MissionCharter
from sd_schemas.executor import ExecutorState, GotoResumePoint, ResumePoint
from sd_schemas.safety import HARD_MAX_ALT_AGL_M, HARD_MAX_SPEED_MS, MIN_ALT_AGL_M

__all__ = [
    "AI_SCHEMA_MAX_ALT_M",
    "AI_SCHEMA_MAX_SPEED_MS",
    "AI_SCHEMA_MIN_ALT_M",
    "FORBIDDEN_SKILL_NAMES",
    "SKILL_REGISTRY",
    "UNRESOLVED",
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

# Fixed bounds in the AI-facing schemas: the hard ceilings. Per turn the AI sees the
# tighter bounds of the active charter; the gate, not these numbers, enforces the limits.
AI_SCHEMA_MIN_ALT_M = MIN_ALT_AGL_M
AI_SCHEMA_MAX_ALT_M = HARD_MAX_ALT_AGL_M
AI_SCHEMA_MAX_SPEED_MS = HARD_MAX_SPEED_MS

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
    resume_point: ResumePoint | None = None  # what `resume` would continue


class GateTargets(SdModel):
    """Where an order would take the vehicle. The gate checks every item against the limits.

    ``path`` holds the corner points of the route (current position first); the gate
    samples it before checking. ``resolved = False`` means the targets could not be
    worked out (unknown sector, unknown position, a target that only exists onboard);
    the gate must then reject an order that moves the vehicle.
    """

    resolved: bool = True
    points: list[LatLon] = Field(default_factory=list[LatLon])
    areas: list[GeoJsonPolygon] = Field(default_factory=list[GeoJsonPolygon])
    path: list[LatLon] = Field(default_factory=list[LatLon])
    alts_m: list[float] = Field(default_factory=list[float])
    speed_ms: float | None = None


UNRESOLVED = GateTargets(resolved=False)


class SkillParams(SdModel):
    """Base of every parameter model. ``gate_targets`` tells the gate where the order goes.

    The default is unresolved (fail closed); every skill overrides it.
    """

    def gate_targets(self, ctx: TargetContext) -> GateTargets:
        return UNRESOLVED


class _NoTargets(SkillParams):
    """Base for skills that do not move the vehicle: nothing for the gate to place."""

    def gate_targets(self, ctx: TargetContext) -> GateTargets:
        return GateTargets()


def _here(ctx: TargetContext) -> LatLon | None:
    if ctx.position is None:
        return None
    return LatLon(lat=ctx.position.lat, lon=ctx.position.lon)


def _first_corner(polygon: dict[str, Any]) -> LatLon:
    lon, lat = polygon["coordinates"][0][0]
    return LatLon(lat=lat, lon=lon)


def _route(
    ctx: TargetContext,
    first: LatLon,
    *,
    points: list[LatLon] | None = None,
    areas: list[dict[str, Any]] | None = None,
    alt_m: float | None,
    speed_ms: float | None,
) -> GateTargets:
    """Targets for a move from the current position to ``first``; unresolved without either."""
    here = _here(ctx)
    if here is None or alt_m is None:
        return UNRESOLVED
    return GateTargets(
        points=points if points is not None else [first],
        areas=areas or [],
        path=[here, first],
        alts_m=[alt_m],
        speed_ms=speed_ms,
    )


def _default_alt(ctx: TargetContext) -> float | None:
    return ctx.charter.default_alt_agl_m if ctx.charter is not None else None


# --- parameter models (what the gate and the executor validate) -----------------------

_Alt = Field(gt=0, le=HARD_MAX_ALT_AGL_M)  # metres above home
_Speed = Field(default=None, gt=0, le=HARD_MAX_SPEED_MS)


class TakeoffParams(SkillParams):
    alt_m: float | None = Field(default=None, gt=0, le=HARD_MAX_ALT_AGL_M)  # None = default

    def gate_targets(self, ctx: TargetContext) -> GateTargets:
        alt = self.alt_m if self.alt_m is not None else _default_alt(ctx)
        home = ctx.home or ctx.position
        if home is None or alt is None:
            return UNRESOLVED
        point = LatLon(lat=home.lat, lon=home.lon)
        return GateTargets(points=[point], path=[point], alts_m=[alt])


class GotoParams(SkillParams):
    lat: float = Field(ge=-90, le=90)
    lon: float = Field(ge=-180, le=180)
    alt_m: float = _Alt  # a pure height change = the current position with a new alt_m
    speed_ms: float | None = _Speed  # None = the executor's default (below the charter max)

    def gate_targets(self, ctx: TargetContext) -> GateTargets:
        target = LatLon(lat=self.lat, lon=self.lon)
        return _route(ctx, target, alt_m=self.alt_m, speed_ms=self.speed_ms)


class SurveySectorParams(SkillParams):
    """Search a polygon or a sector. The ground replaces ``sector_id`` with its polygon."""

    polygon: GeoJsonPolygon | None = None
    sector_id: str | None = Field(default=None, pattern=_SECTOR)
    pattern: str = Field(default="lawnmower", max_length=32)  # others: not implemented onboard
    alt_m: float | None = Field(default=None, gt=0, le=HARD_MAX_ALT_AGL_M)
    spacing_m: float | None = Field(default=None, ge=10, le=100)
    speed_ms: float | None = _Speed
    heading_deg: float | None = Field(default=None, ge=0, le=180)

    @model_validator(mode="after")
    def _one_area(self) -> SurveySectorParams:
        if (self.polygon is None) == (self.sector_id is None):
            raise ValueError("give exactly one of polygon and sector_id")
        return self

    def gate_targets(self, ctx: TargetContext) -> GateTargets:
        polygon = self.polygon
        if polygon is None and ctx.charter is not None:
            polygon = next(
                (s.area for s in ctx.charter.sectors if s.sector_id == self.sector_id), None
            )
        if polygon is None:
            return UNRESOLVED
        alt = self.alt_m if self.alt_m is not None else _default_alt(ctx)
        return _route(
            ctx,
            _first_corner(polygon),
            points=[],
            areas=[polygon],
            alt_m=alt,
            speed_ms=self.speed_ms,
        )


class HoldParams(_NoTargets):
    reason: str = Field(default="", max_length=200)


class ResumeParams(SkillParams):
    """Continue the step interrupted by hold. The gate rechecks the remaining route."""

    def gate_targets(self, ctx: TargetContext) -> GateTargets:
        rp = ctx.resume_point
        if rp is None:
            return UNRESOLVED
        if isinstance(rp, GotoResumePoint):
            target = LatLon(lat=rp.lat, lon=rp.lon)
            return _route(ctx, target, alt_m=rp.alt_m, speed_ms=rp.speed_ms)
        remaining = rp.waypoints[rp.next_index :]
        if not remaining:
            return UNRESOLVED
        here = _here(ctx)
        if here is None:
            return UNRESOLVED
        return GateTargets(
            points=remaining,
            path=[here, rp.interrupted_at, *remaining],
            alts_m=[rp.alt_m],
            speed_ms=rp.speed_ms,
        )


class ReturnHomeParams(SkillParams):
    """Always allowed geometrically: flying home must never be blocked by a missing target."""

    reason: str = Field(default="", max_length=200)

    def gate_targets(self, ctx: TargetContext) -> GateTargets:
        return GateTargets()


class ReportParams(_NoTargets):
    priority: Literal["info", "warn", "urgent"] = "info"
    text_sv: str = Field(min_length=1, max_length=300)


class MarkLocationParams(SkillParams):
    """A map marker on the ground; it does not move the vehicle."""

    lat: float = Field(ge=-90, le=90)
    lon: float = Field(ge=-180, le=180)
    label: str = Field(min_length=1, max_length=40)

    def gate_targets(self, ctx: TargetContext) -> GateTargets:
        return GateTargets(points=[LatLon(lat=self.lat, lon=self.lon)])


class ProposeCharterChangeParams(CharterProposal, _NoTargets):
    """Validated as a charter proposal by the ground station, never by the gate."""


class OrbitParams(SkillParams):
    lat: float = Field(ge=-90, le=90)
    lon: float = Field(ge=-180, le=180)
    radius_m: float = Field(ge=20, le=150)
    alt_m: float = _Alt
    turns: int = Field(default=1, ge=1, le=3)

    def gate_targets(self, ctx: TargetContext) -> GateTargets:
        # The circle itself is sampled by the gate around the centre (radius_m).
        centre = LatLon(lat=self.lat, lon=self.lon)
        return _route(ctx, centre, alt_m=self.alt_m, speed_ms=None)


class InspectDetectionParams(SkillParams):
    """Stage 2. The detection's position exists only onboard: always unresolved here."""

    detection_id: Id
    alt_m: float = _Alt
    standoff_m: float = Field(ge=15, le=200)


class CaptureStillsParams(_NoTargets):
    count: int = Field(default=1, ge=1, le=20)
    interval_s: float = Field(default=1.0, gt=0, le=60)


class SetCameraTiltParams(_NoTargets):
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
        here = _here(ctx)
        if here is None:
            return UNRESOLVED
        if self.lat is None or self.lon is None:
            return GateTargets(points=[here], path=[here])  # land where it is
        target = LatLon(lat=self.lat, lon=self.lon)
        return GateTargets(points=[target], path=[here, target])


class FollowTrackParams(SkillParams):
    """Stage 2. The target moves; every set-point is checked onboard. Unresolved here."""

    track_id: Id
    standoff_m: float = Field(ge=15, le=200)
    alt_m: float = _Alt
    max_speed_ms: float = Field(gt=0, le=HARD_MAX_SPEED_MS)


class StopFollowParams(_NoTargets):
    """Stop following and hover. A later ``resume`` goes through the gate on its own."""

    then: Literal["hold"] = "hold"


class FollowMeParams(SkillParams):
    """Stage 2. Unresolved here: the operator's position is checked onboard."""

    standoff_m: float = Field(ge=10, le=50)
    alt_m: float = _Alt
    max_speed_ms: float = Field(gt=0, le=8)


class FindMeParams(SkillParams):
    """Stage 2. Unresolved here: the operator's position is checked onboard."""

    alt_m: float = _Alt
    hover_offset_m: float = Field(ge=15, le=100)


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


class _SkillSpecCommon(SdModel):
    """Fields shared by the registry entry and its public (JSON) form."""

    name: str = Field(pattern=SKILL_NAME_PATTERN)
    stage: Literal[1, 2, 3]  # 1 = the foundation
    status: Literal["implemented", "stub"]
    runs_on: Literal["onboard", "ground"]
    ai_allowed: bool = True
    requires_approval_in_b: bool = False
    starts_flight: bool = False
    ends_flight: bool = False
    moves_vehicle: bool
    allowed_when: Literal["on_ground", "airborne", "any"]
    ttl_s: int = Field(default=15, gt=0, le=300)
    exempt_from_battery_rule: bool = False
    allowed_under_latch: bool = False
    rate_class: Literal["action", "report", "none"]
    description_en: str = Field(min_length=1, max_length=200)
    synonyms_sv: list[str] = Field(min_length=1)
    title_sv: str = Field(min_length=1)


class SkillSpec(_SkillSpecCommon):
    params_model: type[SkillParams]  # what the gate and the executor validate
    ai_params_model: type[SdModel] | None = None  # what the AI sees, when different
    allowed_from: frozenset[ExecutorState] = frozenset()  # empty for ground skills and stubs

    @model_validator(mode="after")
    def _consistent(self) -> SkillSpec:
        if self.allowed_under_latch and self.moves_vehicle:
            raise ValueError(
                f"{self.name}: a skill that moves the vehicle is never allowed under latch"
            )
        if self.moves_vehicle and issubclass(self.params_model, _NoTargets):
            raise ValueError(f"{self.name}: a skill that moves the vehicle needs gate targets")
        if self.starts_flight and self.allowed_when != "on_ground":
            raise ValueError(f"{self.name}: a skill that starts the flight runs on the ground")
        if self.allowed_from and (self.status == "stub" or self.runs_on == "ground"):
            raise ValueError(f"{self.name}: only implemented onboard skills have allowed_from")
        return self

    def public(self) -> SkillSpecPublic:
        data = self.model_dump(exclude={"params_model", "ai_params_model", "allowed_from"})
        ai = self.ai_params_model
        return SkillSpecPublic(
            **data,
            allowed_from=sorted(self.allowed_from),
            params_schema=self.params_model.model_json_schema(),
            ai_params_schema=ai.model_json_schema() if ai is not None else None,
        )


class SkillSpecPublic(_SkillSpecCommon):
    """What ``GET /commander/v1/skills`` returns: a ``SkillSpec`` with JSON Schemas."""

    allowed_from: list[ExecutorState]
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
