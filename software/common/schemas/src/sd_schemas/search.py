"""Stage 2 search: the search profile, candidates, their assessments and a person's decision.

A detection is only a hint. Candidates are never deleted (the tables are append-only) and
only a logged-in person moves one to ``confirmed``. Nothing here describes who someone is:
no face, photo, ethnicity, religion or identity fields, and no field for a name.
"""

from __future__ import annotations

import re
from enum import StrEnum
from typing import Any, Final, Literal, Self

from pydantic import AwareDatetime, Field, field_validator, model_validator

from sd_schemas.base import OPERATOR_REF_PATTERN, Id, SdModel
from sd_schemas.colors import ColorName
from sd_schemas.link import BBox

__all__ = [
    "FIELD_KEY_PATTERN",
    "FUSION_RADIUS_M",
    "FUSION_RADIUS_MAX_M",
    "PRIORITY_HIGH",
    "PRIORITY_MEDIUM",
    "SEARCH_BATCH_MAX",
    "SEARCH_OBSERVATIONS_MAX",
    "AssessmentEngine",
    "AssessmentFlag",
    "Candidate",
    "CandidateAssessment",
    "CandidateBatch",
    "CandidateCard",
    "CandidateCounts",
    "CandidateDecision",
    "CandidateDetail",
    "CandidateObservationRow",
    "CandidateQueue",
    "CandidateRevision",
    "CandidateState",
    "CandidateWritten",
    "Certainty",
    "DecisionKind",
    "FieldVerdict",
    "Garment",
    "Hair",
    "Item",
    "LastSeen",
    "PriorityClass",
    "SearchDetection",
    "SearchHeartbeat",
    "SearchObservation",
    "SearchProfile",
    "SearchTrack",
    "StoredAssessment",
    "StoredDecision",
    "SubjectCategory",
    "fusion_radius_m",
    "priority_class",
]

Certainty = Literal["certain", "likely", "unsure"]

SubjectCategory = Literal[
    "dementia",
    "hiker",
    "despondent",
    "gatherer",
    "hunter",
    "child_1_6",
    "child_7_12",
    "child_13_15",
    "autism",
    "other",
]

MAX_GARMENTS: Final = 8
MAX_ITEMS: Final = 6

# The stable keys the assessment and the score use for each field of a profile.
FIELD_KEY_PATTERN: Final = (
    rf"^(garment:[0-{MAX_GARMENTS - 1}]|item:[0-{MAX_ITEMS - 1}]|hair|state)$"
)
_FIELD_KEY = re.compile(FIELD_KEY_PATTERN)


class Garment(SdModel):
    part: Literal["outer", "upper", "lower", "headwear", "footwear", "full_body"]
    kind: str | None = Field(default=None, max_length=40)  # free Swedish text: "jacka"
    colors: list[ColorName] = Field(min_length=1, max_length=3)  # main colour first
    shade: Literal["light", "dark"] | None = None
    pattern: Literal["solid", "striped", "checked", "patterned", "reflective"] | None = None
    certainty: Certainty
    source: str | None = Field(default=None, max_length=80)  # who said it, and when


class Item(SdModel):
    kind: Literal["backpack", "bag", "cane", "walker", "umbrella", "bicycle", "dog", "other"]
    colors: list[ColorName] = Field(default_factory=list[ColorName], max_length=3)
    certainty: Certainty
    note: str | None = Field(default=None, max_length=80)


class Hair(SdModel):
    color: ColorName
    length: Literal["short", "long", "bald", "unknown"]
    certainty: Certainty


class LastSeen(SdModel):
    """The initial planning point (IPP): where and when the person was last seen."""

    lat: float = Field(ge=-90, le=90)
    lon: float = Field(ge=-180, le=180)
    at_utc: AwareDatetime
    heading_deg: float | None = Field(default=None, ge=0, lt=360)  # direction walked, if known
    kind: Literal["point_last_seen", "last_known_position"] = "point_last_seen"
    source: str | None = Field(default=None, max_length=80)


class SearchProfile(SdModel):
    """What the person looked like, as the operator approved it.

    A proposal from the AI has ``approved_by = None`` and is never used for matching.
    The age and height bands are for the people searching, never for the matching.
    """

    profile_id: Id  # one per mission
    profile_version: int = Field(ge=1)  # grows with every approved change
    mission_id: Id
    subject_category: SubjectCategory
    mobility: Literal["good", "reduced", "unknown"] = "unknown"
    age_band: Literal["child", "youth", "adult", "senior"] | None = None
    height_band: Literal["short", "medium", "tall"] | None = None
    garments: list[Garment] = Field(default_factory=list[Garment], max_length=MAX_GARMENTS)
    items: list[Item] = Field(default_factory=list[Item], max_length=MAX_ITEMS)
    hair: Hair | None = None
    likely_state: list[Literal["walking", "sitting", "lying", "unknown"]] = Field(
        default_factory=lambda: ["unknown"], min_length=1, max_length=4
    )
    last_seen: LastSeen
    visual_notes: str = Field(default="", max_length=300)  # visible details
    free_text: str = Field(default="", max_length=2000)  # may hold health details: never sent
    excluded: list[str] = Field(default_factory=list[str], max_length=5)
    approved_by: str | None = Field(default=None, pattern=OPERATOR_REF_PATTERN)
    approved_at: AwareDatetime | None = None

    @field_validator("excluded")
    @classmethod
    def _short_exclusions(cls, value: list[str]) -> list[str]:
        if any(len(v) > 80 for v in value):
            raise ValueError("each exclusion is at most 80 characters")
        return value

    @property
    def approved(self) -> bool:
        return self.approved_by is not None and self.approved_at is not None

    @property
    def field_keys(self) -> list[str]:
        """The keys an assessment answers, in order. Without garments, items or hair
        only "is it a person" is assessed."""
        keys = [f"garment:{i}" for i in range(len(self.garments))]
        keys += [f"item:{i}" for i in range(len(self.items))]
        if self.hair is not None:
            keys.append("hair")
        if keys:
            keys.append("state")
        return keys


class CandidateState(StrEnum):
    NEW = "new"  # fusion created it
    ASSESSING = "assessing"  # an image model is working on it
    READY = "ready"  # assessed by code at least; waits for a person
    INSPECTING = "inspecting"  # the drone is flying a closer look
    CONFIRMED = "confirmed"  # a person said: this is who we are looking for
    REJECTED = "rejected"  # a person said: not them
    MERGED = "merged"  # joined into another candidate (see merged_into)


class Candidate(SdModel):
    """A place where something that may be a person was seen. One row per revision."""

    candidate_id: Id
    rev: int = Field(ge=1)  # a change is a new row with the next rev
    mission_id: Id
    seq: int = Field(ge=1)  # display number: K1, K2, ...
    state: CandidateState
    lat: float = Field(ge=-90, le=90)
    lon: float = Field(ge=-180, le=180)
    sigma95_m: float = Field(ge=0, le=1000)
    n_seen: int = Field(ge=0)  # stills with a detection here
    n_in_view: int = Field(ge=0)  # stills whose footprint covered the place
    best_detector_score: float = Field(ge=0, le=1)
    refined_person_score: float | None = Field(default=None, ge=0, le=1)
    priority: float = Field(ge=0, le=1)
    merged_into: Id | None = None
    latest_assessment_id: Id | None = None
    first_seen_utc: AwareDatetime
    last_seen_utc: AwareDatetime


FieldVerdict = Literal["match", "conflict", "not_visible"]
# No local model: without a network only the code score and a person assess (B25).
AssessmentEngine = Literal["code", "fast", "strong", "oracle"]
AssessmentFlag = Literal["motion_claim", "identity_claim", "format_repaired"]


class CandidateAssessment(SdModel):
    """One engine's view of a candidate against one profile version.

    The model fills only ``person``, ``person_p``, ``fields``, ``overall`` and
    ``explanation_sv``; the code fills the rest. Field keys follow ``FIELD_KEY_PATTERN``;
    unknown keys are dropped. ``flags`` mark claims nobody can check from a still (that
    someone moves, or who they are): the panel shows them as unverified.
    """

    candidate_id: Id
    profile_version: int = Field(ge=0)  # 0 = no approved profile yet ("is it a person")
    engine: AssessmentEngine
    model_id: str = Field(min_length=1, max_length=120)
    person: Literal["yes", "no", "unsure"]
    person_p: float = Field(ge=0, le=1)
    fields: dict[str, FieldVerdict] = Field(default_factory=dict[str, FieldVerdict])
    overall: float = Field(ge=0, le=1)  # for ranking only
    explanation_sv: str = Field(max_length=300)
    flags: list[AssessmentFlag] = Field(default_factory=list[AssessmentFlag], max_length=3)
    crops_used: list[Id] = Field(default_factory=list[Id], max_length=3)
    latency_ms: int = Field(ge=0)
    cost_sek: float = Field(default=0.0, ge=0)
    evidence: dict[str, float] = Field(default_factory=dict[str, float], max_length=32)

    @field_validator("fields", mode="before")
    @classmethod
    def _known_keys_only(cls, value: Any) -> Any:
        if isinstance(value, dict):
            return {
                k: v
                for k, v in value.items()  # pyright: ignore[reportUnknownVariableType]
                if isinstance(k, str) and _FIELD_KEY.match(k)
            }
        return value


DecisionKind = Literal["confirm", "reject", "inspect", "unsure", "reopen"]


class CandidateDecision(SdModel):
    """A person's decision on a candidate. Only a logged-in person's session may send it."""

    decision: DecisionKind
    note: str = Field(default="", max_length=300)
    subject_rev: int = Field(ge=1)  # the rev the person saw; an older one is refused


PriorityClass = Literal["high", "medium", "low"]
PRIORITY_HIGH: Final = 0.60
PRIORITY_MEDIUM: Final = 0.30


def priority_class(priority: float) -> PriorityClass:
    """The queue's class for a priority: high >= 0.60, medium >= 0.30, else low."""
    if priority >= PRIORITY_HIGH:
        return "high"
    if priority >= PRIORITY_MEDIUM:
        return "medium"
    return "low"


class CandidateCard(SdModel):
    """One candidate as the panel's queue shows it."""

    candidate_id: Id
    rev: int = Field(ge=1)  # sent back as CandidateDecision.subject_rev
    seq: int = Field(ge=1)
    state: CandidateState
    priority: float = Field(ge=0, le=1)
    priority_class: PriorityClass
    lat: float = Field(ge=-90, le=90)
    lon: float = Field(ge=-180, le=180)
    sigma95_m: float = Field(ge=0, le=1000)
    n_seen: int = Field(ge=0)
    n_in_view: int = Field(ge=0)
    best_detector_score: float = Field(ge=0, le=1)
    refined_person_score: float | None = Field(default=None, ge=0, le=1)
    thumb_detection_id: Id | None = None  # the crop with the largest px_size
    engine: AssessmentEngine | None = None  # of the latest assessment
    person: Literal["yes", "no", "unsure"] | None = None
    fields: dict[str, FieldVerdict] = Field(default_factory=dict[str, FieldVerdict])
    flags: list[AssessmentFlag] = Field(default_factory=list[AssessmentFlag], max_length=3)
    assessed_profile_version: int | None = Field(default=None, ge=0)
    current_profile_version: int | None = Field(default=None, ge=1)
    near_seq: int | None = Field(default=None, ge=1)  # "near rejected K7"
    near_state: Literal["rejected", "confirmed"] | None = None
    merged_into: Id | None = None
    first_seen_utc: AwareDatetime
    last_seen_utc: AwareDatetime


# --- the search service's API (/search/v1) ------------------------------------------------

SEARCH_BATCH_MAX: Final = 50
# Fusion: a sighting joins a candidate within FUSION_RADIUS_M + the larger 95 % radius,
# at most FUSION_RADIUS_MAX_M. sd-search fuses with it; the panel's "near" uses the same.
FUSION_RADIUS_M: Final = 4.0
FUSION_RADIUS_MAX_M: Final = 12.0


def fusion_radius_m(sigma_a: float, sigma_b: float) -> float:
    return min(FUSION_RADIUS_MAX_M, FUSION_RADIUS_M + max(sigma_a, sigma_b))


SEARCH_OBSERVATIONS_MAX: Final = 200


class SearchDetection(SdModel):
    """A stored detection of a target class, as the search service reads it."""

    detection_id: Id
    seq: int = Field(ge=1)  # the feed's cursor (after_seq)
    still_id: Id
    label: str = Field(pattern=r"^[a-z][a-z0-9_]{1,31}$")
    score: float = Field(ge=0, le=1)
    lat: float = Field(ge=-90, le=90)
    lon: float = Field(ge=-180, le=180)
    sigma95_m: float = Field(ge=0, le=1000)
    px_size: float = Field(gt=0)
    bbox: BBox  # in still pixels
    crop_box: BBox | None = None  # the crop's square in the still; None for older rows
    has_crop: bool
    captured_utc: AwareDatetime


class SearchObservation(SdModel):
    detection_id: Id
    distance_m: float = Field(ge=0, le=1000)


class CandidateRevision(SdModel):
    """A new candidate (``base_rev`` 0) or the next revision of one, from the search
    service. It may set ``new``, ``assessing``, ``ready`` or ``merged``; ``state`` None keeps
    the state in force (a candidate a person or the pilot holds). Confirmed and rejected
    are a person's and are refused here (E1)."""

    candidate_id: Id
    base_rev: int = Field(ge=0)  # the rev this one follows; 0 = a new candidate
    state: Literal["new", "assessing", "ready", "merged"] | None
    lat: float = Field(ge=-90, le=90)
    lon: float = Field(ge=-180, le=180)
    sigma95_m: float = Field(ge=0, le=1000)
    n_seen: int = Field(ge=1)
    best_detector_score: float = Field(ge=0, le=1)
    refined_person_score: float | None = Field(default=None, ge=0, le=1)
    priority: float = Field(ge=0, le=1)
    merged_into: Id | None = None
    first_seen_utc: AwareDatetime
    last_seen_utc: AwareDatetime
    observations: list[SearchObservation] = Field(
        default_factory=list[SearchObservation], max_length=SEARCH_OBSERVATIONS_MAX
    )
    assessment: CandidateAssessment | None = None  # the code's (engine "code") only

    @model_validator(mode="after")
    def _consistent(self) -> Self:
        if (self.state == "merged") != (self.merged_into is not None):
            raise ValueError("merged_into is set exactly when the state is merged")
        if self.base_rev == 0 and self.state is None:
            raise ValueError("a new candidate needs a state")
        if self.assessment is not None and (
            self.assessment.engine != "code" or self.assessment.candidate_id != self.candidate_id
        ):
            raise ValueError("only the code's assessment of this candidate rides along")
        return self


class CandidateBatch(SdModel):
    """Revisions written in one transaction: all or none."""

    revisions: list[CandidateRevision] = Field(min_length=1, max_length=SEARCH_BATCH_MAX)


class CandidateWritten(SdModel):
    candidate_id: Id
    rev: int = Field(ge=1)
    seq: int = Field(ge=1)
    state: CandidateState


class SearchTrack(SdModel):
    """A candidate in force with everything fusion needs to resume after a restart."""

    candidate: Candidate
    label: str
    decided: bool  # a person has recorded a decision on it
    sightings: list[SearchDetection]


class SearchHeartbeat(SdModel):
    """``cloud``: ``off`` (not configured), ``ok`` or ``down`` (only the code assesses)."""

    cloud: Literal["off", "ok", "down"] = "off"
    missions: int = Field(default=0, ge=0)


# --- the panel's queue (/api/v1) ----------------------------------------------------------


class CandidateCounts(SdModel):
    review: int = Field(ge=0)  # open, priority >= PRIORITY_MEDIUM
    low: int = Field(ge=0)  # open, below it
    confirmed: int = Field(ge=0)
    rejected: int = Field(ge=0)


class CandidateQueue(SdModel):
    items: list[CandidateCard]
    counts: CandidateCounts


class CandidateObservationRow(SdModel):
    detection_id: Id
    distance_m: float = Field(ge=0)
    label: str
    score: float = Field(ge=0, le=1)
    px_size: float = Field(gt=0)
    lat: float = Field(ge=-90, le=90)
    lon: float = Field(ge=-180, le=180)
    sigma95_m: float = Field(ge=0)
    has_crop: bool
    captured_utc: AwareDatetime


class StoredAssessment(CandidateAssessment):
    id: Id
    created_at: AwareDatetime


class StoredDecision(SdModel):
    id: Id
    decision: DecisionKind
    note: str = Field(max_length=300)
    subject_rev: int = Field(ge=1)
    operator_ref: str = Field(pattern=OPERATOR_REF_PATTERN)
    created_at: AwareDatetime


class CandidateDetail(SdModel):
    card: CandidateCard
    mission_id: Id
    observations: list[CandidateObservationRow]
    assessments: list[StoredAssessment]
    decisions: list[StoredDecision]
