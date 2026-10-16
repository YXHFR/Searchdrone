"""The skill registry: the 18 entries, their rules, and how a new skill is added."""

from __future__ import annotations

from typing import Any

import pytest
from pydantic import ValidationError

from sd_schemas.base import LatLonAlt
from sd_schemas.skills import (
    FORBIDDEN_SKILL_NAMES,
    SKILL_REGISTRY,
    GotoAiParams,
    GotoParams,
    MarkLocationAiParams,
    ReportParams,
    SkillParams,
    SkillRegistry,
    SkillSpec,
    SurveySectorAiParams,
    SurveySectorParams,
    TakeoffParams,
    TargetContext,
)

NAMES = {
    "takeoff",
    "goto",
    "survey_sector",
    "hold",
    "resume",
    "return_home",
    "report",
    "mark_location",
    "propose_charter_change",
    "orbit",
    "set_camera_tilt",
    "capture_stills",
    "land_at",
    "inspect_detection",
    "follow_track",
    "stop_follow",
    "follow_me",
    "find_me",
}

# name: stage, runs_on, requires_approval_in_b, starts/ends, moves_vehicle, allowed_when,
#       allowed_under_latch, exempt_from_battery_rule, rate_class
TABLE: dict[str, tuple[Any, ...]] = {
    "takeoff": (1, "onboard", False, "starts", True, "on_ground", False, False, "action"),
    "goto": (1, "onboard", False, "-", True, "airborne", False, False, "action"),
    "survey_sector": (1, "onboard", False, "-", True, "airborne", False, False, "action"),
    "hold": (1, "onboard", False, "-", False, "airborne", False, True, "action"),
    "resume": (1, "onboard", False, "-", True, "airborne", False, False, "action"),
    "return_home": (1, "onboard", False, "ends", True, "airborne", False, True, "action"),
    "report": (1, "ground", False, "-", False, "any", True, True, "report"),
    "mark_location": (1, "ground", False, "-", False, "any", True, True, "report"),
    "propose_charter_change": (1, "ground", True, "-", False, "any", False, True, "report"),
    "orbit": (1, "onboard", False, "-", True, "airborne", False, False, "action"),
    "set_camera_tilt": (1, "onboard", False, "-", False, "any", False, True, "action"),
    "capture_stills": (1, "onboard", False, "-", False, "any", False, True, "action"),
    "land_at": (1, "onboard", True, "ends", True, "airborne", False, True, "action"),
    "inspect_detection": (2, "onboard", False, "-", True, "airborne", False, False, "action"),
    "follow_track": (2, "onboard", True, "-", True, "airborne", False, False, "action"),
    "stop_follow": (2, "onboard", False, "-", False, "airborne", False, True, "action"),
    "follow_me": (2, "onboard", True, "-", True, "airborne", False, False, "action"),
    "find_me": (2, "onboard", True, "-", True, "airborne", False, False, "action"),
}


def test_exactly_the_18_skills() -> None:
    assert len(SKILL_REGISTRY) == 18
    assert SKILL_REGISTRY.names() == NAMES
    assert set(TABLE) == NAMES


ONBOARD_IMPLEMENTED = {"takeoff", "goto", "survey_sector", "hold", "resume", "return_home"}


def test_only_built_skills_are_implemented() -> None:
    """The onboard executor implements six skills; every other one is still a stub."""
    implemented = {k.name for k in SKILL_REGISTRY if k.status == "implemented"}
    assert implemented == ONBOARD_IMPLEMENTED
    for k in SKILL_REGISTRY:
        assert bool(k.allowed_from) is (k.name in ONBOARD_IMPLEMENTED), k.name


@pytest.mark.parametrize("name", sorted(NAMES))
def test_table_values(name: str) -> None:
    k = SKILL_REGISTRY.get(name)
    stage, runs_on, approval, start_end, moves, when, latch, battery, rate = TABLE[name]
    assert k.stage == stage
    assert k.runs_on == runs_on
    assert k.requires_approval_in_b is approval
    assert k.starts_flight is (start_end == "starts")
    assert k.ends_flight is (start_end == "ends")
    assert k.moves_vehicle is moves
    assert k.allowed_when == when
    assert k.allowed_under_latch is latch
    assert k.exempt_from_battery_rule is battery
    assert k.rate_class == rate
    assert k.ai_allowed is True
    assert k.ttl_s == 15


def test_requires_approval_set() -> None:
    assert {k.name for k in SKILL_REGISTRY if k.requires_approval_in_b} == {
        "land_at",
        "propose_charter_change",
        "follow_track",
        "follow_me",
        "find_me",
    }


def test_only_report_and_mark_location_under_latch() -> None:
    assert {k.name for k in SKILL_REGISTRY if k.allowed_under_latch} == {
        "report",
        "mark_location",
    }


def test_only_takeoff_starts_flight() -> None:
    assert [k.name for k in SKILL_REGISTRY if k.starts_flight] == ["takeoff"]


def test_no_forbidden_names() -> None:
    assert SKILL_REGISTRY.names() & FORBIDDEN_SKILL_NAMES == frozenset()
    assert len(FORBIDDEN_SKILL_NAMES) == 13


@pytest.mark.parametrize("name", sorted(NAMES))
def test_every_entry_is_documented_and_has_a_schema(name: str) -> None:
    k = SKILL_REGISTRY.get(name)
    assert k.description_en
    assert k.synonyms_sv
    assert k.title_sv
    assert issubclass(k.params_model, SkillParams)
    pub = k.public()
    assert pub.params_schema["type"] == "object"
    assert pub.name == name


def _spec(name: str, **kw: Any) -> SkillSpec:
    base: dict[str, Any] = {
        "name": name,
        "stage": 2,
        "status": "stub",
        "runs_on": "onboard",
        "params_model": SkillParams,
        "moves_vehicle": False,
        "allowed_when": "any",
        "rate_class": "action",
        "description_en": "A test skill.",
        "synonyms_sv": ["test"],
        "title_sv": "Test",
    }
    base.update(kw)
    return SkillSpec(**base)


def test_registry_refuses_forbidden_name() -> None:
    with pytest.raises(ValueError, match="forbidden"):
        SkillRegistry([_spec("set_mode")])


def test_registry_refuses_duplicate() -> None:
    with pytest.raises(ValueError, match="duplicate"):
        SkillRegistry([_spec("x_one"), _spec("x_one")])


def test_new_skill_needs_only_a_new_entry() -> None:
    """A fixture registry gets one extra entry; nothing else changes."""
    extended = SkillRegistry([*SKILL_REGISTRY, _spec("scan_river")])
    assert "scan_river" in extended
    assert extended.get("scan_river").status == "stub"
    assert len(extended) == 19
    assert "scan_river" not in SKILL_REGISTRY  # the production registry is untouched
    assert "scan_river" in extended.as_mapping()


def test_unknown_skill_raises_key_error() -> None:
    with pytest.raises(KeyError):
        SKILL_REGISTRY.get("fly_to_the_moon")


def test_spec_consistency_rules() -> None:
    with pytest.raises(ValidationError, match="never allowed under latch"):
        _spec("bad_a", moves_vehicle=True, allowed_under_latch=True)
    with pytest.raises(ValidationError, match="starts the flight"):
        _spec("bad_b", starts_flight=True, allowed_when="airborne")


def test_survey_sector_ai_schema() -> None:
    SurveySectorAiParams.model_validate({"sector": "A"})
    SurveySectorAiParams.model_validate({"sector": "AB"})
    with pytest.raises(ValidationError):
        SurveySectorAiParams.model_validate({"sector": "S1"})


def test_goto_ai_needs_pin_or_point() -> None:
    GotoAiParams(pin="P1")
    GotoAiParams(lat=-35.36, lon=149.16)
    for bad in ({}, {"pin": "P1", "lat": -35.36, "lon": 149.16}, {"lat": -35.36}):
        with pytest.raises(ValidationError):
            GotoAiParams.model_validate(bad)


def test_mark_location_ai_target() -> None:
    MarkLocationAiParams(label="M")
    with pytest.raises(ValidationError):
        MarkLocationAiParams(label="M", pin="P1", lat=-35.36, lon=149.16)


def test_survey_needs_an_area() -> None:
    with pytest.raises(ValidationError):
        SurveySectorParams()


def test_report_text_bounds() -> None:
    with pytest.raises(ValidationError):
        ReportParams(text_sv="")
    with pytest.raises(ValidationError):
        ReportParams(text_sv="x" * 301)


def test_gate_targets() -> None:
    here = LatLonAlt(lat=-35.3632, lon=149.1652, alt_amsl_m=584)
    ctx = TargetContext(charter=None, position=here, home=here)
    goto = GotoParams(lat=-35.3615, lon=149.1652, alt_m=200, speed_ms=8).gate_targets(ctx)
    assert goto.alts_m == [200]
    assert goto.speed_ms == 8
    assert [(p.lat, p.lon) for p in goto.path] == [(-35.3632, 149.1652), (-35.3615, 149.1652)]
    takeoff = TakeoffParams(alt_m=40).gate_targets(ctx)
    assert takeoff.alts_m == [40]
    assert takeoff.points[0].lat == here.lat
