"""B2-01: the stage 2 schemas (search profile, candidates, colours, perception messages)."""

from __future__ import annotations

import base64
import re

import pytest
from pydantic import ValidationError

from sd_schemas import link, search
from sd_schemas.colors import COLOR_TABLE, ColorName
from sd_schemas_examples import JPEG, examples

EXAMPLES = examples()


def test_profile_has_no_field_for_a_name() -> None:
    assert not [f for f in search.SearchProfile.model_fields if "name" in f]


def test_color_table_covers_every_colour_but_unknown() -> None:
    assert set(COLOR_TABLE) == set(ColorName) - {ColorName.UNKNOWN}
    assert len(COLOR_TABLE) == 12
    for name, ref in COLOR_TABLE.items():
        achromatic = name in (ColorName.BLACK, ColorName.WHITE, ColorName.GRAY)
        assert (ref.delta_e is None) == achromatic, name
        assert (ref.c_max is not None) == achromatic, name
    assert COLOR_TABLE[ColorName.RED].lab == (45, 65, 45)
    assert COLOR_TABLE[ColorName.GRAY].l_min == 25 and COLOR_TABLE[ColorName.GRAY].l_max == 80


def test_field_keys_follow_the_profile() -> None:
    profile = EXAMPLES[search.SearchProfile]
    assert isinstance(profile, search.SearchProfile)
    assert profile.approved
    assert profile.field_keys == ["garment:0", "state"]
    bare = profile.model_copy(update={"garments": [], "approved_by": None, "approved_at": None})
    assert bare.field_keys == [] and not bare.approved
    for key in (*profile.field_keys, "item:5", "hair"):
        assert re.match(search.FIELD_KEY_PATTERN, key)


def test_profile_rejects_a_non_operator_approver() -> None:
    data = EXAMPLES[search.SearchProfile].model_dump(mode="json")
    for who in ("brain:ground", "search", "system:ground"):
        with pytest.raises(ValidationError):
            search.SearchProfile.model_validate({**data, "approved_by": who})


def test_assessment_drops_unknown_field_keys() -> None:
    data = EXAMPLES[search.CandidateAssessment].model_dump(mode="json")
    data["fields"] = {"garment:0": "match", "face": "match", "item:9": "conflict", "hair": "x"}
    with pytest.raises(ValidationError):  # a known key with a bad verdict still fails
        search.CandidateAssessment.model_validate(data)
    data["fields"].pop("hair")
    item = search.CandidateAssessment.model_validate(data)
    assert item.fields == {"garment:0": "match"}


def test_assessment_uses_at_most_three_crops() -> None:
    data = EXAMPLES[search.CandidateAssessment].model_dump(mode="json")
    data["crops_used"] = [data["crops_used"][0]] * 4
    with pytest.raises(ValidationError):
        search.CandidateAssessment.model_validate(data)


@pytest.mark.parametrize(
    ("priority", "klass"),
    [(0.0, "low"), (0.29, "low"), (0.3, "medium"), (0.6, "high"), (1, "high")],
)
def test_priority_class(priority: float, klass: str) -> None:
    assert search.priority_class(priority) == klass


def test_candidate_states() -> None:
    assert [s.value for s in search.CandidateState] == [
        "new",
        "assessing",
        "ready",
        "inspecting",
        "confirmed",
        "rejected",
        "merged",
    ]


def test_decision_kinds_and_rev() -> None:
    search.CandidateDecision(decision="reopen", subject_rev=1)
    with pytest.raises(ValidationError):
        search.CandidateDecision.model_validate({"decision": "approve", "subject_rev": 1})
    with pytest.raises(ValidationError):
        search.CandidateDecision.model_validate({"decision": "confirm", "subject_rev": 0})


# --- perception messages ---------------------------------------------------------------


def test_crop_jpeg_is_base64_in_json_and_bytes_in_python() -> None:
    crop = EXAMPLES[link.CropData]
    assert isinstance(crop, link.CropData)
    data = crop.model_dump(mode="json")
    assert data["jpeg"] == base64.b64encode(JPEG).decode()
    assert link.CropData.model_validate(data).jpeg == JPEG
    assert link.CropData.model_validate_json(crop.model_dump_json()) == crop


@pytest.mark.parametrize(
    "jpeg",
    [
        b"\x89PNG\r\n\x1a\n" + bytes(20),  # not a JPEG
        b"\xff\xd8\xff" + bytes(link.CROP_JPEG_MAX_BYTES),  # too large
        "not base64!",
    ],
)
def test_crop_refuses_other_images(jpeg: bytes | str) -> None:
    data = EXAMPLES[link.CropData].model_dump()
    with pytest.raises(ValidationError):
        link.CropData.model_validate({**data, "jpeg": jpeg})


def test_crop_and_view_sizes_are_capped() -> None:
    data = EXAMPLES[link.CropData].model_dump()
    with pytest.raises(ValidationError):
        link.CropData.model_validate({**data, "width": link.CROP_MAX_PX + 1})
    req = EXAMPLES[link.CropRequest].model_dump()
    with pytest.raises(ValidationError):
        link.CropRequest.model_validate({**req, "side_px": link.VIEW_SIDE_MAX_PX + 1})
    with pytest.raises(ValidationError):
        link.CropRequest.model_validate({**req, "max_px": link.CROP_MAX_PX + 1})


def test_report_holds_at_most_twenty_detections() -> None:
    report = EXAMPLES[link.PerceptionReport]
    assert isinstance(report, link.PerceptionReport)
    data = report.model_dump()
    data["detections"] = data["detections"] * (link.PERCEPTION_DETECTIONS_MAX + 1)
    with pytest.raises(ValidationError):
        link.PerceptionReport.model_validate(data)
    with pytest.raises(ValidationError):
        link.PerceptionReport.model_validate({**report.model_dump(), "footprint": []})


def test_keepout_list_is_capped_and_ordered() -> None:
    install = EXAMPLES[link.KeepoutInstall]
    assert isinstance(install, link.KeepoutInstall)
    data = install.model_dump()
    data["keepouts"] = data["keepouts"] * (link.KEEPOUTS_MAX + 1)
    with pytest.raises(ValidationError):
        link.KeepoutInstall.model_validate(data)
    with pytest.raises(ValidationError):
        link.KeepoutInstall.model_validate({**install.model_dump(), "install_seq": -1})
    keepout = install.keepouts[0].model_dump()
    with pytest.raises(ValidationError):
        link.Keepout.model_validate({**keepout, "radius_m": 0})


def test_stage2_kinds_are_reliable_and_need_a_ttl() -> None:
    assert {"PerceptionReport", "CropData", "CropRequest", "KeepoutInstall"} == (
        link.TTL_REQUIRED_KINDS
    )
    for kind in link.TTL_REQUIRED_KINDS:
        assert kind in link.KIND_MODELS
        assert link.is_reliable(kind, EXAMPLES[link.KIND_MODELS[kind]])  # type: ignore[arg-type]
