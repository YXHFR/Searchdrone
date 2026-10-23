"""The code's colour score (``sd_search.color``) on synthetic crops."""

from __future__ import annotations

import numpy as np
import pytest
from search_helpers import (  # pyright: ignore[reportMissingImports]
    BLUE,
    GRASS,
    RED,
    YELLOW,
    crop_jpeg,
    profile,
)

from sd_schemas.link import BBox
from sd_search.color import (
    MIN_PERSON_PX,
    CropView,
    assess_colors,
    box_in_crop,
    ciede2000,
    srgb_to_lab,
)

CID = "0192f4a1-7c2e-7d10-9a3b-5e1f0c2d8a22"
DET = "0192f4a1-7c2e-7d10-9a3b-5e1f0c2d8a23"
BOX = BBox(x=32, y=32, w=32, h=32)  # crop_jpeg's box in a 96 px crop


def view(jpeg: bytes, *, px: float = 40.0, did: str = DET, box: BBox = BOX) -> CropView:
    return CropView(detection_id=did, px_size=px, jpeg=jpeg, box=box)


def test_color_scorer_ciede2000_matches_the_reference_data() -> None:
    # Sharma, Wu, Dalal (2005), pairs 1 and 7.
    # In units of 1e-4, so the pairs never read as positions.
    a = np.array([[500000, 26772, -797751], [500000, 0, 0]]) / 1e4
    b = np.array([[500000, 0, -827485], [500000, -10000, 20000]]) / 1e4
    assert ciede2000(a, b) == pytest.approx(np.array([20425, 23669]) / 1e4, abs=1e-4)


def test_color_scorer_lab_of_white_and_black() -> None:
    lab = srgb_to_lab(np.array([[255, 255, 255], [0, 0, 0]], dtype=np.uint8))
    assert lab[0] == pytest.approx([100.0, 0.0, 0.0], abs=0.05)
    assert lab[1] == pytest.approx([0.0, 0.0, 0.0], abs=0.05)


def test_color_scorer_box_in_crop() -> None:
    # A 60 px square of the still around a 20 px box, shrunk to 30 px.
    box = box_in_crop(BBox(x=120, y=220, w=20, h=20), BBox(x=100, y=200, w=60, h=60), 30, 30)
    assert (box.x, box.y, box.w, box.h) == (10, 10, 10, 10)


def test_color_scorer_match_on_the_main_colour() -> None:
    a = assess_colors(CID, profile(), [view(crop_jpeg(RED))], person_p=0.6)
    assert a.engine == "code" and a.profile_version == 1
    assert a.fields == {"garment:0": "match", "state": "not_visible"}
    assert a.evidence["red"] > 0.8 and a.evidence["person_px"] >= MIN_PERSON_PX
    assert [str(c) for c in a.crops_used] == [DET] and a.person == "unsure" and a.person_p == 0.6
    assert a.overall == 1.0 and "röd" in a.explanation_sv


def test_color_scorer_conflict_only_with_another_saturated_colour() -> None:
    blue = [view(crop_jpeg(BLUE))]
    assert assess_colors(CID, profile(), blue, person_p=0.6).fields["garment:0"] == "conflict"
    unsure = profile(garments=[{"part": "upper", "colors": ["red"], "certainty": "unsure"}])
    assert assess_colors(CID, unsure, blue, person_p=0.6).fields["garment:0"] == "not_visible"
    lower = profile(garments=[{"part": "lower", "colors": ["red"], "certainty": "certain"}])
    assert assess_colors(CID, lower, blue, person_p=0.6).fields["garment:0"] == "not_visible"
    grey = [view(crop_jpeg((128, 128, 128)))]  # not saturated: no evidence against red
    assert assess_colors(CID, profile(), grey, person_p=0.6).fields["garment:0"] == "not_visible"


def test_color_scorer_small_parts_match_at_eight_percent() -> None:
    p = profile(
        garments=[
            {"part": "upper", "colors": ["blue"], "certainty": "certain"},
            {"part": "headwear", "colors": ["yellow"], "certainty": "likely"},
        ],
        items=[{"kind": "backpack", "colors": ["red"], "certainty": "likely"}],
    )
    # Top tenth yellow (a cap), the rest blue: the cap reaches 0.08, a backpack would not.
    jpeg = crop_jpeg([YELLOW, *[BLUE] * 9], box=40, size=120)
    a = assess_colors(CID, p, [view(jpeg, box=BBox(x=40, y=40, w=40, h=40))], person_p=0.5)
    assert a.fields["garment:0"] == "match" and a.fields["garment:1"] == "match"
    assert a.fields["item:0"] == "not_visible"


def test_color_scorer_too_few_person_pixels_is_not_visible() -> None:
    same_as_grass = [view(crop_jpeg(GRASS))]
    a = assess_colors(CID, profile(), same_as_grass, person_p=0.4)
    assert set(a.fields.values()) == {"not_visible"} and a.crops_used == []
    tiny = [view(crop_jpeg(RED, box=4, size=12), box=BBox(x=4, y=4, w=4, h=4))]
    assert set(assess_colors(CID, profile(), tiny, person_p=0.4).fields.values()) == {"not_visible"}
    broken = [view(b"not a jpeg")]
    assert set(assess_colors(CID, profile(), broken, person_p=0.4).fields.values()) == {
        "not_visible"
    }


def test_color_scorer_exclusion_gives_conflict_on_upper() -> None:
    p = profile(
        garments=[{"part": "upper", "colors": ["unknown"], "certainty": "likely"}],
        excluded=["inte gul jacka"],
    )
    a = assess_colors(CID, p, [view(crop_jpeg(YELLOW))], person_p=0.5)
    assert a.fields["garment:0"] == "conflict"
    other = assess_colors(CID, p, [view(crop_jpeg(RED))], person_p=0.5)
    assert other.fields["garment:0"] == "not_visible"


def test_color_scorer_mean_over_crops_largest_first() -> None:
    red, blue = crop_jpeg(RED), crop_jpeg(BLUE)
    ids = [f"0192f4a1-7c2e-7d10-9a3b-5e1f0c2d8a3{i}" for i in range(4)]
    crops = [
        view(blue, px=10, did=ids[0]),
        view(red, px=50, did=ids[1]),
        view(red, px=40, did=ids[2]),
        view(red, px=30, did=ids[3]),
    ]
    a = assess_colors(CID, profile(), crops, person_p=0.5)
    assert [str(c) for c in a.crops_used] == [ids[1], ids[2], ids[3]]  # at most three, the largest
    assert a.fields["garment:0"] == "match"
    mixed = assess_colors(CID, profile(), [crops[0], crops[1]], person_p=0.5)
    assert 0.3 < mixed.evidence["red"] < 0.7  # the mean of a red and a blue crop


def test_color_scorer_without_a_profile_records_only_shares() -> None:
    a = assess_colors(CID, None, [view(crop_jpeg(RED))], person_p=0.3)
    assert a.profile_version == 0 and a.fields == {} and a.evidence["red"] > 0.8


def test_color_scorer_refuses_an_unapproved_profile() -> None:
    proposal = profile(approved_by=None, approved_at=None)
    with pytest.raises(ValueError, match="approved"):
        assess_colors(CID, proposal, [view(crop_jpeg(RED))], person_p=0.3)
