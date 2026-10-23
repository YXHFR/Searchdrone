"""The archive layouts: which members are pictures, their ids and sequences, and the
person boxes in their labels."""

from __future__ import annotations

import pytest
from bench.layouts import LAYOUTS, PERSON, sequence_of, voc_boxes, yolo_boxes


def test_yolo_boxes_are_fractions_of_the_picture() -> None:
    boxes = yolo_boxes(b"0 0.5 0.5 0.1 0.2\n", 1000, 500, PERSON)
    assert boxes == [pytest.approx((450.0, 200.0, 550.0, 300.0))]


def test_yolo_boxes_clip_to_the_picture_and_skip_other_classes() -> None:
    data = b"0 0.02 0.5 0.1 0.1\n3 0.5 0.5 0.1 0.1\n\n0 0.999 0.999 0.0001 0.0001\n"
    assert yolo_boxes(data, 100, 100, PERSON) == [pytest.approx((0.0, 45.0, 7.0, 55.0))]
    assert yolo_boxes(None, 100, 100, PERSON) == []


@pytest.mark.parametrize("data", [b"0 0.5 0.5 0.1\n", b"x 0.5 0.5 0.1 0.1\n", b"\xff\xfe"])
def test_a_broken_yolo_label_is_refused(data: bytes) -> None:
    with pytest.raises(ValueError):
        yolo_boxes(data, 100, 100, PERSON)


def test_yolo_labels_have_limits() -> None:
    with pytest.raises(ValueError, match="too many"):
        yolo_boxes(b"0 0.5 0.5 0.1 0.1\n" * 501, 100, 100, PERSON)
    with pytest.raises(ValueError, match="too large"):
        yolo_boxes(b"0" * ((1 << 20) + 1), 100, 100, PERSON)


def test_voc_refuses_a_dtd() -> None:
    with pytest.raises(ValueError):
        voc_boxes(b'<!DOCTYPE x [<!ENTITY a "b">]><annotation/>', 10, 10)


def test_wisard_takes_labelled_visible_pictures_only() -> None:
    names = [
        "210924_FHL_Enterprise_VIS_0126/DJI_0001.jpg",
        "210924_FHL_Enterprise_VIS_0126/DJI_0001.txt",
        "210924_FHL_Enterprise_VIS_0126/DJI_0002.jpg",  # no label: left out
        "210924_FHL_Enterprise_IR_0127/DJI_0003.jpg",  # thermal
        "210924_FHL_Enterprise_IR_0127/DJI_0003.txt",
        "deep/210924_FHL_VIS/DJI_0004.jpg",  # not one folder down
        "deep/210924_FHL_VIS/DJI_0004.txt",
    ]
    members = LAYOUTS["wisard_vis"].members(names)
    assert [(m.image_id, m.seq, m.label) for m in members] == [
        (
            "210924_FHL_Enterprise_VIS_0126/DJI_0001",
            "210924_FHL",
            "210924_FHL_Enterprise_VIS_0126/DJI_0001.txt",
        )
    ]


def test_simfarm_rounds_are_sequences() -> None:
    names = ["r1/images/a.png", "r1/labels/a.txt", "r1/images/b.png", "r1/images/c.jpg", "r1/x.png"]
    members = LAYOUTS["simfarm"].members(names)
    assert [(m.image_id, m.seq, m.label) for m in members] == [
        ("r1/a", "r1", "r1/labels/a.txt"),
        ("r1/b", "r1", None),  # a picture with nobody in it
    ]


def test_heridal_members_carry_the_place_as_sequence() -> None:
    names = [
        "h/JPEGImages/train_TRS_0001.jpg",
        "h/Annotations/train_TRS_0001.xml",
        "h/JPEGImages/test_TRS_0002.jpg",
        "__MACOSX/h/JPEGImages/._train_TRS_0003.jpg",
        "h/JPEGImages/other.jpg",
    ]
    members = LAYOUTS["heridal_voc"].members(names)
    assert [(m.image_id, m.seq) for m in members] == [
        ("train_TRS_0001", "TRS"),
        ("test_TRS_0002", "TRS"),
    ]
    assert members[0].label == "h/Annotations/train_TRS_0001.xml" and members[1].label is None


@pytest.mark.parametrize("layout", ["heridal_voc", "wisard_vis", "simfarm"])
def test_members_use_the_one_sequence_rule(layout: str) -> None:
    names = {
        "heridal_voc": ["h/JPEGImages/test_ABC_0001.jpg"],
        "wisard_vis": ["200402_Karen_VIS_1/a.jpg", "200402_Karen_VIS_1/a.txt"],
        "simfarm": ["r9/images/a.png"],
    }[layout]
    for m in LAYOUTS[layout].members(names):
        assert m.seq == sequence_of(layout, m.image_id)
