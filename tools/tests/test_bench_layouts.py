"""The archive layouts: which members are pictures, their ids and sequences, and the
person boxes in their labels."""

from __future__ import annotations

import json
from typing import Any

import pytest
from bench.common import BenchError
from bench.layouts import KIOS_SITES, LAYOUTS, PERSON, sequence_of, voc_boxes, yolo_boxes


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


def _kios_label(image: str) -> str:
    _, part, name = image.split("/")
    return f"Annotations/Yolo/{part}/{name.removesuffix('.jpg')}.txt"


def test_kios_takes_its_pictures_with_their_yolo_labels() -> None:
    names = [
        "Images/",
        "Images/Train/12_people_0042_390.jpg",
        "Images/Valid/7_ppl_00001_crop.jpg",
        "Images/Test/4_ppl_4_00001.jpg",
        "Images/Other/9_ppl_4_00002.jpg",  # not one of the archive's parts
        "Images/Train/notes.txt",  # not a picture
        "Annotations/Yolo/Train/12_people_0042_390.txt",
        "Annotations/Yolo/Valid/7_ppl_00001_crop.txt",
        "Annotations/Yolo/Test/4_ppl_4_00001.txt",
        "Annotations/VOC/Test/4_ppl_4_00001.xml",  # VOC is never read
    ]
    members = LAYOUTS["kios_yolo"].members(names)
    assert [(m.image_id, m.seq, m.label) for m in members] == [
        ("12_people_0042_390", "campus_exercise", "Annotations/Yolo/Train/12_people_0042_390.txt"),
        ("7_ppl_00001_crop", "shore", "Annotations/Yolo/Valid/7_ppl_00001_crop.txt"),
        ("4_ppl_4_00001", "meadow", "Annotations/Yolo/Test/4_ppl_4_00001.txt"),
    ]


@pytest.mark.parametrize(
    ("a", "b"),
    [
        # the archive's own train and test folders hold frames of one recording
        ("Images/Train/1_people_0007_0.jpg", "Images/Test/2_people_0007_15.jpg"),
        # two recordings at one site
        ("Images/Train/1_ppl_00001_crop.jpg", "Images/Test/2_ppl_2_00001_crop.jpg"),
        # a recording that moves from the exercise to the campus joins them
        ("Images/Train/1_people_0042_0.jpg", "Images/Test/2_people_0043_3855.jpg"),
        ("Images/Train/1_people_0042_8160.jpg", "Images/Test/2_people_0049_160.jpg"),
    ],
)
def test_kios_near_copies_share_a_site(a: str, b: str) -> None:
    one, two = LAYOUTS["kios_yolo"].members([a, b, _kios_label(a), _kios_label(b)])
    assert one.seq == two.seq


def test_kios_sites_are_apart() -> None:
    sites = {
        sequence_of("kios_yolo", f"1_{rec}_0")
        for rec in ("people_0007", "peoplepegia", "ppl", "ppl_3", "ppl_4")
    }
    assert len(sites) == 5
    assert set(KIOS_SITES.values()) == sites


def test_kios_refuses_an_unknown_recording() -> None:
    """A recording added to the archive later has no site yet: it never lands in a part
    by accident."""
    with pytest.raises(BenchError, match="unknown recording"):
        LAYOUTS["kios_yolo"].members(["Images/Train/1_people_0099_0.jpg"])
    with pytest.raises(ValueError, match="unknown recording"):
        sequence_of("kios_yolo", "1_people_0099_0")


def test_kios_refuses_a_picture_without_labels() -> None:
    """Every KIOS picture shows people, so a missing label file is a broken archive, not
    an empty picture (it would turn real people into false alarms)."""
    with pytest.raises(BenchError, match="saknar etikettfil"):
        LAYOUTS["kios_yolo"].members(["Images/Train/1_people_0050_0.jpg"])


@pytest.mark.parametrize(
    "name", ["Images/Train/notes.jpg", "Images/Test/4_PPL_00001.jpg", "Images/Valid/ppl_1.jpg"]
)
def test_kios_refuses_a_picture_with_an_unknown_name(name: str) -> None:
    """A name of another form could be a recording the site list has never seen; it
    would be left out of the split without a word, so the set is refused instead."""
    with pytest.raises(BenchError, match="okänt namn"):
        LAYOUTS["kios_yolo"].members([name, _kios_label(name)])


# --- KIOS car/person (Zenodo 21216709): videos in zips, every tenth frame ----------------

_KCP = "60m-random-walking/DJI_20250326112218_0001_D/DJI_20250326112218_0001_D"


def _kcp(frame: int, video: str = _KCP) -> str:
    return f"{video}/frame_{frame:06d}.jpg"


def _kcp_label(frame: int, folder: str = "01D") -> str:
    return f"YOLO/{folder}/frame_{frame:06d}.txt"


def test_kios_carperson_takes_every_tenth_frame_with_its_labels() -> None:
    other = "40m-random-walking/DJI_20250326112630_0006_D/DJI_20250326112630_0006_D"
    names = [
        *(_kcp(n) for n in (0, 1, 9, 10, 20)),
        *(_kcp_label(n) for n in (0, 1, 9, 10, 20)),
        _kcp(30, other),
        _kcp_label(30, "06D"),
        "60m-random-walking/DJI_20250326112218_0001_D.zip",  # the zip itself
        "YOLO/obj.names",
    ]
    members = LAYOUTS["kios_carperson"].members(names)
    assert [(m.image_id, m.seq, m.label) for m in members] == [
        ("DJI_20250326112218_0001_D/frame_000000", "DJI_20250326112218_0001_D", _kcp_label(0)),
        ("DJI_20250326112218_0001_D/frame_000010", "DJI_20250326112218_0001_D", _kcp_label(10)),
        ("DJI_20250326112218_0001_D/frame_000020", "DJI_20250326112218_0001_D", _kcp_label(20)),
        (
            "DJI_20250326112630_0006_D/frame_000030",
            "DJI_20250326112630_0006_D",
            _kcp_label(30, "06D"),
        ),
    ]
    assert LAYOUTS["kios_carperson"].nested


def test_kios_carperson_reads_humans_not_cars() -> None:
    label = b"0 0.5 0.5 0.01 0.02\n1 0.2 0.2 0.04 0.06\n"
    boxes = LAYOUTS["kios_carperson"].boxes(label, 1920, 1080)
    assert boxes == [pytest.approx((950.4, 529.2, 969.6, 550.8))]


def test_kios_carperson_refuses_a_frame_without_labels() -> None:
    with pytest.raises(BenchError, match="saknar etikettfil"):
        LAYOUTS["kios_carperson"].members([_kcp(10)])


@pytest.mark.parametrize(
    "name",
    [
        "60m-random-walking/DJI_20250326112218_0001_D/DJI_20250326112218_0001_D/img_000010.jpg",
        "60m-random-walking/DJI_2025032611_0001_D/DJI_2025032611_0001_D/frame_000010.jpg",
        "60m-random-walking/DJI_20250326112218_0001_D/DJI_20250326112303_0002_D/frame_000010.jpg",
    ],
)
def test_kios_carperson_refuses_an_unknown_name(name: str) -> None:
    with pytest.raises(BenchError, match="okänt namn"):
        LAYOUTS["kios_carperson"].members([name, _kcp_label(10)])


# --- SeaDronesSee ODv2: COCO, swimmers only, the authors' parts pooled -------------------


def _coco(images: list[tuple[int, str, str]], boxes: list[tuple[int, int, list[float]]]) -> bytes:
    return json.dumps(
        {
            "categories": [
                {"id": 0, "name": "ignored"},
                {"id": 1, "name": "swimmer"},
                {"id": 2, "name": "boat"},
            ],
            "images": [
                {
                    "id": i,
                    "file_name": f,
                    "width": 3840,
                    "height": 2160,
                    "source": {"folder_name": v},
                }
                for i, f, v in images
            ],
            "annotations": [
                {"id": n, "image_id": i, "category_id": c, "bbox": b}
                for n, (i, c, b) in enumerate(boxes)
            ],
        }
    ).encode()


def _sds_files() -> dict[str, bytes]:
    return {
        "annotations/instances_train.json": _coco(
            [(1, "1.jpg", "DJI_0032"), (2, "2.jpg", "NorthSea_Fri_DJI_0110")],
            [(1, 1, [10, 20, 30, 40]), (1, 2, [0, 0, 200, 100]), (2, 2, [5, 5, 50, 50])],
        ),
        "annotations/instances_val.json": _coco(
            [(3, "3.jpg", "DJI_0032")], [(3, 1, [100.5, 200, 12, 8])]
        ),
        "images/train/1.jpg": b"",
        "images/train/2.jpg": b"",
        "images/val/3.jpg": b"",
        "images/test/4.jpg": b"",  # no labels: never read
    }


def _sds_members(files: dict[str, bytes]) -> list[Any]:
    return LAYOUTS["seadronessee_coco"].find(sorted(files), files.__getitem__)


def test_seadronessee_pools_the_parts_by_video_and_keeps_swimmers_only() -> None:
    members = _sds_members(_sds_files())
    assert [(m.image_id, m.seq, m.image, m.boxes) for m in members] == [
        ("DJI_0032/1", "DJI_0032", "images/train/1.jpg", ((10.0, 20.0, 40.0, 60.0),)),
        ("NorthSea_Fri_DJI_0110/2", "NorthSea_Fri_DJI_0110", "images/train/2.jpg", ()),
        ("DJI_0032/3", "DJI_0032", "images/val/3.jpg", ((100.5, 200.0, 112.5, 208.0),)),
    ]


def test_seadronessee_needs_its_labels_read() -> None:
    with pytest.raises(BenchError, match="etiketterna"):
        LAYOUTS["seadronessee_coco"].find(sorted(_sds_files()))


def test_seadronessee_refuses_a_labelled_picture_that_is_missing() -> None:
    files = _sds_files()
    del files["images/val/3.jpg"]
    with pytest.raises(BenchError, match="saknas"):
        _sds_members(files)


@pytest.mark.parametrize(
    ("file_name", "folder"), [("../1.jpg", "DJI_0032"), ("1.jpg", "a/b"), ("1.png", "DJI_0032")]
)
def test_seadronessee_refuses_an_unknown_name(file_name: str, folder: str) -> None:
    files = _sds_files()
    files["annotations/instances_val.json"] = _coco([(3, file_name, folder)], [])
    with pytest.raises(BenchError, match="okänt namn"):
        _sds_members(files)


def test_seadronessee_refuses_broken_labels() -> None:
    files = _sds_files()
    files["annotations/instances_val.json"] = b'{"images": []}'
    with pytest.raises(BenchError, match="går inte att läsa"):
        _sds_members(files)


@pytest.mark.parametrize(
    "layout",
    ["heridal_voc", "wisard_vis", "kios_yolo", "simfarm", "kios_carperson", "seadronessee_coco"],
)
def test_members_use_the_one_sequence_rule(layout: str) -> None:
    if layout == "seadronessee_coco":
        members = _sds_members(_sds_files())
    else:
        names = {
            "heridal_voc": ["h/JPEGImages/test_ABC_0001.jpg"],
            "wisard_vis": ["200402_Karen_VIS_1/a.jpg", "200402_Karen_VIS_1/a.txt"],
            "kios_yolo": [
                "Images/Valid/3_peoplepegia_105.jpg",
                _kios_label("Images/Valid/3_peoplepegia_105.jpg"),
            ],
            "simfarm": ["r9/images/a.png"],
            "kios_carperson": [_kcp(40), _kcp_label(40)],
        }[layout]
        members = LAYOUTS[layout].members(names)
    assert members
    for m in members:
        assert m.seq == sequence_of(layout, m.image_id)
