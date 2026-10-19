"""The class register: its rows, its rules, and that a new class needs only a row and a
source."""

from __future__ import annotations

import time
from pathlib import Path

import pytest

import sd_detect.classes
from sd_detect import (
    SOURCES,
    BBox,
    ClassRegistry,
    ColorDetector,
    Detection,
    Frame,
    RegistryError,
    Still,
    build_sources,
    color,
    default_registry,
    detect_all,
    keep_plausible,
)
from sd_detect.classes import DEFAULT_REGISTER
from sd_detect.still import StillDetector

SIM = {"SD_SIM": "1"}
CAR2 = 'id = "car2"\nname_sv = "bil"\nkind = "context"\nsource = "oracle"'
ROW_AA = '[[class]]\nid = "aa"\nname_sv = "a"\nkind = "context"\nsource = "oracle"\n'
ROW_BB = '[[class]]\nid = "bb"\nname_sv = "b"\nkind = "context"\nsource = "oracle"\n'


def test_start_classes() -> None:
    reg = default_registry()
    assert reg.ids() == ("person", "red_block", "car", "building", "road")
    assert reg.get("red_block").name_sv == "rött block"
    assert reg.get("red_block").kind == "target"
    assert reg.get("person").kind == "target"
    assert {s.id for s in reg if s.kind == "context"} == {"car", "building", "road"}
    assert [s.id for s in reg.by_source("color")] == ["red_block"]
    assert {s.id for s in reg.by_source("oracle")} == {"car", "building", "road"}
    assert sorted(reg.by_gz_label()) == [1, 2, 10, 11, 12]


def test_red_car_is_no_red_block() -> None:
    reg = default_registry()
    assert reg.get("red_block").fits_size(0.5)
    assert not reg.get("red_block").fits_size(4.5)
    assert reg.get("car").fits_size(4.5)
    assert reg.get("road").fits_size(10_000)  # no size limit


@pytest.mark.parametrize(
    ("row", "match"),
    [
        ('id = "face"\nname_sv = "x"\nkind = "target"\nsource = "color"', "identities"),
        ('id = "car2"\nname_sv = "ansikte"\nkind = "target"\nsource = "color"', "identities"),
        ('id = "Car"\nname_sv = "bil"\nkind = "context"\nsource = "oracle"', "id"),
        ('id = "car2"\nname_sv = "bil"\nkind = "goal"\nsource = "oracle"', "kind"),
        (
            'id = "car2"\nname_sv = "bil"\nkind = "context"\nsource = "oracle"\nsize_m = [3, 1]',
            "size_m",
        ),
        ('id = "car2"\nname_sv = "bil"\nkind = "context"\nsource = "oracle"\nextra = 1', "extra"),
    ],
)
def test_bad_rows(row: str, match: str) -> None:
    with pytest.raises(RegistryError, match=match):
        ClassRegistry.from_toml(f"[[class]]\n{row}\n")


def test_register_level_errors() -> None:
    with pytest.raises(RegistryError, match="no classes"):
        ClassRegistry.from_toml("")
    with pytest.raises(RegistryError, match="twice"):
        ClassRegistry.from_toml(
            '[[class]]\nid = "car"\nname_sv = "bil"\nkind = "context"\nsource = "oracle"\n' * 2
        )
    with pytest.raises(RegistryError, match="top-level"):
        ClassRegistry.from_toml("classes = []\n" + ROW_AA)
    with pytest.raises(RegistryError, match="not valid TOML"):
        ClassRegistry.from_toml("[[class]\n")
    label3 = "params = { gz_label = 3 }\n"
    reg = ClassRegistry.from_toml(ROW_AA + label3 + ROW_BB + label3)
    with pytest.raises(RegistryError, match="gz_label 3"):
        reg.by_gz_label()


def test_unknown_source_is_an_error() -> None:
    reg = ClassRegistry.from_toml(
        '[[class]]\nid = "aa"\nname_sv = "a"\nkind = "target"\nsource = "sonar"\n'
    )
    with pytest.raises(RegistryError, match="sonar"):
        build_sources(reg, env=SIM)


def test_oracle_and_model_sources_outside_the_simulator() -> None:
    built = build_sources(default_registry(), env={})
    assert list(built.sources) == ["color"]
    assert built.skipped == {"model": "inte byggd än", "oracle": "bara i simulatorn"}
    in_sim = build_sources(default_registry(), env=SIM)
    assert sorted(in_sim.sources) == ["color", "oracle"]
    assert in_sim.sources["oracle"].source == "oracle"


def _blue_frame() -> Frame:
    w, h = 40, 30
    px = bytearray(b"\x40\x80\x30" * (w * h))  # green grass
    for y in range(10, 16):
        for x in range(12, 22):
            i = (y * w + x) * 3
            px[i : i + 3] = b"\x20\x30\xd0"  # a blue patch
    return Frame(width=w, height=h, rgb=bytes(px), ts_mono_ms=0)


def test_a_new_class_needs_only_a_register_row() -> None:
    """A blue tarpaulin as a new target class: one more row, the existing colour source."""
    row = (
        '\n[[class]]\nid = "blue_tarp"\nname_sv = "blå tältduk"\nkind = "target"\n'
        'source = "color"\nsize_m = [1.0, 6.0]\n'
        "params = { hue_deg = [[200, 250]], min_saturation = 0.5, min_value = 0.3, "
        "min_area_px = 6 }\n"
    )
    reg = ClassRegistry.from_toml(DEFAULT_REGISTER.read_text(encoding="utf-8") + row)
    built = build_sources(reg, env={})
    found = detect_all(Still(frame=_blue_frame()), built.sources, reg)
    assert [(d.label, d.source) for d in found] == [("blue_tarp", "color")]
    assert found[0].bbox == BBox(x=12, y=10, w=10, h=6)
    # 10 px across at 4 px per metre is 2.5 m: a tarpaulin, kept; at 1 px/m it is 10 m.
    assert keep_plausible(found, 4.0, reg) == found
    assert keep_plausible(found, 1.0, reg) == []


class _SonarSource:
    name = "sonar"
    license = "Apache-2.0"
    source = "model"

    def detect_still(self, still: Still) -> list[Detection]:
        del still
        return [
            Detection(label="buoy", score=0.9, bbox=BBox(x=1, y=1, w=2, h=2)),
            Detection(label="red_block", score=0.9, bbox=BBox(x=1, y=1, w=2, h=2)),
        ]


def test_a_new_class_with_a_new_source() -> None:
    """A new class whose source is a new detector: a row plus one factory entry."""
    row = '\n[[class]]\nid = "buoy"\nname_sv = "boj"\nkind = "context"\nsource = "sonar"\n'
    reg = ClassRegistry.from_toml(DEFAULT_REGISTER.read_text(encoding="utf-8") + row)
    sonar = _SonarSource()
    assert isinstance(sonar, StillDetector)
    factories = {**SOURCES, "sonar": lambda _reg, _env: sonar}
    built = build_sources(reg, factories, env={})
    assert built.sources["sonar"] is sonar
    found = detect_all(Still(frame=_blue_frame()), {"sonar": sonar}, reg)
    # The source may report only its own classes: red_block belongs to the colour source.
    assert [(d.label, d.source) for d in found] == [("buoy", "model")]


def test_register_file_ships_with_the_package() -> None:
    assert DEFAULT_REGISTER.is_file()
    assert DEFAULT_REGISTER.parent == Path(sd_detect.classes.__file__).parent


@pytest.mark.parametrize(
    ("params", "match"),
    [
        ("{ min_saturation = 0.5, min_value = 0.3, min_area_px = 6 }", "bad colour settings"),
        (
            "{ hue_deg = [[350, 370]], min_saturation = 0.5, min_value = 0.3, min_area_px = 6 }",
            "0-360",
        ),
        ("{ hue_deg = [], min_saturation = 0.5, min_value = 0.3, min_area_px = 6 }", "0-360"),
        (
            "{ hue_deg = [[0, 10]], min_saturation = 0.5, min_value = 0.3, min_area_px = 0 }",
            "min_area_px",
        ),
        (
            "{ hue_deg = [[0, 10]], min_saturation = 0.5, min_value = 0.3, min_area_px = 6, "
            "min_score = 1.5 }",
            "min_score",
        ),
    ],
)
def test_bad_colour_settings(params: str, match: str) -> None:
    reg = ClassRegistry.from_toml(ROW_AA.replace('"oracle"', '"color"') + f"params = {params}\n")
    with pytest.raises(RegistryError, match=match):
        ColorDetector.from_registry(reg)


def test_patch_cap_and_size_check_input() -> None:
    w, h = 128, 128
    px = bytearray(b"\x40\x80\x30" * (w * h))
    for i in range(100):  # 100 separate red dots of 3 x 3 pixels
        x0, y0 = 2 + (i % 10) * 6, 2 + (i // 10) * 6
        for y in range(y0, y0 + 3):
            for x in range(x0, x0 + 3):
                px[(y * w + x) * 3 : (y * w + x) * 3 + 3] = b"\xd0\x20\x20"
    found = ColorDetector.from_registry(default_registry()).detect(
        Frame(width=w, height=h, rgb=bytes(px), ts_mono_ms=0)
    )
    assert len(found) == color.MAX_PATCHES
    with pytest.raises(ValueError, match="px_per_m"):
        keep_plausible(found, 0.0, default_registry())


def test_a_picture_mostly_in_the_colour_finds_nothing_quickly() -> None:
    w, h = 1280, 720
    red = Frame(width=w, height=h, rgb=b"\xd0\x20\x20" * (w * h), ts_mono_ms=0)
    det = ColorDetector.from_registry(default_registry())
    start = time.monotonic()
    assert det.detect(red) == []
    assert time.monotonic() - start < 1.0  # no flood fill over a million pixels
    assert det.skipped_large == 1


def test_a_thin_sliver_of_red_is_no_block() -> None:
    """A full red square is reported; a diagonal sliver of the same red (a bit of a red
    car's paint between its windows and shadow) scores below the class's ``min_score``."""
    w, h = 64, 64
    px = bytearray(b"\x40\x80\x30" * (w * h))

    def red(x: int, y: int) -> None:
        px[(y * w + x) * 3 : (y * w + x) * 3 + 3] = b"\xd0\x20\x20"

    for y in range(10, 22):  # a 12 x 12 square
        for x in range(10, 22):
            red(x, y)
    for i in range(14):  # a diagonal line, 14 px long and 1 px wide
        red(36 + i, 36 + i)
    detector = ColorDetector.from_registry(default_registry())
    found = detector.detect(Frame(width=w, height=h, rgb=bytes(px), ts_mono_ms=0))
    assert [(d.bbox.x, d.bbox.y) for d in found] == [(10, 10)]
    assert found[0].score >= detector.rules[0].min_score == 0.5
