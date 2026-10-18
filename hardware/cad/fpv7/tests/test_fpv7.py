"""Acceptance checks for the fpv7 baseline model."""

from __future__ import annotations

import hashlib
import re
from pathlib import Path
from typing import Any

import assembly
import cadquery as cq
import pytest

README = Path(assembly.__file__).with_name("README.md")


def test_no_parts_overlap(report: dict[str, Any]) -> None:
    assert report["overlaps"] == []


def test_every_prop_has_its_margin(report: dict[str, Any], params: dict[str, Any]) -> None:
    limit = params["checks"]["prop_margin_min"]
    assert limit >= 5
    assert len(report["prop_margins"]) == 4
    for row in report["prop_margins"]:
        assert row["distance_mm"] >= limit, row


def test_camera_view_is_clear(report: dict[str, Any]) -> None:
    assert report["fov_hits"] == []


def test_camera_view_check_sees_a_part_in_the_way(
    drone: assembly.Drone, params: dict[str, Any]
) -> None:
    """The view check is not vacuous: a battery moved under the camera is reported."""
    import checks

    cam = next(pt for pt in drone.parts if pt.name == "camera")
    battery = next(pt for pt in drone.parts if pt.name == "battery")
    dx = cam.solid.Center().x - battery.solid.Center().x
    moved = assembly.Part("battery", "battery", battery.solid.translate((dx, 0, -100)), 1.0)
    others = [pt for pt in drone.parts if pt.name != "battery"] + [moved]
    hits = checks.fov_hits(assembly.Drone(others, [], drone.fov, drone.camera_name))
    assert [h["part"] for h in hits] == ["battery"]


def test_overlap_check_sees_an_overlap(drone: assembly.Drone) -> None:
    import checks

    motor = next(pt for pt in drone.parts if pt.name == "motor_1")
    twin = assembly.Part("twin", "motor", motor.solid.translate((5, 0, 0)), 1.0)
    found = checks.overlaps([motor, twin])
    assert [(o["a"], o["b"]) for o in found] == [("motor_1", "twin")]


def test_mass_matches_the_parts(report: dict[str, Any], params: dict[str, Any]) -> None:
    listed = sum(pt["mass_g"] for pt in report["parts"])
    assert report["mass_g"] == pytest.approx(listed, abs=0.5)
    frame = sum(pt["mass_g"] for pt in report["parts"] if pt["group"] == "frame")
    assert frame == pytest.approx(params["frame"]["mass"], abs=0.1)


def test_centre_of_gravity_is_near_the_middle(report: dict[str, Any]) -> None:
    """Off-centre mass makes the motors work unevenly; keep it within 5 mm in x and y."""
    import checks

    assert report["cog_offset_max_mm"] <= 5
    assert checks.cog_ok(report)
    assert not checks.cog_ok({**report, "cog_mm": [6.0, 0.0, 0.0]})


def test_mass_is_within_the_hardware_limit(report: dict[str, Any]) -> None:
    assert report["max_takeoff_mass_g"] > 0
    assert report["mass_g"] <= report["max_takeoff_mass_g"]
    assert report["ok"]


def test_prop_margin_check_sees_a_part_too_close(drone: assembly.Drone) -> None:
    """The margin check is not vacuous: a block 3 mm under a disc is reported at 3 mm."""
    import checks

    prop = next(pt for pt in drone.parts if pt.name == "prop_1")
    box = prop.solid.BoundingBox()
    block = cq.Solid.makeBox(10, 10, 2, cq.Vector(box.center.x, box.center.y, box.zmin - 5))
    rows = checks.prop_margins([prop, assembly.Part("block", "printed", block, 1.0)], 5)
    assert rows == [{"prop": "prop_1", "nearest": "block", "distance_mm": 3.0}]


def test_inertia_is_physical_and_compared(report: dict[str, Any]) -> None:
    ixx, iyy, izz = report["inertia_diag_kg_m2"]
    tensor = report["inertia_kg_m2"]
    for i in range(3):
        for j in range(3):
            assert tensor[i][j] == pytest.approx(tensor[j][i], abs=1e-9)
    assert min(ixx, iyy, izz) > 0
    # Triangle inequality for principal moments of a rigid body (diagonal is close to them).
    assert ixx + iyy >= izz * 0.99
    # A quadcopter turns most slowly about z.
    assert izz > max(ixx, iyy)
    # The hand estimate is rough; the model must land within a factor of two of it.
    for model, ref in zip(
        report["inertia_diag_kg_m2"], report["reference_inertia_kg_m2"], strict=True
    ):
        assert 0.5 <= model / ref <= 2.0


def digests(folder: Path) -> dict[str, str]:
    return {
        str(f.relative_to(folder)): hashlib.sha256(f.read_bytes()).hexdigest()
        for f in sorted(folder.rglob("*"))
        if f.is_file() and f.suffix in {".step", ".stl", ".json", ".md"}
    }


def test_exports_are_repeatable(
    drone: assembly.Drone, params: dict[str, Any], tmp_path: Path
) -> None:
    assembly.export(drone, params, tmp_path / "a")
    again = assembly.build(params)
    assembly.export(again, params, tmp_path / "b")
    first, second = digests(tmp_path / "a"), digests(tmp_path / "b")
    assert {"fpv7.step", "fpv7.stl", "report.json", "report.md"} <= set(first)
    assert len([k for k in first if k.startswith("stl/")]) == len(drone.parts)
    assert first == second
    assert (tmp_path / "a" / "fpv7_views.png").stat().st_size > 10_000


def leaves(node: Any, prefix: str = "") -> list[str]:
    if isinstance(node, dict):
        return [leaf for k, v in node.items() for leaf in leaves(v, f"{prefix}{k}.")]
    if isinstance(node, (int, float)) and not isinstance(node, bool):
        return [prefix.rstrip(".")]
    if isinstance(node, list) and node and all(isinstance(v, (int, float)) for v in node):
        return [prefix.rstrip(".")]
    return []


def test_every_value_has_a_source_row(params: dict[str, Any]) -> None:
    """Each number in params.yaml has a row in README.md with its source and date."""
    rows = {}
    sources = README.read_text(encoding="utf-8").split("## Sources", 1)[1]
    for line in sources.splitlines():
        m = re.match(r"^\|\s*`([\w.]+)`\s*\|(.+)\|\s*$", line)
        if m:
            rows[m[1]] = [c.strip() for c in m[2].split("|")]
    missing = [leaf for leaf in leaves(params) if leaf not in rows and leaf != "schema_version"]
    assert missing == []
    for key, cells in rows.items():
        assert key in leaves(params), f"README row for unknown value {key}"
        source, date = cells[-2], cells[-1]
        assert source, key
        assert re.fullmatch(r"\d{4}-\d{2}-\d{2}|-", date), (key, date)
