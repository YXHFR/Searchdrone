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


@pytest.mark.parametrize(
    ("change", "check"),
    [
        ({"overlaps": [{"a": "x", "b": "y", "volume_mm3": 1.0}]}, "overlaps"),
        (
            {"prop_margins": [{"prop": "prop_1", "nearest": "x", "distance_mm": 4.99}]},
            "prop_margin",
        ),
        ({"fov_hits": [{"part": "x", "volume_mm3": 1.0}]}, "camera_view"),
        ({"cog_mm": [0.0, -5.01, 0.0]}, "cog"),
        ({"mass_g": 2500.01}, "mass"),
    ],
)
def test_each_failed_check_fails_the_report(
    report: dict[str, Any], change: dict[str, Any], check: str
) -> None:
    """Every check fails its verdict on its own, just past the limits (5 mm, 2.5 kg)."""
    import checks

    assert report["prop_margin_min_mm"] == 5
    assert report["cog_offset_max_mm"] == 5
    assert report["max_takeoff_mass_g"] == 2500
    assert all(checks.verdicts(report).values())
    bad = checks.verdicts({**report, **change})
    assert [k for k, ok in bad.items() if not ok] == [check]
    assert "FEL" in checks.markdown({**report, **change})


def test_cog_off_centre_fails_the_report(drone: assembly.Drone, params: dict[str, Any]) -> None:
    """A CoG limit tighter than the model's offset makes ``ok`` (the exit code) false."""
    import checks

    strict = {**params, "checks": {**params["checks"], "cog_offset_max": 0.0}}
    assert checks.report(drone, strict)["ok"] is False
    assert checks.report(drone, params)["ok"] is True


def test_overweight_fails_the_report_and_just_cad(
    drone: assembly.Drone,
    params: dict[str, Any],
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A take-off mass limit below the model's mass makes ``ok`` false and main() exit 1."""
    import checks

    limits = tmp_path / "limits.yaml"
    limits.write_text("max_takeoff_mass_kg: 1.0\n", encoding="utf-8")
    monkeypatch.setattr(checks, "LIMITS", limits)
    assert checks.report(drone, params)["ok"] is False
    assert assembly.main([str(tmp_path / "out")]) == 1


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


def test_covers_keep_their_margins(report: dict[str, Any], params: dict[str, Any]) -> None:
    """The printed canopy and nose fairing: light enough, clear of the propellers, no
    overlap, and the camera sees past the fairing through its window."""
    import checks

    assert report["cover_mass_g"] <= params["checks"]["cover_mass_max"] <= 80
    covers = {"canopy", "nose_fairing"}
    assert covers <= {pt["name"] for pt in report["parts"]}
    assert not [o for o in report["overlaps"] if {o["a"], o["b"]} & covers]
    assert not [h for h in report["fov_hits"] if h["part"] in covers]
    assert checks.verdicts(report)["cover_mass"]


def test_cover_to_prop_distance(drone: assembly.Drone, params: dict[str, Any]) -> None:
    import checks

    props = [pt for pt in drone.parts if pt.group == "prop"]
    covers = [pt for pt in drone.parts if pt.group == "cover"]
    for prop in props:
        for cover in covers:
            dist = prop.solid.distance(cover.solid)
            assert dist >= params["checks"]["prop_margin_min"], (prop.name, cover.name, dist)
    assert (
        checks.verdicts({**checks.report(drone, params), "cover_mass_g": 80.1})["cover_mass"]
        is False
    )


def test_covers_cut_drag(report: dict[str, Any]) -> None:
    ae = report["aero"]
    assert 0 < ae["frontal_area_before_m2"] <= ae["frontal_area_after_m2"]
    assert 0 < ae["cda_after_m2"] < ae["cda_before_m2"]
    assert ae["drag_after_n"] < ae["drag_before_n"]


def test_every_part_has_its_own_step(tmp_path: Path, drone: assembly.Drone) -> None:
    part = next(pt for pt in drone.parts if pt.name == "canopy")
    out = tmp_path / "canopy.step"
    assembly.write_part_step(part, out)
    text = out.read_text(encoding="utf-8")
    assert "PRODUCT('canopy'" in text
    assert assembly.STEP_DATE in text


def test_vendor_models_fit_their_envelopes(drone: assembly.Drone) -> None:
    """The manufacturers' models (downloaded by `just cad-detailed`) sit where the simple
    envelopes are and are no bigger than them, within 2 mm."""
    import vendor

    if not vendor.available():
        pytest.skip("manufacturer models not downloaded; run just cad-detailed")
    by_name = {pt.name: pt for pt in drone.parts}
    for m in vendor.models():
        model = vendor.placed(m, by_name[m["anchor"]].solid).BoundingBox()
        env = cq.Compound.makeCompound([by_name[n].solid for n in m["replaces"]]).BoundingBox()
        for lo, hi, elo, ehi in [
            (model.xmin, model.xmax, env.xmin, env.xmax),
            (model.ymin, model.ymax, env.ymin, env.ymax),
            (model.zmin, model.zmax, env.zmin, env.zmax),
        ]:
            assert lo >= elo - 2 and hi <= ehi + 2, (m["name"], lo, hi, elo, ehi)
