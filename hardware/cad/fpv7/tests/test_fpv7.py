"""Acceptance checks for the fpv7 model."""

from __future__ import annotations

import hashlib
import math
import re
from pathlib import Path
from typing import Any

import assembly
import cadquery as cq
import cfd
import pytest
from OCP.Bnd import Bnd_Box
from OCP.BRepBndLib import BRepBndLib

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
    # The frame kit: its plates, arms and standoffs, and its own screws and nuts.
    kit = [
        pt
        for pt in report["parts"]
        if pt["group"] == "frame" or pt["name"].split("_")[0] in ("screw", "nut")
    ]
    assert sum(pt["mass_g"] for pt in kit) == pytest.approx(params["frame"]["mass"], abs=0.1)


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
        (
            {"plan_margins": [{"prop": "prop_1", "nearest": "x", "distance_mm": 2.99}]},
            "plan_margin",
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
    assert report["plan_margin_min_mm"] == 3
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
    """A take-off mass limit below the model's mass makes ``ok`` false and main() exit 1.
    main() gets the session's drone and skips the files (test_exports_are_repeatable
    writes them), so only its exit code is tested here."""
    import checks

    limits = tmp_path / "limits.yaml"
    limits.write_text("max_takeoff_mass_kg: 1.0\n", encoding="utf-8")
    monkeypatch.setattr(checks, "LIMITS", limits)
    assert checks.report(drone, params)["ok"] is False
    monkeypatch.setattr(assembly, "build", lambda p: drone)
    monkeypatch.setattr(assembly, "export", lambda d, p, out: checks.report(d, p))
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
    shown = drone.parts + drone.display
    assert len([k for k in first if k.startswith("stl/")]) == len(shown)
    assert len([k for k in first if k.startswith("step/")]) == len(shown) + len(drone.forms)
    assert "step/canopy_form.step" in first
    assert first == second
    for png in ("fpv7_views.png", "fpv7_closeup.png"):
        assert (tmp_path / "a" / png).stat().st_size > 10_000


def test_step_has_groups_and_colours(drone: assembly.Drone, tmp_path: Path) -> None:
    """fpv7.step: a sub-assembly per group, every part a named solid with its group's
    colour, the blades and the discs apart, millimetres."""
    out = tmp_path / "fpv7.step"
    assembly.write_step(drone, out)
    text = out.read_text(encoding="utf-8")
    groups = ("frame", "fastener", "motor", "prop", "prop_disc", "battery", "camera", "cable")
    for group in groups:
        assert f"PRODUCT('{group}_group'" in text, group
    for pt in drone.parts + drone.display:
        assert f"PRODUCT('{pt.name}'" in text, pt.name
    styled = text.count("STYLED_ITEM('color'")
    assert styled == len(drone.parts) + len(drone.display)
    assert len(re.findall(r"COLOUR_RGB\(", text)) == len(
        {pt.group for pt in drone.parts + drone.display}
    )
    assert "SI_UNIT(.MILLI.,.METRE.)" in text
    # Read back: the same number of solids.
    shape = cq.importers.importStep(str(out))
    assert len(shape.solids().vals()) == len(drone.parts) + len(drone.display)


def test_style_order_is_fixed() -> None:
    """Styles written in another order (as OCCT does from run to run) come out the same."""
    head = "DATA;\n#10 = SHAPE(#1);\n"
    chain = (
        "#{m} = MECHANICAL_DESIGN_GEOMETRIC_PRESENTATION_REPRESENTATION('',(#{i}),#{c});\n"
        "#{i} = STYLED_ITEM('color',(#{p}),#{t});\n"
        "#{p} = PRESENTATION_STYLE_ASSIGNMENT((#{u}));\n"
        "#{u} = SURFACE_STYLE_USAGE(.BOTH.,#{s});\n"
        "#{s} = SURFACE_SIDE_STYLE('',(#{a}));\n"
        "#{a} = SURFACE_STYLE_FILL_AREA(#{f});\n"
        "#{f} = FILL_AREA_STYLE('',(#{fc}));\n"
        "#{fc} = FILL_AREA_STYLE_COLOUR('',#{rgb});\n"
    )

    def text(order: list[tuple[int, int]]) -> str:
        out, n, rgb = head, 20, None
        for target, context in order:
            ids = dict(zip("mipusaf", range(n, n + 7), strict=True))
            ids["fc"] = n + 7
            out += chain.format(**ids, c=context, t=target, rgb=rgb or n + 8)
            if rgb is None:
                rgb = n + 8
                out += f"#{rgb} = COLOUR_RGB('',0.1,0.1,0.1);\n"
                n += 1
            n += 8
        return out + "ENDSEC;\n"

    a = assembly._sort_styles(text([(64, 394), (413, 743), (785, 883)]))
    b = assembly._sort_styles(text([(785, 883), (64, 394), (413, 743)]))
    assert a == b
    assert a.index("#64);") < a.index("#413);") < a.index("#785);")
    assert a.count("COLOUR_RGB") == 1


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

    assert report["cover_mass_g"] <= params["checks"]["cover_mass_max"]
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
    assert "COLOUR_RGB('',0.95" in text


def exact_box(shape: cq.Shape) -> tuple[float, ...]:
    """(xmin, ymin, zmin, xmax, ymax, zmax) from the geometry. Shape.BoundingBox() uses a
    triangulation when the shape has one, and the pictures and the drag estimate leave a
    coarse one on every part, which can stick out of a curved surface by millimetres."""
    box = Bnd_Box()
    BRepBndLib.AddOptimal_s(shape.wrapped, box, False, False)
    return box.Get()


def test_canopy_form_is_a_clean_solid(drone: assembly.Drone) -> None:
    """The canopy's outer form for editing in Fusion: one closed solid with few faces,
    from which the canopy is the form minus the form shrunk by the wall."""
    (form,) = drone.forms
    assert form.name == "canopy_form"
    solid = form.solid
    assert solid.isValid()
    assert len(solid.Solids()) == 1
    assert len(solid.Faces()) <= 12
    canopy = next(pt for pt in drone.parts if pt.name == "canopy")
    assert exact_box(solid) == pytest.approx(exact_box(canopy.solid), abs=0.5)


def test_blades_stay_inside_their_discs(drone: assembly.Drone, params: dict[str, Any]) -> None:
    """The drawn propellers: one valid solid each, inside the swept disc the checks use,
    turning alternately (front left clockwise seen from above), and blades shown only
    when params say so."""
    assert params["display"]["prop_blades"] is True
    discs = {pt.name: pt for pt in drone.parts if pt.group == "prop"}
    blades = drone.display
    assert [pt.name for pt in blades] == [f"prop_blades_{i}" for i in range(1, 5)]
    for pt in blades:
        assert pt.solid.isValid() and len(pt.solid.Solids()) == 1
        disc = discs[pt.name.replace("_blades", "")].solid.BoundingBox()
        box = pt.solid.BoundingBox()
        assert box.zmin >= disc.zmin - 1e-6 and box.zmax <= disc.zmax + 1e-6
        r = math.hypot(box.xlen, box.ylen) / 2
        assert r <= params["prop"]["diameter_in"] * 25.4 / 2 * math.sqrt(2) + 1e-6
        assert box.xmin >= disc.xmin - 1e-6 and box.xmax <= disc.xmax + 1e-6
        assert box.ymin >= disc.ymin - 1e-6 and box.ymax <= disc.ymax + 1e-6
    import parts

    cw, ccw = parts.prop_blades(params["prop"], True), parts.prop_blades(params["prop"], False)
    assert cw.Volume() == pytest.approx(ccw.Volume(), rel=1e-6)
    assert cw.Center().y == pytest.approx(-ccw.Center().y, abs=1e-6)
    hidden = {**params, "display": {"prop_blades": False}}
    assert assembly.build(hidden).display == []


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


def test_nothing_above_the_discs_hangs_over_them(
    report: dict[str, Any], params: dict[str, Any]
) -> None:
    """Seen from above, the canopy (and the computer, tray and GPS) stay outside the discs."""
    assert len(report["plan_margins"]) == 4
    for row in report["plan_margins"]:
        assert row["distance_mm"] >= params["checks"]["plan_margin_min"], row


def test_plan_check_catches_a_canopy_over_the_discs(
    drone: assembly.Drone, params: dict[str, Any]
) -> None:
    """A canopy 30 % wider and without its scallops hangs over the discs, which the 3D
    distance check alone does not catch."""
    import checks
    import parts

    can = params["canopy"]
    wide = {**can, "stations": [[x, w * 1.3, h] for x, w, h in can["stations"]]}
    plain = parts.canopy(wide, 0, -62, 10, keep_out=[], keep_out_radius=0)
    canopy = next(pt for pt in drone.parts if pt.name == "canopy")
    z = canopy.solid.BoundingBox().zmin - plain.BoundingBox().zmin
    old = assembly.Part("canopy", "cover", plain.translate((can["x"], 0, z)), 30.0)
    others = [pt for pt in drone.parts if pt.name != "canopy"] + [old]
    rows = checks.plan_margins(others)
    assert min(r["distance_mm"] for r in rows) < 0
    assert all(
        r["distance_mm"] >= params["checks"]["prop_margin_min"]
        for r in checks.prop_margins(others, 5)
        if r["nearest"] == "canopy"
    )


AERO = {
    "speed": 10,
    "air_density": 1.225,
    "pitch_deg": 0,
    "cd_bluff": 1.0,
    "cd_faired": 0.5,
    "cd_cylinder": 0.9,
    "cd_plate": 1.6,
    "wake_factor": 0.35,
    "wake_length": 150,
    "reference_cda_m2": [0.018, 0.023],
}


def test_frontal_area_and_drag_by_hand() -> None:
    """A 10 x 20 x 30 mm box seen from the front is 20 x 30 = 600 mm^2. With a 2 mm cover
    of the same outline in front of it, the cover counts at cd_faired and the box, in its
    wake, at wake_factor."""
    import aero
    import checks

    block = assembly.Part("block", "electronics", cq.Solid.makeBox(10, 20, 30), 1.0)
    cover = assembly.Part("cover", "cover", cq.Solid.makeBox(2, 20, 30, cq.Vector(10, 0, 0)), 1.0)
    assert checks.frontal_area_mm2([block]) == pytest.approx(600, rel=1e-6)
    ae = aero.aero(assembly.Drone([block, cover], [], block.solid, "none"), {"aero": AERO})
    assert ae["frontal_area_before_m2"] == pytest.approx(600e-6)
    assert ae["frontal_area_after_m2"] == pytest.approx(600e-6)
    assert ae["cda_before_m2"] == pytest.approx(600e-6)
    assert ae["cda_after_m2"] == pytest.approx(0.5 * 600e-6 + 0.35 * 600e-6)
    # q = 0.5 * 1.225 * 10^2 = 61.25 Pa
    assert ae["drag_before_n"] == pytest.approx(61.25 * 600e-6, abs=1e-3)


def test_wake_and_hidden_parts_by_hand() -> None:
    """A box 100 mm behind another counts at wake_factor, 200 mm behind (past
    wake_length) fully; a rod inside a box does not count; a pitched-down drone meets the
    air on its underside too."""
    import aero

    def box(name: str, x: float) -> assembly.Part:
        return assembly.Part(
            name, "electronics", cq.Solid.makeBox(10, 20, 30, cq.Vector(x, 0, 0)), 1
        )

    near = aero.drag([box("front", 0), box("behind", -110)], AERO)
    assert near["by_part"]["front"] == pytest.approx(600e-6)
    assert near["by_part"]["behind"] == pytest.approx(0.35 * 600e-6)
    far = aero.drag([box("front", 0), box("behind", -210)], AERO)
    assert far["by_part"]["behind"] == pytest.approx(600e-6)
    rod = assembly.Part(
        "standoff_1",
        "frame",
        cq.Solid.makeCylinder(2, 4, cq.Vector(5, 10, 12), cq.Vector(1, 0, 0)),
        1,
    )
    inside = aero.drag([box("front", 0), rod], AERO)
    assert inside["by_part"]["standoff_1"] == 0
    flat = aero.drag(
        [assembly.Part("bottom_plate", "frame", cq.Solid.makeBox(200, 100, 2), 1)], AERO
    )
    pitched = aero.drag(
        [assembly.Part("bottom_plate", "frame", cq.Solid.makeBox(200, 100, 2), 1)],
        {**AERO, "pitch_deg": 15},
    )
    assert flat["area_m2"] == pytest.approx(200e-6)
    assert pitched["area_m2"] == pytest.approx(
        (200 * math.sin(math.radians(15)) + 2 * math.cos(math.radians(15))) * 100e-6, rel=0.02
    )


def test_drag_classes(drone: assembly.Drone) -> None:
    import aero

    kinds = {pt.name: aero.shape_class(pt) for pt in drone.parts}
    assert kinds["canopy"] == kinds["nose_fairing"] == "faired"
    assert kinds["motor_1"] == kinds["standoff_1"] == kinds["screw_1"] == "cylinder"
    assert kinds["arm_1"] == kinds["bottom_plate"] == kinds["reinforce_front"] == "plate"
    assert kinds["battery"] == kinds["esc"] == "bluff"
    assert kinds["motor_lead_1_1"] == kinds["gps_mast"] == "cylinder"


def test_drag_is_in_the_measured_range(report: dict[str, Any]) -> None:
    """Cd A of the whole body lands within a factor of two of what NASA measured for
    quadcopter bodies without rotors, and the parts add up to the total."""
    ae = report["aero"]
    lo, hi = ae["reference_cda_m2"]
    assert lo / 2 <= ae["cda_after_m2"] <= hi * 2
    assert sum(ae["cda_by_class_m2"].values()) == pytest.approx(ae["cda_after_m2"], abs=1e-5)


def test_cfd_pressure_drag_by_hand(tmp_path: Path) -> None:
    """A 0.1 m cube with pressure 50 m^2/s^2 on its front face only: the force is 0.5 N
    per kg/m^3 of air, Cd A = 50 * 0.01 / (0.5 * 10^2) = 0.01 m^2. The faces wind into the
    body, as OpenFOAM writes a wall patch."""
    points = [(x, y, z) for x in (0, 0.1) for y in (0, 0.1) for z in (0, 0.1)]
    # Point index = 4x + 2y + z (each 0 or 1); the front face is x = 0.1.
    faces = [[4, 5, 7, 6], [0, 2, 3, 1], [0, 1, 5, 4], [2, 6, 7, 3], [0, 4, 6, 2], [1, 3, 7, 5]]
    pressure = [50, 0, 0, 0, 0, 0]
    text = ["# vtk DataFile Version 2.0", "drone", "ASCII", "DATASET POLYDATA"]
    text.append(f"POINTS {len(points)} float")
    text += [" ".join(str(c) for c in pt) for pt in points]
    text.append(f"POLYGONS {len(faces)} {5 * len(faces)}")
    text += [" ".join(str(i) for i in [4, *f]) for f in faces]
    text += [f"CELL_DATA {len(faces)}", "FIELD FieldData 1", f"p 1 {len(faces)} float"]
    text.append(" ".join(str(v) for v in pressure))
    path = tmp_path / "drone_900.vtk"
    path.write_text("\n".join(text) + "\n", encoding="utf-8")
    assert cfd.pressure_cda(path, 10) == pytest.approx(0.01)
    assert cfd.wetted_area(path) == pytest.approx(0.06)
