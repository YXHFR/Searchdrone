"""The fpv7 assembly: every part in place, with its mass, and the exports behind `just cad`.

    uv run --group cad python hardware/cad/fpv7/assembly.py [OUT_DIR]

writes to OUT_DIR (default build/cad/fpv7):

- ``fpv7.step``: the assembly with one named solid per part;
- ``fpv7.stl`` and ``stl/<part>.stl``: the whole drone and every part on its own;
- ``report.md`` and ``report.json``: mass, centre of gravity, inertia, overlaps,
  propeller margins and the camera's field of view (checks.py);
- ``fpv7_views.png``: the drone from three angles (render.py).

Axes: x forward, y left, z up (millimetres). The origin is the frame centre and z = 0 is
the underside of the bottom plate. The same output comes out every time: the STEP header
carries a fixed date, assembly occurrences are numbered from 1 and the parts are always
written in the same order.
"""

from __future__ import annotations

import json
import math
import re
import sys
from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import cadquery as cq
import checks
import parts
import yaml

HERE = Path(__file__).resolve().parent
PARAMS = HERE / "params.yaml"
DEFAULT_OUT = HERE.parents[2] / "build" / "cad" / "fpv7"

# Fixed STEP header date so that two builds give the same bytes.
STEP_DATE = "2000-01-01T00:00:00"
STEP_DATE_RE = re.compile(r"(FILE_NAME\('[^']*',')[^']*(')")
# OCCT numbers assembly occurrences with a counter that lives as long as the process;
# they are renumbered from 1 in file order.
STEP_NAUO_RE = re.compile(r"(NEXT_ASSEMBLY_USAGE_OCCURRENCE\(')\d+(')")

# One colour per group for the PNG. The STEP file has none: OCCT writes colour styles in a
# different order on every run, and the exports must be repeatable.
COLORS: dict[str, tuple[float, float, float]] = {
    "frame": (0.15, 0.15, 0.15),
    "motor": (0.55, 0.55, 0.6),
    "prop": (0.2, 0.45, 0.9),
    "battery": (0.85, 0.65, 0.1),
    "electronics": (0.1, 0.55, 0.25),
    "computer": (0.75, 0.2, 0.2),
    "camera": (0.6, 0.2, 0.7),
    "printed": (0.9, 0.9, 0.85),
}


@dataclass(frozen=True)
class Part:
    """One solid in place, with the mass it stands for (grams)."""

    name: str
    group: str
    solid: cq.Shape
    mass_g: float


@dataclass(frozen=True)
class PointMass:
    """Mass without a solid of its own (cables, screws, straps), at a point in mm."""

    name: str
    mass_g: float
    at: tuple[float, float, float]


@dataclass(frozen=True)
class Drone:
    parts: list[Part]
    point_masses: list[PointMass]
    fov: cq.Shape
    camera_name: str


def load_params(path: Path = PARAMS) -> dict[str, Any]:
    with path.open(encoding="utf-8") as f:
        return yaml.safe_load(f)


def moved(solid: cq.Shape, x: float = 0, y: float = 0, z: float = 0, rz: float = 0) -> cq.Shape:
    """Rotate about the z axis by ``rz`` degrees, then move by (x, y, z)."""
    out = solid.rotate(cq.Vector(0, 0, 0), cq.Vector(0, 0, 1), rz) if rz else solid
    return out.translate(cq.Vector(x, y, z))


def heights(p: dict[str, Any]) -> dict[str, float]:
    """The z levels that the layout is built on (mm)."""
    f = p["frame"]
    arm_bottom = f["bottom_plate"]["thickness"]
    arm_top = arm_bottom + f["arm"]["thickness"]
    top_plate = arm_top + f["standoff"]["height"]
    return {
        "arm_bottom": arm_bottom,
        "arm_top": arm_top,
        "top_plate": top_plate,
        "top_plate_top": top_plate + f["top_plate"]["thickness"],
    }


def build(p: dict[str, Any]) -> Drone:
    f = p["frame"]
    lay = p["layout"]
    z = heights(p)
    out: list[Part] = []

    # Frame: bottom plate, four arms on top of it, standoffs, top plate. The frame's mass
    # is spread over its pieces by volume.
    frame_solids: list[tuple[str, cq.Shape]] = [("bottom_plate", parts.plate(f["bottom_plate"]))]
    motor_angles = [45.0, 135.0, 225.0, 315.0]
    for i, ang in enumerate(motor_angles, start=1):
        frame_solids.append((f"arm_{i}", moved(parts.arm(f), z=z["arm_bottom"], rz=ang)))
    sx, sy = f["standoff"]["pattern_x"] / 2, f["standoff"]["pattern_y"] / 2
    for i, (x, y) in enumerate([(sx, sy), (-sx, sy), (-sx, -sy), (sx, -sy)], start=1):
        frame_solids.append((f"standoff_{i}", moved(parts.standoff(f), x, y, z["arm_top"])))
    frame_solids.append(("top_plate", moved(parts.plate(f["top_plate"]), z=z["top_plate"])))
    frame_volume = sum(s.Volume() for _, s in frame_solids)
    for name, s in frame_solids:
        out.append(Part(name, "frame", s, f["mass"] * s.Volume() / frame_volume))

    # Motors on the arm tips, propeller discs on the motors.
    r = f["wheelbase"] / 2
    m, pr = p["motor"], p["prop"]
    for i, ang in enumerate(motor_angles, start=1):
        x, y = r * math.cos(math.radians(ang)), r * math.sin(math.radians(ang))
        out.append(
            Part(f"motor_{i}", "motor", moved(parts.motor(m), x, y, z["arm_top"]), m["mass"])
        )
        prop_z = z["arm_top"] + m["height"] + pr["gap_above_motor"]
        out.append(Part(f"prop_{i}", "prop", moved(parts.prop_disc(pr), x, y, prop_z), pr["mass"]))

    # Legs under the arms, long enough to stand clear of the battery.
    leg = p["legs"]
    leg_bottom = -(p["battery"]["height"] + leg["ground_clearance"])
    leg_p = {**leg, "height": z["arm_bottom"] - leg_bottom}
    for i, ang in enumerate(motor_angles, start=1):
        x, y = (
            leg["radius"] * math.cos(math.radians(ang)),
            leg["radius"] * math.sin(math.radians(ang)),
        )
        out.append(
            Part(
                f"leg_{i}",
                "printed",
                moved(parts.leg(leg_p), x, y, leg_bottom, rz=ang),
                leg["mass"],
            )
        )

    # Flight stack between the plates: ESC on soft mounts over the arm roots, FC above.
    st = p["stack"]
    esc_z = z["arm_top"] + st["grommet_height"]
    out.append(
        Part(
            "esc",
            "electronics",
            moved(parts.board(st["esc"]), lay["stack_x"], 0, esc_z),
            st["esc"]["mass"],
        )
    )
    fc_z = esc_z + st["esc"]["height"] + st["board_gap"]
    out.append(
        Part(
            "flight_controller",
            "electronics",
            moved(parts.board(st["fc"]), lay["stack_x"], 0, fc_z),
            st["fc"]["mass"],
        )
    )
    for key in ("bec", "receiver"):
        bx, by = lay[key]
        out.append(
            Part(
                key,
                "electronics",
                moved(parts.board(p[key]), bx, by, z["arm_bottom"]),
                p[key]["mass"],
            )
        )

    # On the top plate: Remote ID, the companion computer on a damped tray, the GPS mast.
    tp = z["top_plate_top"]
    rx, ry = lay["remote_id"]
    out.append(
        Part(
            "remote_id",
            "electronics",
            moved(parts.board(p["remote_id"]), rx, ry, tp),
            p["remote_id"]["mass"],
        )
    )
    pi, tray = p["pi5"], p["pi_tray"]
    tray_z = tp + tray["damper_height"]
    out.append(
        Part("pi_tray", "printed", moved(parts.board(tray), lay["pi_x"], 0, tray_z), tray["mass"])
    )
    pcb_z = tray_z + tray["height"] + pi["underside_clearance"]
    out.append(Part("pi5", "computer", moved(parts.board(pi), lay["pi_x"], 0, pcb_z), pi["mass"]))
    # The USB and Ethernet block sits on the Pi's rear edge (-x), the GPIO end faces forward.
    ports = pi["ports"]
    ports_x = lay["pi_x"] - pi["length"] / 2 + ports["length"] / 2 - ports["overhang"]
    out.append(
        Part(
            "pi5_ports",
            "computer",
            moved(parts.board(ports), ports_x, 0, pcb_z + pi["height"]),
            ports["mass"],
        )
    )
    cool = p["cooler"]
    cool_x = lay["pi_x"] + pi["length"] / 2 - cool["offset_from_gpio_end"]
    out.append(
        Part(
            "active_cooler",
            "computer",
            moved(parts.board(cool), cool_x, 0, pcb_z + pi["height"]),
            cool["mass"],
        )
    )
    hat = p["ai_hat"]
    hat_x = lay["pi_x"] + pi["length"] / 2 - hat["length"] / 2
    hat_z = pcb_z + pi["height"] + hat["standoff_height"]
    out.append(Part("ai_hat", "computer", moved(parts.board(hat), hat_x, 0, hat_z), hat["mass"]))
    gps = p["gps"]
    out.append(
        Part(
            "gps_mast",
            "printed",
            moved(parts.mast(gps["mast"]), lay["gps_x"], 0, tp),
            gps["mast"]["mass"],
        )
    )
    out.append(
        Part(
            "gps",
            "electronics",
            moved(parts.board(gps), lay["gps_x"], 0, tp + gps["mast"]["height"]),
            gps["mass"],
        )
    )

    # Under the bottom plate: the battery, and the camera ahead of it, looking straight down.
    b = p["battery"]
    out.append(
        Part(
            "battery",
            "battery",
            moved(parts.board(b), lay["battery_x"], 0, -b["height"]),
            b["mass"],
        )
    )
    cam, mount = p["camera"], p["camera_mount"]
    cam_x = lay["camera_x"]
    out.append(
        Part(
            "camera_mount",
            "printed",
            moved(parts.board(mount), cam_x, 0, -mount["height"]),
            mount["mass"],
        )
    )
    cam_bottom = -mount["height"] - cam["height"]
    # The sensor's long side (horizontal field of view) across the drone, along y.
    out.append(
        Part("camera", "camera", moved(parts.board(cam), cam_x, 0, cam_bottom, rz=90), cam["mass"])
    )
    fov = moved(
        parts.fov_pyramid(cam["hfov_deg"], cam["vfov_deg"], p["checks"]["fov_length"]),
        cam_x,
        0,
        cam_bottom,
        rz=90,
    )

    points = [
        PointMass(
            "wiring_and_screws",
            p["misc"]["mass"],
            (lay["stack_x"], 0.0, (z["arm_top"] + z["top_plate"]) / 2),
        )
    ]
    return Drone(out, points, fov, "camera")


def write_step(drone: Drone, path: Path) -> None:
    assy = cq.Assembly(name="fpv7")
    for part in drone.parts:
        assy.add(part.solid, name=part.name)
    assy.export(str(path), "STEP")
    text = path.read_text(encoding="utf-8")
    text = STEP_DATE_RE.sub(rf"\g<1>{STEP_DATE}\g<2>", text, count=1)
    counter = iter(range(1, len(text)))
    text = STEP_NAUO_RE.sub(lambda m: f"{m[1]}{next(counter)}{m[2]}", text)
    path.write_text(text, encoding="utf-8")


def write_stl(shape: cq.Shape, path: Path, tolerance: float) -> None:
    cq.exporters.export(shape, str(path), "STL", tolerance=tolerance, angularTolerance=0.2)


def export(drone: Drone, p: dict[str, Any], out_dir: Path) -> dict[str, Any]:
    import render  # matplotlib is only needed for the picture

    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "stl").mkdir(exist_ok=True)
    tol = p["checks"]["stl_tolerance"]
    write_step(drone, out_dir / "fpv7.step")
    write_stl(cq.Compound.makeCompound([pt.solid for pt in drone.parts]), out_dir / "fpv7.stl", tol)
    for part in drone.parts:
        write_stl(part.solid, out_dir / "stl" / f"{part.name}.stl", tol)
    report = checks.report(drone, p)
    (out_dir / "report.json").write_text(
        json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    (out_dir / "report.md").write_text(checks.markdown(report), encoding="utf-8")
    render.views(drone, COLORS, out_dir / "fpv7_views.png")
    return report


def main(argv: Sequence[str] | None = None) -> int:
    args = list(sys.argv[1:] if argv is None else argv)
    out_dir = Path(args[0]) if args else DEFAULT_OUT
    p = load_params()
    report = export(build(p), p, out_dir)
    sys.stdout.write(checks.markdown(report))
    sys.stdout.write(f"\nSkrev STEP, STL, rapport och bild till {out_dir}\n")
    return 0 if report["ok"] else 1


if __name__ == "__main__":
    sys.exit(main())
