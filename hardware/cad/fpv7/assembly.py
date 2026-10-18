"""The fpv7 assembly: every part in place, with its mass, and the exports behind `just cad`.

    uv run --group cad python hardware/cad/fpv7/assembly.py [OUT_DIR] [detailed]

writes to OUT_DIR (default build/cad/fpv7):

- ``fpv7.step``: the assembly, a sub-assembly per group (frame, motor, prop, ...) with one
  named, coloured solid per part, for Fusion or any other CAD program (README.md);
- ``step/<part>.step``: every part on its own, in place, for further work in a CAD program,
  and ``step/canopy_form.step``: the canopy's outer form as one closed solid with few faces,
  to edit and shell again;
- with ``detailed``, after vendor.py has downloaded the models:
  ``build/cad/fpv7-detailed/fpv7_detailed.step``, the assembly with the manufacturers'
  Raspberry Pi 5 and Camera Module 3 models (kept apart because it may not be shared);
- ``fpv7.stl`` and ``stl/<part>.stl``: the whole drone and every part on its own (the
  printed parts, such as ``canopy.stl`` and ``nose_fairing.stl``, go to the slicer);
- ``report.md`` and ``report.json``: mass, centre of gravity, inertia, overlaps,
  propeller margins and the camera's field of view (checks.py);
- ``fpv7_views.png`` and ``fpv7_closeup.png``: the drone from three angles and the centre
  of the frame up close (render.py).

Axes: x forward, y left, z up (millimetres). The origin is the frame centre and z = 0 is
the underside of the bottom plate. The same output comes out every time: the STEP header
carries a fixed date, assembly occurrences are numbered from 1, the colour styles are
written in a fixed order and the parts are always written in the same order.
"""

from __future__ import annotations

import itertools
import json
import math
import re
import sys
from collections.abc import Sequence
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import cadquery as cq
import checks
import parts
import yaml

HERE = Path(__file__).resolve().parent
PARAMS = HERE / "params.yaml"
DEFAULT_OUT = HERE.parents[2] / "build" / "cad" / "fpv7"
# Outside DEFAULT_OUT on purpose: CI publishes DEFAULT_OUT, and the detailed STEP holds
# manufacturer models that may not be shared.
DETAILED_OUT = HERE.parents[2] / "build" / "cad" / "fpv7-detailed"

# Fixed STEP header date so that two builds give the same bytes.
STEP_DATE = "2000-01-01T00:00:00"
STEP_DATE_RE = re.compile(r"(FILE_NAME\('[^']*',')[^']*(')")
# OCCT numbers assembly occurrences with a counter that lives as long as the process,
# and wraps the line where the number's length puts it. Each occurrence is rewritten on
# one line, numbered from 1 in file order.
STEP_NAUO_RE = re.compile(r"NEXT_ASSEMBLY_USAGE_OCCURRENCE\('\d+',(.*?)\);", re.DOTALL)
# A colour style: the presentation representation and the styled item that ties a style
# to a solid.
STEP_STYLE_RE = re.compile(
    r"#(?P<mdgpr>\d+)\s*=\s*MECHANICAL_DESIGN_GEOMETRIC_PRESENTATION_REPRESENTATION\(\s*''\s*,"
    r"\s*\(\s*#(?P<item>\d+)\s*\)\s*,\s*#(?P<context>\d+)\s*\);\s*"
    r"#(?P=item)\s*=\s*STYLED_ITEM\(\s*'color'\s*,\s*\(\s*#(?P<psa>\d+)\s*\)\s*,"
    r"\s*#(?P<target>\d+)\s*\);"
)
STEP_ENTITY_RE = re.compile(r"^#(\d+)\s*=\s*([^;]*);", re.MULTILINE)
STEP_STYLE_TYPES = {
    "MECHANICAL_DESIGN_GEOMETRIC_PRESENTATION_REPRESENTATION",
    "STYLED_ITEM",
    "PRESENTATION_STYLE_ASSIGNMENT",
    "SURFACE_STYLE_USAGE",
    "SURFACE_SIDE_STYLE",
    "SURFACE_STYLE_FILL_AREA",
    "FILL_AREA_STYLE",
    "FILL_AREA_STYLE_COLOUR",
    "COLOUR_RGB",
}

# One colour per group, for the STEP files and the pictures.
COLORS: dict[str, tuple[float, float, float]] = {
    "frame": (0.15, 0.15, 0.15),
    "motor": (0.55, 0.55, 0.6),
    "prop": (0.2, 0.45, 0.9),
    "battery": (0.85, 0.65, 0.1),
    "electronics": (0.1, 0.55, 0.25),
    "computer": (0.1, 0.45, 0.55),
    "camera": (0.6, 0.2, 0.7),
    "printed": (0.9, 0.9, 0.85),
    "cover": (0.95, 0.55, 0.15),
    "fastener": (0.7, 0.7, 0.72),
    "cable": (0.8, 0.1, 0.1),
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
    # Shown in the STEP file and the pictures but left out of the checks: the propeller
    # blades (the checks use the swept discs).
    display: list[Part] = field(default_factory=list)
    # Exported on their own only, to edit in a CAD program: the canopy's outer form.
    forms: list[Part] = field(default_factory=list)


def load_params(path: Path = PARAMS) -> dict[str, Any]:
    with path.open(encoding="utf-8") as f:
        return yaml.safe_load(f)


def moved(solid: cq.Shape, x: float = 0, y: float = 0, z: float = 0, rz: float = 0) -> cq.Shape:
    """Rotate about the z axis by ``rz`` degrees, then move by (x, y, z)."""
    out = solid.rotate(cq.Vector(0, 0, 0), cq.Vector(0, 0, 1), rz) if rz else solid
    return out.translate(cq.Vector(x, y, z))


def shell_mass(solid: cq.Shape, p: dict[str, Any], materials: dict[str, float]) -> float:
    """Mass in grams of a printed part: its volume times the density (g/cm^3) of its material."""
    return solid.Volume() / 1000 * materials[p["material"]]


def heights(p: dict[str, Any]) -> dict[str, float]:
    """The z levels that the layout is built on (mm).

    The arms lie on the bottom plate and the reinforcement plates on the arm roots; the
    25 mm standoffs stand on the bottom plate and carry the top plate.
    """
    f = p["frame"]
    bottom_top = f["bottom_plate"]["thickness"]
    arm_top = bottom_top + f["arm"]["thickness"]
    top_plate = bottom_top + f["standoff"]["height"]
    return {
        "bottom_top": bottom_top,
        "arm_bottom": bottom_top,
        "arm_top": arm_top,
        "reinforce_top": arm_top + f["reinforce_plate"]["thickness"],
        "top_plate": top_plate,
        "top_plate_top": top_plate + f["top_plate"]["thickness"],
    }


@dataclass(frozen=True)
class Arm:
    """Where one arm lies: its root screw, its direction (unit vector) and its motor."""

    root: tuple[float, float]
    direction: tuple[float, float]
    motor: tuple[float, float]
    length: float

    @property
    def angle(self) -> float:
        return math.degrees(math.atan2(self.direction[1], self.direction[0]))

    def at(self, s: float, offset: float = 0.0) -> tuple[float, float]:
        """The point ``s`` along the arm from the root screw, ``offset`` to its left."""
        dx, dy = self.direction
        return (self.root[0] + s * dx - offset * dy, self.root[1] + s * dy + offset * dx)


# Motors 1-4: front left, rear left, rear right, front right (x forward, y left).
MOTOR_SIGNS = [(1, 1), (-1, 1), (-1, -1), (1, -1)]
# Turning direction seen from above, as ArduPilot's quad X: the front-left and rear-right
# propellers turn clockwise.
CLOCKWISE = [True, False, True, False]


def arms(p: dict[str, Any]) -> list[Arm]:
    """The four arms of the H frame: from the root screws to the motors."""
    f = p["frame"]
    mx, my = (v / 2 for v in f["motor_spacing"])
    rx, ry = f["arm"]["root_screw"]
    out = []
    for sx, sy in MOTOR_SIGNS:
        root, motor = (sx * rx, sy * ry), (sx * mx, sy * my)
        length = math.dist(root, motor)
        direction = ((motor[0] - root[0]) / length, (motor[1] - root[1]) / length)
        out.append(Arm(root, direction, motor, length))
    return out


def mirror_y(points: list[list[float]]) -> list[tuple[float, float]]:
    """Points [x, y] and their mirror images in y (each once if y = 0)."""
    out: list[tuple[float, float]] = []
    for x, y in points:
        out += [(x, y), (x, -y)] if y else [(x, 0.0)]
    return out


def build(p: dict[str, Any]) -> Drone:
    f = p["frame"]
    lay = p["layout"]
    z = heights(p)
    sc = p["screws"]
    mat = p["materials"]
    out: list[Part] = []
    display: list[Part] = []
    the_arms = arms(p)
    hole = f["hole"]
    a = f["arm"]

    # ---- Frame -----------------------------------------------------------------------
    # Pieces with their material; the kit's mass is spread over them by volume x density.
    frame_pieces: list[tuple[str, cq.Shape, str]] = []
    standoffs = mirror_y(f["standoff"]["positions"])
    outer = [arm.root for arm in the_arms]
    inner = [arm.at(-a["root_hole_gap"]) for arm in the_arms]
    sx, sy = (v / 2 for v in [f["stack_holes"][0]] * 2)
    s2 = f["stack_holes"][1] / 2
    stack = [(lay["stack_x"] + i * sx, j * sy, hole) for i in (1, -1) for j in (1, -1)]
    stack += [(lay["stack_x"] + i * s2, j * s2, 2.2) for i in (1, -1) for j in (1, -1)]
    bottom_holes = [(x, y, hole) for x, y in standoffs + outer + inner] + stack
    frame_pieces.append(
        (
            "bottom_plate",
            parts.plate(
                f["bottom_plate"],
                bottom_holes,
                standoffs + outer + inner,
                sc["m3_flat"],
                countersink_below=True,
            ),
            "carbon",
        )
    )
    for i, arm in enumerate(the_arms, start=1):
        shape = parts.arm(f, arm.length).translate(cq.Vector(-a["root_length"], 0, 0))
        frame_pieces.append(
            (f"arm_{i}", moved(shape, *arm.root, z["arm_bottom"], rz=arm.angle), "carbon")
        )
    for name, side in (("reinforce_front", 1), ("reinforce_rear", -1)):
        holes = [pt for pt in outer + inner if pt[0] * side > 0]
        frame_pieces.append(
            (name, moved(parts.reinforce_plate(f, holes), z=z["arm_top"]), "carbon")
        )
    so = f["standoff"]
    for i, (x, y) in enumerate(standoffs, start=1):
        shape = parts.standoff(so["diameter"], so["bore"], so["height"])
        frame_pieces.append((f"standoff_{i}", moved(shape, x, y, z["bottom_top"]), "aluminium"))
    arm_so = f["arm_standoff"]
    for i, (x, y) in enumerate(outer, start=1):
        shape = parts.standoff(arm_so["diameter"], arm_so["bore"], arm_so["height"])
        frame_pieces.append(
            (f"arm_standoff_{i}", moved(shape, x, y, z["reinforce_top"]), "aluminium")
        )
    cam_x, cam_d = f["top_plate"]["cam_hole"]
    top_holes = [(x, y, hole) for x, y in standoffs] + [(cam_x, 0, cam_d)]
    frame_pieces.append(
        (
            "top_plate",
            moved(
                parts.plate(f["top_plate"], top_holes, standoffs, sc["m3_flat"]), z=z["top_plate"]
            ),
            "carbon",
        )
    )
    # Frame screws: flat heads into the standoffs from above and below, flat heads from
    # below through the arm roots, button heads on the short standoffs, thin nuts on the
    # inner arm screws (under the ESC, so they must stay below it).
    flat8 = parts.screw(sc["m3_flat"], 8, flat=True)
    from_below = flat8.mirror("XY")
    n = itertools.count(1)
    for x, y in standoffs:
        frame_pieces.append((f"screw_{next(n)}", moved(flat8, x, y, z["top_plate_top"]), "steel"))
        frame_pieces.append((f"screw_{next(n)}", moved(from_below, x, y, 0), "steel"))
    clamp = parts.screw(sc["m3_flat"], 12, flat=True).mirror("XY")
    cap = parts.screw(sc["m3_button"], 8)
    for x, y in outer:
        frame_pieces.append((f"screw_{next(n)}", moved(clamp, x, y, 0), "steel"))
        frame_pieces.append(
            (f"screw_{next(n)}", moved(cap, x, y, z["reinforce_top"] + arm_so["height"]), "steel")
        )
    short = parts.screw(sc["m3_flat"], 12, flat=True).mirror("XY")
    nut = parts.nut(*sc["m3_nut"])
    for i, (x, y) in enumerate(inner, start=1):
        frame_pieces.append((f"screw_{next(n)}", moved(short, x, y, 0), "steel"))
        frame_pieces.append((f"nut_{i}", moved(nut, x, y, z["reinforce_top"]), "steel"))
    weights = [s.Volume() * mat[m] for _, s, m in frame_pieces]
    for (name, s, _), w in zip(frame_pieces, weights, strict=True):
        group = "frame" if name.split("_")[0] not in ("screw", "nut") else "fastener"
        out.append(Part(name, group, s, f["mass"] * w / sum(weights)))

    # ---- Motors and propellers --------------------------------------------------------
    m, pr = p["motor"], p["prop"]
    motor_shape = parts.motor(m, 3.0)
    motor_screw = parts.screw(sc["m3_button"], sc["motor_length"]).mirror("XY")
    disc = parts.prop_disc(pr)
    prop_z = z["arm_top"] + m["height"] + pr["gap_above_motor"]
    for i, arm in enumerate(the_arms, start=1):
        x, y = arm.motor
        out.append(
            Part(
                f"motor_{i}",
                "motor",
                moved(motor_shape, x, y, z["arm_top"], rz=arm.angle),
                m["mass"],
            )
        )
        for k, (px, py) in enumerate(parts.motor_screw_points(m), start=1):
            shape = moved(motor_screw, *_rot((px, py), arm.angle, (x, y)), z["arm_bottom"])
            out.append(
                Part(
                    f"motor_screw_{i}_{k}", "fastener", shape, shape.Volume() / 1000 * mat["steel"]
                )
            )
        out.append(Part(f"prop_{i}", "prop", moved(disc, x, y, prop_z), pr["mass"]))
        blades = parts.prop_blades(pr, CLOCKWISE[i - 1])
        display.append(Part(f"prop_blades_{i}", "prop", moved(blades, x, y, prop_z), pr["mass"]))

    # ---- Legs under the arms, long enough to stand clear of the battery -----------------
    leg = p["legs"]
    leg_bottom = -(p["battery"]["height"] + leg["ground_clearance"])
    leg_p = {**leg, "height": z["arm_bottom"] - leg_bottom}
    for i, arm in enumerate(the_arms, start=1):
        x, y = arm.at(arm.length - leg["from_motor"])
        out.append(
            Part(
                f"leg_{i}",
                "printed",
                moved(parts.leg(leg_p), x, y, leg_bottom, rz=arm.angle),
                leg["mass"],
            )
        )

    # ---- Between the plates: the flight stack, BEC and receiver -------------------------
    st = p["stack"]
    esc_z = z["bottom_top"] + st["base_height"]
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
                moved(parts.board(p[key]), bx, by, z["bottom_top"]),
                p[key]["mass"],
            )
        )

    # ---- On the top plate: Remote ID, the companion computer on its tray, the GPS mast ---
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
    can = p["canopy"]
    tray_z = tp + tray["damper_height"]
    boss_x = [x - lay["pi_x"] for x in can["screws"]["x"]]
    out.append(
        Part(
            "pi_tray",
            "printed",
            moved(parts.tray(tray, boss_x), lay["pi_x"], 0, tray_z),
            tray["mass"],
        )
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
    gps_z = tp + gps["mast"]["height"]
    out.append(
        Part("gps", "electronics", moved(parts.board(gps), lay["gps_x"], 0, gps_z), gps["mass"])
    )

    # ---- Under the bottom plate: battery, straps, XT60, camera ---------------------------
    b = p["battery"]
    out.append(
        Part(
            "battery",
            "battery",
            moved(parts.board(b), lay["battery_x"], 0, -b["height"]),
            b["mass"],
        )
    )
    for i, x in enumerate(p["strap"]["x"], start=1):
        out.append(
            Part(f"strap_{i}", "printed", moved(parts.strap(p["strap"], b), x), p["strap"]["mass"])
        )
    xt = p["xt60"]
    xt_x, xt_z = lay["xt60"]
    xt_shape = parts.xt60_pair(xt)
    out.append(Part("xt60", "electronics", moved(xt_shape, xt_x, 0, xt_z), xt["mass"]))
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

    # ---- Cables (approximate routes, as tubes) ------------------------------------------
    out += cables(p, the_arms, z, pcb_z, tray_z)

    # ---- Printed covers ------------------------------------------------------------------
    # The top canopy's rim stands clear above the propeller discs; it is held by four M2
    # screws through its side walls into the tray's bosses. The nose fairing hangs under
    # the frame ahead of the battery, with a window cut where the camera looks through it.
    prop_top = prop_z + pr["swept_height"]
    rim = prop_top + can["rim_clearance"]
    screw_z = tray_z + tray["height"] + tray["boss"][2] / 2 + 1
    # Each screw head sits on the canopy's outer wall: the widest point of the form over
    # the head's footprint.
    form = parts.canopy_form(can)
    screw_points = []
    for x in can["screws"]["x"]:
        probe = parts.box(4, 200, 4).translate(cq.Vector(x, 0, screw_z - rim - 2))
        half_w = form.intersect(probe).BoundingBox().ymax + 0.05
        screw_points += [(x, side * half_w, screw_z - rim) for side in (1, -1)]
    canopy_solid = parts.canopy(
        can,
        vent_x=hat_x,
        mast_x=lay["gps_x"],
        mast_diameter=gps["mast"]["diameter"],
        keep_out=[arm.motor for arm in the_arms],
        keep_out_radius=pr["diameter_in"] * 25.4 / 2 + can["prop_keep_out"],
        screw_points=screw_points,
    )
    out.append(
        Part("canopy", "cover", moved(canopy_solid, z=rim), shell_mass(canopy_solid, can, mat))
    )
    m2 = parts.screw(sc["m2_button"], can["screws"]["length"])
    for i, (x, y, zs) in enumerate(screw_points, start=1):
        side = 1 if y > 0 else -1
        # Head outside the wall, shank along -y (left side) or +y (right side).
        shape = m2.rotate(cq.Vector(0, 0, 0), cq.Vector(1, 0, 0), -90 * side)
        shape = moved(shape, x, y, zs + rim)
        out.append(
            Part(f"canopy_screw_{i}", "fastener", shape, shape.Volume() / 1000 * mat["steel"])
        )
    nf = p["nose_fairing"]
    extra = nf["window_extra_deg"]
    window = moved(
        parts.fov_pyramid(
            cam["hfov_deg"] + 2 * extra, cam["vfov_deg"] + 2 * extra, p["checks"]["fov_length"]
        ),
        cam_x,
        0,
        cam_bottom,
        rz=90,
    )
    fairing = parts.cut(moved(parts.nose_fairing(nf), nf["x_tip"]), window)
    out.append(Part("nose_fairing", "cover", fairing, shell_mass(fairing, nf, mat)))

    points = [
        PointMass(
            "misc", p["misc"]["mass"], (lay["stack_x"], 0.0, (z["arm_top"] + z["top_plate"]) / 2)
        )
    ]
    if not p["display"]["prop_blades"]:
        display = []
    forms = [Part("canopy_form", "cover", moved(form, z=rim), 0.0)]
    return Drone(out, points, fov, "camera", display, forms)


def _rot(pt: tuple[float, float], angle: float, at: tuple[float, float]) -> tuple[float, float]:
    """``pt`` turned by ``angle`` degrees about the origin, then moved to ``at``."""
    c, s = math.cos(math.radians(angle)), math.sin(math.radians(angle))
    return (at[0] + c * pt[0] - s * pt[1], at[1] + s * pt[0] + c * pt[1])


def cables(
    p: dict[str, Any], the_arms: list[Arm], z: dict[str, float], pcb_z: float, tray_z: float
) -> list[Part]:
    """The cables as tubes along approximate routes, with their mass from their length.

    Motor leads run on top of each arm and rise beside the ESC; the ESC's battery
    pigtail runs from the XT60 behind the battery round the rear of the frame and
    forward between the plates; the BEC feeds the Pi through the top plate's front hole;
    the camera's ribbon runs up in front of the frame and back under the canopy rim; the
    GPS lead runs down the mast and through a top-plate window to the flight controller;
    the receiver's two antennas leave the rear of the frame in a V, pointing down.
    """
    w = p["wiring"]
    lay = p["layout"]
    a = p["frame"]["arm"]
    out: list[Part] = []

    def add(
        name: str,
        pts: list[tuple[float, float, float]],
        kind: str,
        offsets: list[float],
        plane: tuple[float, float] = (0.0, 1.0),
    ) -> None:
        spec = w[kind]
        for j, off in enumerate(offsets, start=1):
            shifted = [(x + off * plane[0], y + off * plane[1], zz) for x, y, zz in pts]
            shape = parts.tube(shifted, spec["diameter"])
            mass = spec["g_per_m"] * parts.path_length(shifted) / 1000
            suffix = f"_{j}" if len(offsets) > 1 else ""
            out.append(Part(f"{name}{suffix}", "cable", shape, mass))

    # Motor leads: three 18 AWG wires side by side along the arm top, then up.
    mw = w["motor"]
    lead_z = z["arm_top"] + mw["diameter"] / 2
    spread = [(k - (mw["count"] - 1) / 2) * (mw["diameter"] + 0.1) for k in range(mw["count"])]
    for i, arm in enumerate(the_arms, start=1):
        start = arm.at(arm.length - p["motor"]["base_diameter"] / 2 - 1.5)
        bend = arm.at(a["root_length"] - 2)
        dx, dy = arm.direction
        add(
            f"motor_lead_{i}",
            [(*start, lead_z), (*bend, lead_z), (*bend, lead_z + 6)],
            "motor",
            spread,
            plane=(-dy, dx),
        )

    # Battery side: the battery lead into the XT60, the ESC pigtail round the rear.
    bw = w["battery"]
    xt_x, xt_z = lay["xt60"]
    xt_len = 2 * p["xt60"]["length"] - 9
    gap = bw["diameter"] / 2 + 1.35
    batt_rear = lay["battery_x"] - p["battery"]["length"] / 2
    add(
        "battery_lead", [(batt_rear, 0, xt_z), (xt_x + xt_len / 2, 0, xt_z)], "battery", [-gap, gap]
    )
    rear = xt_x - xt_len / 2
    riser_x = rear - 4
    pig_z = 14.0
    esc_rear = lay["stack_x"] - p["stack"]["esc"]["length"] / 2 - 0.5
    add(
        "esc_pigtail",
        [(rear, 0, xt_z), (riser_x, 0, xt_z), (riser_x, 0, pig_z), (esc_rear, 0, pig_z)],
        "battery",
        [-gap, gap],
    )
    # out[-2:] are the pigtail wires; the battery lead's mass is in the battery's.
    for k, part in enumerate(out):
        if part.name.startswith("battery_lead"):
            out[k] = Part(part.name, part.group, part.solid, 0.0)

    # BEC to the Pi: up through the top plate's front hole, back under the tray's nose.
    pw = w["power"]
    cam_hole_x = p["frame"]["top_plate"]["cam_hole"][0]
    bec_top = z["bottom_top"] + p["bec"]["height"] + pw["diameter"] + 0.5
    pz = tray_z - 1.6
    tray_front = lay["pi_x"] + p["pi_tray"]["length"] / 2 + 2
    add(
        "pi_power",
        [
            (cam_hole_x - 4, 0, bec_top),
            (cam_hole_x, 0, bec_top),
            (cam_hole_x, 0, pz),
            (tray_front, 0, pz),
        ],
        "power",
        [-1.0, 1.0],
    )

    # Camera ribbon: from the camera mount's front, up ahead of the frame, back under the
    # canopy rim to the Pi.
    rb = w["ribbon"]
    front = lay["camera_x"] + p["camera_mount"]["length"] / 2 + 0.2
    up_x = front + 3
    ribbon_z = tray_z + p["pi_tray"]["height"] - 1.5
    pts = [
        (front, 0.0, -3.0),
        (up_x, 0.0, -3.0),
        (up_x, 0.0, ribbon_z + 2),
        (tray_front, 0.0, ribbon_z + 2),
    ]
    shape = parts.ribbon(pts, rb["width"], rb["thickness"])
    out.append(Part("camera_ribbon", "cable", shape, rb["g_per_m"] * parts.path_length(pts) / 1000))

    # GPS lead: down beside the mast, through the rear top-plate window, to the FC.
    gw = w["gps"]
    gx = lay["gps_x"]
    mast_r = p["gps"]["mast"]["diameter"] / 2
    gy = mast_r + gw["diameter"] / 2 + 0.05
    gps_bottom = z["top_plate_top"] + p["gps"]["mast"]["height"] - 0.5
    over_foot = z["top_plate_top"] + p["gps"]["mast"]["foot"][1] + gw["diameter"] / 2 + 1
    fc_rear = lay["stack_x"] - p["stack"]["fc"]["length"] / 2 - 1.5
    fc_mid = z["bottom_top"] + p["stack"]["base_height"] + p["stack"]["esc"]["height"] + 6
    window_x = p["frame"]["top_plate"]["windows"][2][0] + 6
    add(
        "gps_lead",
        [
            (gx, gy, gps_bottom),
            (gx, gy, over_foot),
            (gx + 8, gy, over_foot),
            (window_x, 3, over_foot),
            (window_x, 3, fc_mid),
            (fc_rear, 3, fc_mid),
        ],
        "gps",
        [0.0],
    )

    # Receiver antennas: thin coax out of the rear between the plates, then the active
    # element pointing back and down, the two in a V.
    an = p["antenna"]
    rx, ry = lay["receiver"]
    rz = z["bottom_top"] + p["receiver"]["height"] / 2
    rear_rx = rx - p["receiver"]["length"] / 2
    ext = p["frame"]["bottom_plate"]["outline"][-1][0] - 6
    spread_ = math.radians(an["spread_deg"])
    droop = math.radians(an["droop_deg"])
    for i, side in enumerate((1, -1), start=1):
        start = (rear_rx - 1.5, ry + side * 4, rz)
        exit_ = (ext, ry + side * 12, rz)
        direction = (
            -math.cos(spread_) * math.cos(droop),
            side * math.sin(spread_) * math.cos(droop),
            -math.sin(droop),
        )
        shape = parts.antenna(an, start, exit_, direction)
        out.append(Part(f"antenna_{i}", "cable", shape, an["mass"]))
    return out


def normalise_step(path: Path) -> None:
    """Make a STEP file the same on every build: fixed header date, assembly occurrences
    numbered from 1, each on one line, and the colour styles in a fixed order."""
    text = path.read_text(encoding="utf-8")
    text = STEP_DATE_RE.sub(rf"\g<1>{STEP_DATE}\g<2>", text, count=1)
    counter = itertools.count(1)

    def renumber(m: re.Match[str]) -> str:
        rest = re.sub(r"\s*\n\s*", "", m[1])
        return f"NEXT_ASSEMBLY_USAGE_OCCURRENCE('{next(counter)}',{rest});"

    text = STEP_NAUO_RE.sub(renumber, text)
    path.write_text(_sort_styles(text), encoding="utf-8")


def _sort_styles(text: str) -> str:
    """OCCT writes the colour styles at the end of the file, one per solid, in an order
    that changes from run to run. Write them again in order of the solid's entity number,
    numbered on from where they start, each colour defined once at its first use."""
    first = STEP_STYLE_RE.search(text)
    if first is None:
        return text
    start, end = first.start(), text.index("ENDSEC;", first.start())
    block = dict(STEP_ENTITY_RE.findall(text[start:end]))
    if any(body.split("(")[0].strip() not in STEP_STYLE_TYPES for body in block.values()):
        return text  # something else in the block: leave it as written

    def colour(ref: str) -> str:
        """The COLOUR_RGB arguments that a style leads to."""
        todo = [ref]
        while todo:
            body = block[todo.pop()]
            if body.startswith("COLOUR_RGB"):
                return re.sub(r"\s+", "", body)[len("COLOUR_RGB") :]
            todo += re.findall(r"#(\d+)", body)
        raise ValueError(f"no colour for style #{ref}")

    styles = sorted(
        (int(m["target"]), int(m["context"]), colour(m["psa"]))
        for m in STEP_STYLE_RE.finditer(text, start, end)
    )
    n = itertools.count(int(first["mdgpr"]))
    defined: dict[str, int] = {}
    lines = []
    for target, context, rgb in styles:
        mdgpr, item, psa, usage, side, area, fill = (next(n) for _ in range(7))
        lines += [
            f"#{mdgpr} = MECHANICAL_DESIGN_GEOMETRIC_PRESENTATION_REPRESENTATION("
            f"'',(#{item}),#{context});",
            f"#{item} = STYLED_ITEM('color',(#{psa}),#{target});",
            f"#{psa} = PRESENTATION_STYLE_ASSIGNMENT((#{usage}));",
            f"#{usage} = SURFACE_STYLE_USAGE(.BOTH.,#{side});",
            f"#{side} = SURFACE_SIDE_STYLE('',(#{area}));",
            f"#{area} = SURFACE_STYLE_FILL_AREA(#{fill});",
        ]
        colour_ref = next(n)
        lines += [
            f"#{fill} = FILL_AREA_STYLE('',(#{colour_ref}));",
            f"#{colour_ref} = FILL_AREA_STYLE_COLOUR('',#{defined.get(rgb) or colour_ref + 1});",
        ]
        if rgb not in defined:
            defined[rgb] = next(n)
            lines.append(f"#{defined[rgb]} = COLOUR_RGB{rgb};")
    return text[:start] + "\n".join(lines) + "\n" + text[end:]


def _assembly(
    name: str, items: list[Part], colors: dict[str, tuple[float, float, float]]
) -> cq.Assembly:
    """An assembly with one sub-assembly per group (in the order of first appearance),
    each holding its parts as named, coloured solids."""
    assy = cq.Assembly(name=name)
    groups: dict[str, cq.Assembly] = {}
    for part in items:
        group = (
            part.group
            if part.group != "prop" or part.name.startswith("prop_blades")
            else "prop_disc"
        )
        if group not in groups:
            groups[group] = cq.Assembly(name=group)
        groups[group].add(part.solid, name=part.name, color=cq.Color(*colors[part.group]))
    for group, sub in groups.items():
        assy.add(sub, name=group)
    return assy


def write_step(drone: Drone, path: Path) -> None:
    """The whole drone: a sub-assembly per group with a coloured solid per part. The
    propeller blades are in ``prop``; the swept discs that the checks use are in
    ``prop_disc``."""
    _assembly("fpv7", drone.parts + drone.display, COLORS).export(str(path), "STEP")
    normalise_step(path)


def write_part_step(part: Part, path: Path) -> None:
    """One part on its own, in place in the drone's coordinates, named after the part."""
    # A named root with the solid as its child: a root that holds the solid itself gets an
    # OCCT name with a process-wide counter in it.
    assy = cq.Assembly(name=f"fpv7_{part.name}")
    assy.add(part.solid, name=part.name, color=cq.Color(*COLORS[part.group]))
    assy.export(str(path), "STEP")
    normalise_step(path)


def write_stl(shape: cq.Shape, path: Path, tolerance: float) -> None:
    cq.exporters.export(shape, str(path), "STL", tolerance=tolerance, angularTolerance=0.2)


def export(drone: Drone, p: dict[str, Any], out_dir: Path) -> dict[str, Any]:
    import render  # matplotlib is only needed for the picture

    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "stl").mkdir(exist_ok=True)
    (out_dir / "step").mkdir(exist_ok=True)
    tol = p["checks"]["stl_tolerance"]
    write_step(drone, out_dir / "fpv7.step")
    write_stl(cq.Compound.makeCompound([pt.solid for pt in drone.parts]), out_dir / "fpv7.stl", tol)
    for part in drone.parts + drone.display:
        write_stl(part.solid, out_dir / "stl" / f"{part.name}.stl", tol)
    for part in drone.parts + drone.display + drone.forms:
        write_part_step(part, out_dir / "step" / f"{part.name}.step")
    report = checks.report(drone, p)
    (out_dir / "report.json").write_text(
        json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    (out_dir / "report.md").write_text(checks.markdown(report), encoding="utf-8")
    render.views(drone, COLORS, out_dir / "fpv7_views.png")
    return report


def write_detailed_step(drone: Drone, path: Path) -> None:
    """The assembly with the manufacturer models from vendor.yaml in place of the
    envelope parts they replace (vendor.py must have downloaded them)."""
    import vendor

    by_name = {pt.name: pt for pt in drone.parts}
    replaced = {name for m in vendor.models() for name in m["replaces"]}
    kept = [pt for pt in drone.parts + drone.display if pt.name not in replaced]
    assy = _assembly("fpv7_detailed", kept, COLORS)
    for m in vendor.models():
        assy.add(vendor.placed(m, by_name[m["anchor"]].solid), name=m["name"])
    assy.export(str(path), "STEP")
    normalise_step(path)


def main(argv: Sequence[str] | None = None) -> int:
    args = list(sys.argv[1:] if argv is None else argv)
    detailed = "detailed" in args
    paths = [a for a in args if a != "detailed"]
    out_dir = Path(paths[0]) if paths else DEFAULT_OUT
    p = load_params()
    drone = build(p)
    report = export(drone, p, out_dir)
    if detailed:
        DETAILED_OUT.mkdir(parents=True, exist_ok=True)
        write_detailed_step(drone, DETAILED_OUT / "fpv7_detailed.step")
    sys.stdout.write(checks.markdown(report))
    sys.stdout.write(f"\nSkrev STEP, STL, rapport och bild till {out_dir}\n")
    return 0 if report["ok"] else 1


if __name__ == "__main__":
    sys.exit(main())
