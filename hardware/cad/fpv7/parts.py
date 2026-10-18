"""Solids for the fpv7 parts, each built around its own origin.

Every function takes the part's section of params.yaml (millimetres) and returns a
CadQuery solid. Boards and boxes sit on z = 0 and are centred in x and y; assembly.py
moves them into place. Frame plates, arms, motors, screws, cables and covers are drawn
in some detail; circuit boards stay boxes as tall as their tallest component, and a
propeller is checked as the disc its blades sweep (the blades are drawn for the pictures
and the Fusion file).

Boolean operations run single-threaded (``cut``, ``fuse``): CadQuery runs them in
parallel by default, and the result then differs in the last digits from run to run;
the exports must be the same on every build.
"""

from __future__ import annotations

import itertools
import math
from collections.abc import Sequence
from typing import Any

import cadquery as cq
from OCP.BRepAlgoAPI import BRepAlgoAPI_BooleanOperation, BRepAlgoAPI_Cut, BRepAlgoAPI_Fuse
from OCP.TopTools import TopTools_ListOfShape

Params = dict[str, Any]
Point2 = tuple[float, float]
Point3 = tuple[float, float, float]


def _boolean(
    op: BRepAlgoAPI_BooleanOperation, shape: cq.Shape, tools: Sequence[cq.Shape]
) -> cq.Shape:
    args, others = TopTools_ListOfShape(), TopTools_ListOfShape()
    args.Append(shape.wrapped)
    for t in tools:
        others.Append(t.wrapped)
    op.SetArguments(args)
    op.SetTools(others)
    op.SetRunParallel(False)
    op.Build()
    return cq.Shape.cast(op.Shape())


def _single(shape: cq.Shape) -> cq.Shape:
    """A compound that holds one solid becomes that solid."""
    solids = shape.Solids()
    return solids[0] if len(solids) == 1 else shape


def cut(shape: cq.Shape, *tools: cq.Shape) -> cq.Shape:
    """``shape`` minus ``tools``, single-threaded."""
    return _single(_boolean(BRepAlgoAPI_Cut(), shape, tools)) if tools else shape


def fuse(shape: cq.Shape, *tools: cq.Shape) -> cq.Shape:
    """``shape`` joined with ``tools`` into one solid, single-threaded, with the seams
    between faces of the same surface removed."""
    if not tools:
        return shape
    return _single(_boolean(BRepAlgoAPI_Fuse(), shape, tools).clean())


def box(length: float, width: float, height: float) -> cq.Solid:
    """A box from z = 0 to z = height, centred in x and y."""
    return cq.Solid.makeBox(length, width, height, cq.Vector(-length / 2, -width / 2, 0))


def cylinder(diameter: float, height: float) -> cq.Solid:
    """A cylinder from z = 0 to z = height on the z axis."""
    return cq.Solid.makeCylinder(diameter / 2, height)


def rounded_box(length: float, width: float, height: float, radius: float) -> cq.Shape:
    """A box from z = 0 to height, centred in x and y, with rounded vertical edges."""
    r = min(radius, length / 2 - 0.01, width / 2 - 0.01)
    wp = cq.Workplane("XY").rect(length, width).extrude(height)
    return wp.edges("|Z").fillet(r).val() if r > 0 else wp.val()  # type: ignore[return-value]


def mirrored_outline(half: Sequence[Sequence[float]]) -> list[Point2]:
    """A closed outline from its left half (y >= 0, from the front on the centre line to
    the rear on the centre line), mirrored to the right."""
    left = [(float(x), float(y)) for x, y in half]
    right = [(x, -y) for x, y in reversed(left) if y != 0]
    return left + right


def outline_solid(points: Sequence[Point2], thickness: float, radius: float) -> cq.Shape:
    """A plate cut out along a closed outline, with its corners rounded by ``radius``."""
    sketch = cq.Sketch().polygon([*points, points[0]])
    if radius > 0:
        sketch = sketch.vertices().fillet(radius)
    return cq.Workplane("XY").placeSketch(sketch).extrude(thickness).val()  # type: ignore[return-value]


def window_cutters(windows: Sequence[Sequence[float]], height: float) -> list[cq.Shape]:
    """Rounded rectangular windows [x, y, length, width]; one off the centre line is
    mirrored to the other side."""
    out = []
    for x, y, length, width in windows:
        for side in (1, -1) if y else (1,):
            out.append(
                rounded_box(length, width, height + 2, width / 2 - 0.01).translate(
                    cq.Vector(x, side * y, -1)
                )
            )
    return out


def hole_cutters(holes: Sequence[Sequence[float]], height: float) -> list[cq.Shape]:
    """Round holes [x, y, diameter] through a plate of the given height."""
    return [cylinder(d, height + 2).translate(cq.Vector(x, y, -1)) for x, y, d in holes]


def countersink_cutters(
    points: Sequence[Point2], head: Sequence[float], z: float, from_below: bool
) -> list[cq.Shape]:
    """90 degree countersinks for flat-head screws in a face at height ``z``, for screws
    coming from above (or from below, ``from_below``)."""
    r = head[0] / 2 + 0.1
    out = []
    for x, y in points:
        if from_below:
            cone = cq.Solid.makeCone(
                0, r + 0.5, r + 0.5, cq.Vector(x, y, z + r), cq.Vector(0, 0, -1)
            )
        else:
            cone = cq.Solid.makeCone(0, r + 0.5, r + 0.5, cq.Vector(x, y, z - r))
        out.append(cone)
    return out


def plate(
    p: Params,
    holes: Sequence[Sequence[float]] = (),
    countersinks: Sequence[Point2] = (),
    head: Sequence[float] = (6.72, 1.86),
    countersink_below: bool = False,
) -> cq.Shape:
    """A frame plate: the outline from params, the lightening windows, the holes and the
    countersinks for the flat-head screws (from above, or from below for the bottom plate)."""
    t = p["thickness"]
    body = outline_solid(mirrored_outline(p["outline"]), t, p["corner_radius"])
    tools = window_cutters(p.get("windows", []), t) + hole_cutters(holes, t)
    tools += countersink_cutters(
        countersinks, head, 0 if countersink_below else t, countersink_below
    )
    return cut(body, *tools)


def arm(frame: Params, length: float) -> cq.Shape:
    """One arm along +x: the root end at x = 0, the outer arm screw ``root_length`` further
    out, the motor axis ``length`` past that and the spade tip ``tip_length`` past the motor.

    A bar of constant width with a spade around the motor: a bore, four slots that take
    both motor patterns (``motor_holes``, on circles) and a teardrop window in the tip.
    """
    a = frame["arm"]
    w, tw = a["width"] / 2, a["tip_width"] / 2
    m = a["root_length"] + length
    end = m + a["tip_length"]
    outline = [
        (end, 2.5),
        (m + 14, 7.0),
        (m + 5, tw),
        (m - 12, tw),
        (m - 22, w),
        (2.5, w),
        (0.0, w - 2.5),
        (0.0, -w + 2.5),
        (2.5, -w),
        (m - 22, -w),
        (m - 12, -tw),
        (m + 5, -tw),
        (m + 14, -7.0),
        (end, -2.5),
    ]
    body = outline_solid(outline, a["thickness"], 2.0)
    t = a["thickness"]
    tools: list[cq.Shape] = [cylinder(a["motor_bore"], t + 2).translate(cq.Vector(m, 0, -1))]
    r_in, r_out = (d / 2 for d in a["motor_holes"])
    hole = frame["hole"]
    for k in range(4):
        ang = math.radians(45 + 90 * k)
        mid = (r_in + r_out) / 2
        slot = (
            cq.Workplane("XY")
            .center(m + mid * math.cos(ang), mid * math.sin(ang))
            .slot2D(r_out - r_in + hole, hole, math.degrees(ang))
            .extrude(t + 2)
            .val()
        )
        tools.append(slot.translate(cq.Vector(0, 0, -1)))
    lo, hi, ww = a["window"]
    tools.append(
        cq.Workplane("XY")
        .center(m + (lo + hi) / 2, 0)
        .slot2D(hi - lo, ww, 0)
        .extrude(t + 2)
        .val()
        .translate(cq.Vector(0, 0, -1))
    )
    root = a["root_length"]
    tools += hole_cutters([(root, 0, hole), (root - a["root_hole_gap"], 0, hole)], t)
    return cut(body, *tools)


def reinforce_plate(frame: Params, holes: Sequence[Point2]) -> cq.Shape:
    """The 2 mm plate that clamps the two arm roots of one side (front or rear) onto the
    bottom plate: a pad round both screws of each arm (``holes``, two per arm, in the
    plate's coordinates) and a bar of width 2 ``waist`` between the two pads."""
    r = frame["reinforce_plate"]
    pad, t = r["pad"], r["thickness"]
    left = sorted(h for h in holes if h[1] > 0)
    right = sorted(h for h in holes if h[1] < 0)
    pieces: list[cq.Shape] = []
    for (x1, y1), (x2, y2) in (left, right):
        length = math.dist((x1, y1), (x2, y2))
        angle = math.degrees(math.atan2(y2 - y1, x2 - x1))
        stadium = (
            cq.Workplane("XY")
            .slot2D(length + 2 * pad, 2 * pad, angle)
            .extrude(t)
            .val()
            .translate(cq.Vector((x1 + x2) / 2, (y1 + y2) / 2, 0))
        )
        pieces.append(stadium)  # type: ignore[arg-type]
    bar_x = sum(x for x, _ in holes) / len(holes)
    span = 2 * min(abs(y) for _, y in holes)
    pieces.append(box(2 * r["waist"], span, t).translate(cq.Vector(bar_x, 0, 0)))
    body = fuse(*pieces)
    return cut(body, *hole_cutters([(x, y, frame["hole"]) for x, y in holes], t))


def standoff(diameter: float, bore: float, height: float) -> cq.Shape:
    """A round aluminium standoff with a through bore for the screws."""
    return cut(
        cylinder(diameter, height), cylinder(bore, height + 2).translate(cq.Vector(0, 0, -1))
    )


def screw(head: Sequence[float], length: float, flat: bool = False) -> cq.Shape:
    """A screw standing upright with its head on top: the head's top at z = 0 for a flat
    (countersunk) head, its underside at z = 0 for a button head. The shank (2.9 mm for M3,
    1.9 mm for M2: just inside the clearance holes and standoff bores) points down; a
    flat-head screw's length includes the head."""
    dk, k = head
    shank_d = 2.9 if dk > 4.5 else 1.9
    if flat:
        # 90 degree countersunk head from z = 0 down to where it meets the shank.
        depth = (dk - shank_d) / 2
        cone = cq.Solid.makeCone(shank_d / 2, dk / 2, depth).translate(cq.Vector(0, 0, -depth))
        shank = cylinder(shank_d, length - depth).translate(cq.Vector(0, 0, -length))
        return fuse(cone, shank)
    top = cq.Workplane("XY").circle(dk / 2).extrude(k).edges(">Z").fillet(k * 0.6).val()
    return fuse(top, cylinder(shank_d, length).translate(cq.Vector(0, 0, -length)))  # type: ignore[arg-type]


def nut(diameter: float, height: float) -> cq.Shape:
    """A hexagon nut (across corners ``diameter``) on z = 0 with a 3 mm bore."""
    hexagon = cq.Workplane("XY").polygon(6, diameter).extrude(height).val()
    return cut(hexagon, cylinder(3.0, height + 2).translate(cq.Vector(0, 0, -1)))


def motor(p: Params, screw_holes: float) -> cq.Shape:
    """The outrunner on z = 0: the stator mount, the bell with chamfered edges, the shaft
    and the prop nut on top, with four screw holes on ``mount_circle`` in the base."""
    r_base, r_bell = p["base_diameter"] / 2, p["diameter"] / 2
    hb, h = p["base_height"], p["height"]
    profile = [
        (0, 0),
        (r_base, 0),
        (r_base, hb - 1),
        (r_base + 1, hb),
        (r_bell - 1, hb),
        (r_bell, hb + 1),
        (r_bell, h - 2),
        (r_bell - 2, h),
        (0, h),
    ]
    body = cq.Workplane("XZ").polyline(profile).close().revolve(360, (0, 0, 0), (0, 1, 0)).val()
    shaft_d, shaft_l = p["shaft"]
    nut_d, nut_h = p["nut"]
    shaft = cylinder(shaft_d, shaft_l).translate(cq.Vector(0, 0, h))
    lock = (
        cq.Workplane("XY")
        .polygon(6, nut_d)
        .extrude(nut_h)
        .val()
        .translate(cq.Vector(0, 0, h + shaft_l - nut_h - 1))
    )
    holes = [
        cylinder(screw_holes, 4).translate(
            cq.Vector(
                p["mount_circle"] / 2 * math.cos(math.radians(45 + 90 * k)),
                p["mount_circle"] / 2 * math.sin(math.radians(45 + 90 * k)),
                -0.01,
            )
        )
        for k in range(4)
    ]
    return cut(fuse(body, shaft, lock), *holes)  # type: ignore[arg-type]


def motor_screw_points(p: Params) -> list[Point2]:
    r = p["mount_circle"] / 2
    return [
        (r * math.cos(math.radians(45 + 90 * k)), r * math.sin(math.radians(45 + 90 * k)))
        for k in range(4)
    ]


def prop_disc(p: Params) -> cq.Shape:
    """The volume a spinning propeller sweeps: a disc of the propeller diameter, with a
    bore in the middle where the motor's shaft and nut stand."""
    h = p["swept_height"]
    return cut(
        cylinder(p["diameter_in"] * 25.4, h),
        cylinder(p["bore"], h + 2).translate(cq.Vector(0, 0, -1)),
    )


def prop_blades(p: Params, clockwise: bool) -> cq.Shape:
    """A three-blade propeller seen from above, turning clockwise or counter-clockwise,
    with its hub on z = 0. Each blade is a loft of thin ellipses set at the pitch angle
    atan(pitch / (2 pi r)); the chord follows a typical 7-inch planform (widest at about
    40 % of the radius) but is kept small enough that the blade stays inside the swept
    disc (``swept_height``)."""
    radius = p["diameter_in"] * 25.4 / 2
    pitch = p["pitch_in"] * 25.4
    hub_d, hub_h = p["hub"]
    h = p["swept_height"]
    zc = h / 2  # blades centred in the disc height
    wires = []
    for frac in (0.07, 0.2, 0.3, 0.42, 0.55, 0.7, 0.85, 0.96):
        r = frac * radius
        beta = math.atan(pitch / (2 * math.pi * r))
        chord = (
            p["chord_max"]
            * (0.55 + 0.45 * math.sin(math.pi * min(frac / 0.8, 1.0)))
            * (1.0 if frac < 0.8 else max(0.35, 1 - (frac - 0.8) * 3.2))
        )
        thick = max(0.9, 0.11 * chord * (1.2 - frac))
        chord = min(chord, (h - 0.6 - thick) / math.sin(beta))
        # Chord from trailing to leading edge: forward in the clockwise direction and up.
        direction = cq.Vector(0, -math.cos(beta), math.sin(beta))
        wires.append(
            cq.Wire.makeEllipse(
                chord / 2, thick / 2, cq.Vector(r, 0, zc), cq.Vector(1, 0, 0), direction
            )
        )
    blade = cq.Solid.makeLoft(wires, False)
    blades = [blade.rotate(cq.Vector(0, 0, 0), cq.Vector(0, 0, 1), 120 * k) for k in range(3)]
    hub = cut(cylinder(hub_d, hub_h), cylinder(5.0, hub_h + 2).translate(cq.Vector(0, 0, -1)))
    prop = fuse(hub, *blades)
    # A counter-clockwise propeller is the mirror image of a clockwise one.
    return prop if clockwise else prop.mirror("XZ")


def board(p: Params) -> cq.Solid:
    """A circuit board as a box as tall as its tallest component."""
    return box(p["length"], p["width"], p["height"])


def mast(p: Params) -> cq.Shape:
    """The printed GPS mast: a square foot screwed to the top plate and the tube on it."""
    side, t = p["foot"]
    foot = rounded_box(side, side, t, 4)
    tube = cylinder(p["diameter"], p["height"] - t).translate(cq.Vector(0, 0, t))
    return fuse(foot, tube)


def leg(p: Params) -> cq.Solid:
    """A printed landing leg: a block under the arm."""
    return box(p["length"], p["width"], p["height"])


def tray(p: Params, boss_x: Sequence[float]) -> cq.Shape:
    """The printed Pi tray with a screw boss on each side at every ``boss_x`` for the
    canopy's M2 side screws (a 3 mm hole for a heat-set insert along y)."""
    body = rounded_box(p["length"], p["width"], p["height"], 4)
    bl, bt, bh = p["boss"]
    bosses = []
    holes = []
    z_hole = p["height"] + bh / 2 + 1
    for x in boss_x:
        for side in (1, -1):
            y = side * (p["width"] / 2 - bt / 2 - 0.1)
            bosses.append(box(bl, bt, bh).translate(cq.Vector(x, y, p["height"])))
            hole = cq.Solid.makeCylinder(
                1.5, bt + 2, cq.Vector(x, y - side * (bt / 2 + 1), z_hole), cq.Vector(0, side, 0)
            )
            holes.append(hole)
    return cut(fuse(body, *bosses), *holes)


def fov_pyramid(hfov_deg: float, vfov_deg: float, length: float) -> cq.Solid:
    """The camera's field of view as a pyramid with its apex at the origin, pointing down -z.

    The horizontal field of view spans x, the vertical spans y (assembly.py rotates the
    camera so that its wide side looks where it should). A 0.1 mm square stands in for
    the apex so that the loft has two faces to join.
    """
    half_x = length * math.tan(math.radians(hfov_deg / 2))
    half_y = length * math.tan(math.radians(vfov_deg / 2))
    apex = cq.Workplane("XY").rect(0.1, 0.1).val()
    base = cq.Workplane("XY").workplane(offset=-length).rect(2 * half_x, 2 * half_y).val()
    return cq.Solid.makeLoft([apex, base], True)  # type: ignore[arg-type]


def _section(x: float, width: float, height: float, radius: float, below: float) -> cq.Wire:
    """A canopy section at ``x``: a rectangle from z = -below to ``height`` with its four
    corners rounded (the part below the rim is cut off later, leaving square corners there)."""
    r = min(radius, width / 2 - 0.5, (height + below) / 2 - 0.5)
    sketch = cq.Sketch().rect(width, height + below).vertices().fillet(r)
    face = sketch._faces.Faces()[0]  # the sketch's only face
    wire = face.outerWire()
    # The sketch lies in XY: its x becomes y, its y becomes z, then it moves to x.
    origin = cq.Vector(0, 0, 0)
    upright = wire.rotate(origin, cq.Vector(1, 0, 0), 90).rotate(origin, cq.Vector(0, 0, 1), 90)
    return upright.translate(cq.Vector(x, 0, (height - below) / 2))  # type: ignore[return-value]


def canopy_form(p: Params, inset: float = 0.0, bottom: float = 0.0) -> cq.Shape:
    """The canopy's closed outer form: a loft through the ``stations`` from nose to tail,
    cut flat at z = ``bottom`` (the rim is z = 0). ``inset`` shrinks it on every side but
    the bottom, for the inside of the shell. Few faces, so that it edits well in a CAD
    program."""
    below = 10.0
    stations = [(x + p["x"], w, h) for x, w, h in p["stations"]]
    wires = []
    last = len(stations) - 1
    for i, (x, w, h) in enumerate(stations):
        xi = x - inset if i == 0 else (x + inset if i == last else x)
        wires.append(_section(xi, w - 2 * inset, h - inset, p["edge_radius"] - inset, below))
    loft = cq.Solid.makeLoft(wires, False)
    span = max(x for x, _, _ in stations) - min(x for x, _, _ in stations) + 40
    width = max(w for _, w, _ in stations) + 40
    centre = (max(x for x, _, _ in stations) + min(x for x, _, _ in stations)) / 2
    floor = box(span, width, below + 1).translate(cq.Vector(centre, 0, bottom - below - 1))
    return cut(loft, floor)


def canopy(
    p: Params,
    vent_x: float,
    mast_x: float,
    mast_diameter: float,
    keep_out: list[tuple[float, float]],
    keep_out_radius: float,
    screw_points: Sequence[Point3] = (),
) -> cq.Shape:
    """The top canopy over the companion computer, open underneath, rim on z = 0.

    The outer form (``canopy_form``) minus the same form shrunk by ``wall`` (not OCCT's
    shell: its result differs in the last digits from run to run). A scallop is cut at
    every ``keep_out`` point (the motor axes) so that, seen from above, it stays outside
    the propeller discs. Openings: an intake slot in the nose, a roof vent over the cooler
    and AI HAT at ``vent_x``, an exhaust slot in the tail, a hole for the GPS mast at
    ``mast_x`` and a 2.2 mm hole for each M2 side screw at ``screw_points`` (x, y, z).
    """
    t = p["wall"]
    body = cut(canopy_form(p), canopy_form(p, t, bottom=-1.0))
    top = max(h for _, _, h in p["stations"])
    nose = max(x for x, _, _ in p["stations"]) + p["x"]
    tail = min(x for x, _, _ in p["stations"]) + p["x"]
    iw, ih = p["intake"]
    vl, vw = p["vent"]
    ew, eh = p["exhaust"]
    nose_h = next(h for x, _, h in p["stations"] if x + p["x"] == nose)
    tail_h = next(h for x, _, h in p["stations"] if x + p["x"] == tail)
    cuts: list[cq.Shape] = [
        box(30, iw, ih).translate(cq.Vector(nose - 10, 0, nose_h * 0.5 - ih / 2)),
        rounded_box(vl, vw, 30, 6).translate(cq.Vector(vent_x, 0, top - 15)),
        box(30, ew, eh).translate(cq.Vector(tail + 10, 0, tail_h * 0.5 - eh / 2)),
        cylinder(mast_diameter + 2 * p["mast_clearance"], top + 20).translate(
            cq.Vector(mast_x, 0, -1)
        ),
    ]
    cuts += [
        cylinder(2 * keep_out_radius, top + 20).translate(cq.Vector(x, y, -2)) for x, y in keep_out
    ]
    for x, y, z in screw_points:
        side = 1 if y > 0 else -1
        cuts.append(
            cq.Solid.makeCylinder(1.1, 10, cq.Vector(x, y - side * 5, z), cq.Vector(0, side, 0))
        )
    return cut(body, *cuts)


def nose_fairing(p: Params) -> cq.Solid:
    """The printed nose under the frame, ahead of the battery, hanging from z = 0 down.

    The rear end (-x) is open for the battery; it starts ``length`` behind the nose tip.
    The camera's view window is cut by the caller.
    """
    trim = p["rear_trim"]
    long = {**p, "length": p["length"] + trim}
    t = p["wall"]

    def rounded(length: float, width: float, height: float, inset: float) -> cq.Shape:
        return (
            cq.Workplane("XY")
            .box(length, width, height, centered=(True, True, False))
            .edges("|Z and >X")
            .fillet(p["nose_radius"] - inset)
            .edges("<Z")
            .fillet(p["edge_radius"] - inset)
            .val()
        )

    # Outer form minus inner form (open at the top), not OCCT's shell (see canopy).
    outer = rounded(long["length"], p["width"], p["height"], 0)
    inner = rounded(long["length"] - 2 * t, p["width"] - 2 * t, p["height"], t).translate(
        cq.Vector(0, 0, t)
    )
    body = cut(outer, inner)
    rear = -long["length"] / 2
    open_end = box(2 * trim, p["width"] + 2, p["height"] + 2).translate(cq.Vector(rear, 0, -1))
    # Front tip at x = 0, the shell hanging below z = 0.
    return cut(body, open_end).translate(cq.Vector(-long["length"] / 2, 0, -p["height"]))  # type: ignore[return-value]


def tube(points: Sequence[Point3], diameter: float) -> cq.Shape:
    """A cable along a polyline: a cylinder per segment, joined by spheres at the bends."""
    r = diameter / 2
    pieces: list[cq.Shape] = []
    for a, b in itertools.pairwise(points):
        va, vb = cq.Vector(*a), cq.Vector(*b)
        pieces.append(cq.Solid.makeCylinder(r, (vb - va).Length, va, vb - va))
    pieces += [
        cq.Solid.makeSphere(r, cq.Vector(*pt), angleDegrees1=-90, angleDegrees2=90)
        for pt in points[1:-1]
    ]
    return fuse(pieces[0], *pieces[1:])


def ribbon(points: Sequence[Point3], width: float, thickness: float) -> cq.Shape:
    """A flat cable (FFC) along a polyline of segments parallel to x or z, its width
    along y."""
    pieces: list[cq.Shape] = []
    for a, b in itertools.pairwise(points):
        lo = [min(a[i], b[i]) for i in range(3)]
        hi = [max(a[i], b[i]) for i in range(3)]
        dx, dz = hi[0] - lo[0], hi[2] - lo[2]
        if dx >= dz:
            size = (dx + thickness, width, thickness)
        else:
            size = (thickness, width, dz + thickness)
        centre = [(lo[i] + hi[i]) / 2 for i in range(3)]
        pieces.append(
            cq.Solid.makeBox(
                *size,
                cq.Vector(
                    centre[0] - size[0] / 2, centre[1] - size[1] / 2, centre[2] - size[2] / 2
                ),
            )
        )
    return fuse(pieces[0], *pieces[1:])


def path_length(points: Sequence[Point3]) -> float:
    return sum(math.dist(a, b) for a, b in itertools.pairwise(points))


def strap(p: Params, battery: Params) -> cq.Shape:
    """A battery strap seen as a U around the battery, centred on x = 0: under the battery
    and up both sides to the bottom plate (z = 0); the loop over the plate is left out."""
    w, t = p["width"], p["thickness"]
    bw, bh = battery["width"], battery["height"]
    under = box(w, bw + 2 * t, t).translate(cq.Vector(0, 0, -bh - t))
    sides = [
        box(w, t, bh).translate(cq.Vector(0, side * (bw / 2 + t / 2), -bh)) for side in (1, -1)
    ]
    return fuse(under, *sides)


def antenna(p: Params, start: Point3, exit_: Point3, direction: Point3) -> cq.Shape:
    """A receiver antenna: the thin coax from the receiver to ``exit_``, then the thicker
    active element (``element``: length, diameter) pointing along ``direction``."""
    coax = tube([start, exit_], p["diameter"])
    length, d = p["element"]
    v = cq.Vector(*direction).normalized()
    element = cq.Solid.makeCylinder(d / 2, length, cq.Vector(*exit_), v)
    return fuse(coax, element)


def xt60_pair(p: Params) -> cq.Shape:
    """A mated XT60 pair lying along x, centred on the origin, flat side down."""
    length = 2 * p["length"] - 9  # the plug's pins sit inside the socket
    body = cq.Workplane("XY").box(length, p["height"], p["width"]).edges("|X").fillet(1.5).val()
    return body  # type: ignore[return-value]
