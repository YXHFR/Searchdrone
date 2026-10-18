"""Simplified solids for the fpv7 parts, each built around its own origin.

Every function takes the part's section of params.yaml (millimetres) and returns a
CadQuery solid. Boards and boxes sit on z = 0 and are centred in x and y; assembly.py
moves them into place. The shapes are envelopes, not detailed meshes: a board is a box
as tall as its tallest component, a propeller is the disc its blades sweep.
"""

from __future__ import annotations

import math
from typing import Any

import cadquery as cq

Params = dict[str, Any]


def box(length: float, width: float, height: float) -> cq.Solid:
    """A box from z = 0 to z = height, centred in x and y."""
    return cq.Solid.makeBox(length, width, height, cq.Vector(-length / 2, -width / 2, 0))


def cylinder(diameter: float, height: float) -> cq.Solid:
    """A cylinder from z = 0 to z = height on the z axis."""
    return cq.Solid.makeCylinder(diameter / 2, height)


def plate(p: Params) -> cq.Solid:
    """A frame plate with rounded corners (the plates' outline is simplified to a rectangle)."""
    r = min(p["corner_radius"], p["width"] / 2 - 0.01, p["length"] / 2 - 0.01)
    wp = cq.Workplane("XY").rect(p["length"], p["width"]).extrude(p["thickness"])
    return wp.edges("|Z").fillet(r).val()  # type: ignore[return-value]


def arm(frame: Params) -> cq.Solid:
    """One arm along +x, tapering from its root to just past the motor.

    The root starts ``arm.root_offset`` from the centre, where it is clamped between the
    plates without touching the other arms, and the tip ends ``arm.tip_overhang`` past the
    motor axis; the width goes linearly from ``arm.root_width`` to ``arm.tip_width``.
    """
    a = frame["arm"]
    reach = frame["wheelbase"] / 2 + a["tip_overhang"]
    pts = [
        (a["root_offset"], -a["root_width"] / 2),
        (reach, -a["tip_width"] / 2),
        (reach, a["tip_width"] / 2),
        (a["root_offset"], a["root_width"] / 2),
    ]
    return cq.Workplane("XY").polyline(pts).close().extrude(a["thickness"]).val()  # type: ignore[return-value]


def standoff(frame: Params) -> cq.Solid:
    s = frame["standoff"]
    return cylinder(s["diameter"], s["height"])


def motor(p: Params) -> cq.Solid:
    """The motor as one cylinder: bell diameter, base to the top of the bell."""
    return cylinder(p["diameter"], p["height"])


def prop_disc(p: Params) -> cq.Solid:
    """The volume a spinning propeller sweeps: a disc of the propeller diameter."""
    return cylinder(p["diameter_in"] * 25.4, p["swept_height"])


def board(p: Params) -> cq.Solid:
    """A circuit board as a box as tall as its tallest component."""
    return box(p["length"], p["width"], p["height"])


def mast(p: Params) -> cq.Solid:
    return cylinder(p["diameter"], p["height"])


def leg(p: Params) -> cq.Solid:
    """A printed landing leg: a block under the arm."""
    return box(p["length"], p["width"], p["height"])


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
