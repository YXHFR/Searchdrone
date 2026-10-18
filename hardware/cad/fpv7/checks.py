"""Checks on the fpv7 assembly: mass, centre of gravity, inertia, overlaps, margins, view.

Each part is a solid of uniform density that weighs what params.yaml says, so its
centre of mass is its centroid and its inertia comes from its shape. Point masses add
their mass and their parallel-axis term. Inertia is about the centre of gravity, in the
body axes x forward, y left, z up, in kg m^2.
"""

from __future__ import annotations

from itertools import combinations
from pathlib import Path
from typing import TYPE_CHECKING, Any

import cadquery as cq
import yaml
from aero import aero
from shapely.geometry import Point, Polygon
from shapely.ops import unary_union

if TYPE_CHECKING:
    from assembly import Drone, Part

G_MM2_TO_KG_M2 = 1e-9
# The take-off mass limit for the hardware revision, shared with `just hw-check`.
LIMITS = Path(__file__).resolve().parents[2] / "limits.yaml"
# Shared faces (a motor standing on an arm) give a zero-volume intersection; anything
# above this is a real overlap (mm^3).
OVERLAP_TOL_MM3 = 1e-3


def mass_properties(drone: Drone) -> dict[str, Any]:
    total = sum(pt.mass_g for pt in drone.parts) + sum(pm.mass_g for pm in drone.point_masses)
    moment = [0.0, 0.0, 0.0]
    items: list[tuple[float, cq.Vector, list[list[float]]]] = []
    for pt in drone.parts:
        c = pt.solid.Center()
        vol = pt.solid.Volume()
        # matrixOfInertia is for density 1 (mm^5) about the centroid; scale to the mass.
        local = [[v * pt.mass_g / vol for v in row] for row in cq.Shape.matrixOfInertia(pt.solid)]
        items.append((pt.mass_g, c, local))
    for pm in drone.point_masses:
        items.append((pm.mass_g, cq.Vector(*pm.at), [[0.0] * 3 for _ in range(3)]))
    for mass, c, _ in items:
        moment = [moment[0] + mass * c.x, moment[1] + mass * c.y, moment[2] + mass * c.z]
    cog = [m / total for m in moment]
    inertia = [[0.0] * 3 for _ in range(3)]
    for mass, c, local in items:
        d = [c.x - cog[0], c.y - cog[1], c.z - cog[2]]
        d2 = sum(v * v for v in d)
        for i in range(3):
            for j in range(3):
                shift = mass * ((d2 if i == j else 0.0) - d[i] * d[j])
                inertia[i][j] += (local[i][j] + shift) * G_MM2_TO_KG_M2
    return {"mass_g": total, "cog_mm": cog, "inertia_kg_m2": inertia}


def _bbox_apart(a: cq.Shape, b: cq.Shape, gap: float = 0.0) -> bool:
    box_a, box_b = a.BoundingBox(), b.BoundingBox()
    return (
        box_a.xmax + gap < box_b.xmin or box_b.xmax + gap < box_a.xmin
        or box_a.ymax + gap < box_b.ymin or box_b.ymax + gap < box_a.ymin
        or box_a.zmax + gap < box_b.zmin or box_b.zmax + gap < box_a.zmin
    )  # fmt: skip


def overlaps(parts: list[Part]) -> list[dict[str, Any]]:
    """Every pair of parts whose solids share volume."""
    found = []
    for a, b in combinations(parts, 2):
        if _bbox_apart(a.solid, b.solid):
            continue
        vol = a.solid.intersect(b.solid).Volume()
        if vol > OVERLAP_TOL_MM3:
            found.append({"a": a.name, "b": b.name, "volume_mm3": round(vol, 3)})
    return found


def own_motor(prop: str) -> str:
    return prop.replace("prop_", "motor_")


def prop_margins(parts: list[Part], limit: float) -> list[dict[str, Any]]:
    """For each propeller disc, the nearest other part (its own motor excluded)."""
    rows = []
    for prop in (pt for pt in parts if pt.group == "prop"):
        best: tuple[float, str] = (float("inf"), "")
        for other in parts:
            if other.name in (prop.name, own_motor(prop.name)):
                continue
            # Only parts within reach can be the nearest one below the limit.
            if _bbox_apart(prop.solid, other.solid, gap=max(limit, best[0])):
                continue
            dist = prop.solid.distance(other.solid)
            best = min(best, (dist, other.name))
        rows.append({"prop": prop.name, "nearest": best[1], "distance_mm": round(best[0], 2)})
    return rows


def _silhouette(solid: cq.Shape, axes: tuple[str, str]) -> Any:
    """The solid's outline projected on the plane of two axes, as a shapely shape."""
    vertices, triangles = solid.tessellate(0.5, 0.3)
    a, b = axes
    tris = [
        Polygon([(getattr(vertices[i], a), getattr(vertices[i], b)) for i in tri])
        for tri in triangles
    ]
    return unary_union([t for t in tris if t.area > 1e-6])


def plan_margins(parts: list[Part]) -> list[dict[str, Any]]:
    """Seen from above: how far each part above a propeller disc stays outside it.

    A part whose lowest point is above the top of the discs must not hang over them,
    even if it is far enough away in 3D: it would shade the inflow and a flexing blade
    tip could reach it. Negative means it overlaps the disc by that much.
    """
    props = [pt for pt in parts if pt.group == "prop"]
    rows = []
    for prop in props:
        box = prop.solid.BoundingBox()
        centre, radius = Point(box.center.x, box.center.y), box.xlen / 2
        best: tuple[float, str] = (float("inf"), "")
        for other in parts:
            if other.group == "prop" or other.solid.BoundingBox().zmin < box.zmax:
                continue
            gap = centre.distance(_silhouette(other.solid, ("x", "y"))) - radius
            best = min(best, (gap, other.name))
        rows.append({"prop": prop.name, "nearest": best[1], "distance_mm": round(best[0], 2)})
    return rows


def fov_hits(drone: Drone) -> list[dict[str, Any]]:
    """Parts inside the camera's field of view (the camera itself excluded)."""
    hits = []
    for pt in drone.parts:
        if pt.name == drone.camera_name or _bbox_apart(pt.solid, drone.fov):
            continue
        vol = pt.solid.intersect(drone.fov).Volume()
        if vol > OVERLAP_TOL_MM3:
            hits.append({"part": pt.name, "volume_mm3": round(vol, 3)})
    return hits


def report(drone: Drone, p: dict[str, Any]) -> dict[str, Any]:
    mp = mass_properties(drone)
    ref = p["reference"]
    limit = p["checks"]["prop_margin_min"]
    margins = prop_margins(drone.parts, limit)
    over = overlaps(drone.parts)
    hits = fov_hits(drone)
    diag = [mp["inertia_kg_m2"][i][i] for i in range(3)]
    max_mass_g = yaml.safe_load(LIMITS.read_text(encoding="utf-8"))["max_takeoff_mass_kg"] * 1000
    cog_max = p["checks"]["cog_offset_max"]
    out = {
        "mass_g": round(mp["mass_g"], 1),
        "reference_mass_g": ref["takeoff_mass_g"],
        "max_takeoff_mass_g": max_mass_g,
        "cog_mm": [round(v, 2) for v in mp["cog_mm"]],
        "cog_offset_max_mm": cog_max,
        "cover_mass_g": round(sum(pt.mass_g for pt in drone.parts if pt.group == "cover"), 1),
        "cover_mass_max_g": p["checks"]["cover_mass_max"],
        "aero": aero(drone, p),
        "inertia_kg_m2": [[round(v, 6) for v in row] for row in mp["inertia_kg_m2"]],
        "inertia_diag_kg_m2": [round(v, 6) for v in diag],
        "reference_inertia_kg_m2": [v / 1000 for v in ref["inertia_g_m2"]],
        "overlaps": over,
        "prop_margins": margins,
        "prop_margin_min_mm": limit,
        "plan_margins": plan_margins(drone.parts),
        "plan_margin_min_mm": p["checks"]["plan_margin_min"],
        "fov_hits": hits,
        "parts": [
            *(
                {"name": pt.name, "group": pt.group, "mass_g": round(pt.mass_g, 2)}
                for pt in drone.parts
            ),
            *(
                {"name": pm.name, "group": "point mass", "mass_g": pm.mass_g}
                for pm in drone.point_masses
            ),
        ],
    }
    out["ok"] = all(verdicts(out).values())
    return out


def cog_ok(r: dict[str, Any]) -> bool:
    """The centre of gravity is within the allowed offset from the centre in x and y."""
    x, y, _ = r["cog_mm"]
    return abs(x) <= r["cog_offset_max_mm"] and abs(y) <= r["cog_offset_max_mm"]


def verdicts(r: dict[str, Any]) -> dict[str, bool]:
    """Pass or fail per check; the report's ``ok`` (and the exit code of `just cad`) is all."""
    return {
        "overlaps": not r["overlaps"],
        "prop_margin": all(m["distance_mm"] >= r["prop_margin_min_mm"] for m in r["prop_margins"]),
        "plan_margin": all(m["distance_mm"] >= r["plan_margin_min_mm"] for m in r["plan_margins"]),
        "camera_view": not r["fov_hits"],
        "cog": cog_ok(r),
        "mass": r["mass_g"] <= r["max_takeoff_mass_g"],
        "cover_mass": r["cover_mass_g"] <= r["cover_mass_max_g"],
    }


def markdown(r: dict[str, Any]) -> str:
    def yes(ok: bool) -> str:
        return "ok" if ok else "FEL"

    v = verdicts(r)
    lines = [
        "# fpv7: mass, inertia and checks",
        "",
        "| Quantity | Model | Reference | Model / reference |",
        "|---|---|---|---|",
        f"| Take-off mass (g) | {r['mass_g']:.1f} | {r['reference_mass_g']} "
        f"| {r['mass_g'] / r['reference_mass_g']:.2f} |",
    ]
    for axis, model, ref in zip(
        "xyz", r["inertia_diag_kg_m2"], r["reference_inertia_kg_m2"], strict=True
    ):
        lines.append(f"| I{axis}{axis} (kg m^2) | {model:.5f} | {ref:.5f} | {model / ref:.2f} |")
    cx, cy, cz = r["cog_mm"]
    ae = r["aero"]
    lines += [
        "",
        f"Centre of gravity: x = {cx:.1f} mm, y = {cy:.1f} mm, z = {cz:.1f} mm "
        "(x forward, y left, z up; z = 0 under the bottom plate).",
        "",
        f"Drag in level flight at {ae['speed_m_s']} m/s, without and with the printed covers "
        "(aero.py: Cd A part by part, parts in another's wake at a fraction of their area; "
        "osäkert, to be calibrated against flight logs; propeller discs left out):",
        "",
        "| Quantity | Without covers | With covers |",
        "|---|---|---|",
        f"| Frontal area (cm^2) | {ae['frontal_area_before_m2'] * 1e4:.0f} "
        f"| {ae['frontal_area_after_m2'] * 1e4:.0f} |",
        f"| Cd A (cm^2) | {ae['cda_before_m2'] * 1e4:.0f} | {ae['cda_after_m2'] * 1e4:.0f} |",
        f"| Drag (N) | {ae['drag_before_n']:.2f} | {ae['drag_after_n']:.2f} |",
        "",
        "Cd A with the covers by shape (cm^2): "
        + ", ".join(f"{k} {v * 1e4:.1f}" for k, v in ae["cda_by_class_m2"].items())
        + ". Largest parts: "
        + ", ".join(f"{k} {v * 1e4:.1f}" for k, v in ae["cda_top_parts_m2"].items())
        + f". Measured for whole quadcopter bodies without rotors (NASA, larger drones): "
        f"{ae['reference_cda_m2'][0] * 1e4:.0f}-{ae['reference_cda_m2'][1] * 1e4:.0f} cm^2.",
        "",
        "| Check | Result |",
        "|---|---|",
        f"| No parts overlap | {yes(v['overlaps'])} ({len(r['overlaps'])} overlaps) |",
        f"| Propeller margin >= {r['prop_margin_min_mm']} mm | {yes(v['prop_margin'])} "
        f"(smallest {min(m['distance_mm'] for m in r['prop_margins']):.1f} mm) |",
        f"| Seen from above, nothing over the discs (>= {r['plan_margin_min_mm']} mm) "
        f"| {yes(v['plan_margin'])} "
        f"(smallest {min(m['distance_mm'] for m in r['plan_margins']):.1f} mm) |",
        f"| Camera view clear | {yes(v['camera_view'])} ({len(r['fov_hits'])} parts in view) |",
        f"| Centre of gravity within {r['cog_offset_max_mm']} mm of the centre in x and y "
        f"| {yes(v['cog'])} |",
        f"| Take-off mass <= {r['max_takeoff_mass_g']:.0f} g (hardware/limits.yaml) "
        f"| {yes(v['mass'])} |",
        f"| Printed covers <= {r['cover_mass_max_g']} g | {yes(v['cover_mass'])} "
        f"({r['cover_mass_g']:.1f} g) |",
        "",
        "| Propeller | Nearest part | Distance (mm) |",
        "|---|---|---|",
    ]
    lines += [
        f"| {m['prop']} | {m['nearest']} | {m['distance_mm']:.1f} |" for m in r["prop_margins"]
    ]
    for o in r["overlaps"]:
        lines.append(f"\nOverlap: {o['a']} and {o['b']}, {o['volume_mm3']} mm^3")
    for h in r["fov_hits"]:
        lines.append(f"\nIn the camera's view: {h['part']}, {h['volume_mm3']} mm^3")
    lines += ["", "| Part | Group | Mass (g) |", "|---|---|---|"]
    lines += [f"| {pt['name']} | {pt['group']} | {pt['mass_g']:.1f} |" for pt in r["parts"]]
    return "\n".join(lines) + "\n"
