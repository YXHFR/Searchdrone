"""Drag of the fpv7 body in forward flight: Cd A part by part, with the parts in the wake of
others counted at a fraction of their area.

Every part except the propeller discs is projected onto the plane across the oncoming air
(level flight at ``aero.pitch_deg`` nose down) on a 0.5 mm grid, with how far upstream its
front and rear faces lie at every point. A point of a part then counts

- fully, if nothing lies upstream of it there;
- at ``aero.wake_factor``, if another part lies ahead of it within ``aero.wake_length``
  (in its wake, where the air is slower: shielding, see README.md);
- not at all, if it lies inside another part's outline there (a screw in a standoff).

Each part has a drag coefficient by its shape: printed covers ``cd_faired``, round parts
(motors, standoffs, screws, cables, the mast) ``cd_cylinder``, thin flat parts edge-on
(plates, arms, straps) ``cd_plate``, everything else (boards, battery, connectors)
``cd_bluff``. Cd A is the sum over the parts of coefficient times counted area. The
coefficients are textbook values for single bodies (sources in README.md, osäkert) and the
result is meant to be checked against flight logs; ``reference_cda_m2`` is the range that
NASA measured for whole quadcopter bodies without rotors, as a sanity check.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any

import numpy as np

if TYPE_CHECKING:
    from assembly import Drone, Part

PIXEL = 0.5  # mm
ROUND_PARTS = ("motor", "fastener", "cable")
ROUND_NAMES = ("standoff", "arm_standoff", "gps_mast")
FLAT_NAMES = ("bottom_plate", "top_plate", "arm", "reinforce", "strap")


def shape_class(part: Part) -> str:
    """Which drag coefficient a part gets: faired, cylinder, plate or bluff."""
    base = part.name.rstrip("0123456789_")
    if part.group == "cover":
        return "faired"
    if part.group in ROUND_PARTS or base in ROUND_NAMES:
        return "cylinder"
    if base in FLAT_NAMES or base.startswith("reinforce"):
        return "plate"
    return "bluff"


@dataclass
class Footprint:
    """A part seen along the flow: the grid window it covers and, per cell, how far
    upstream its front and rear faces lie (mm; NaN where it does not cover the cell)."""

    i0: int
    j0: int
    front: np.ndarray
    back: np.ndarray

    @property
    def covered(self) -> np.ndarray:
        return ~np.isnan(self.front)


def flow_axes(pitch_deg: float) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Unit vectors in the body frame: upstream (where the air comes from), and two
    across the flow (y, and 'up' across the flow)."""
    th = math.radians(pitch_deg)
    upstream = np.array([math.cos(th), 0.0, math.sin(th)])
    across = np.array([0.0, 1.0, 0.0])
    up = np.cross(upstream, across)
    return upstream, across, up


def footprint(part: Part, axes: tuple[np.ndarray, np.ndarray, np.ndarray]) -> Footprint:
    upstream, across, up = axes
    vertices, triangles = part.solid.tessellate(0.1, 0.3)
    v = np.array([(p.x, p.y, p.z) for p in vertices])
    tri = v[np.array(triangles)]
    s, t, d = tri @ across / PIXEL, tri @ up / PIXEL, tri @ upstream
    i0, j0 = math.floor(t.min()), math.floor(s.min())
    rows, cols = math.ceil(t.max()) - i0 + 1, math.ceil(s.max()) - j0 + 1
    front = np.full((rows, cols), -np.inf)
    back = np.full((rows, cols), np.inf)
    for k in range(len(tri)):
        _fill(s[k] - j0, t[k] - i0, d[k], front, back)
    covered = np.isfinite(front)
    return Footprint(i0, j0, np.where(covered, front, np.nan), np.where(covered, back, np.nan))


def _fill(
    px: np.ndarray, py: np.ndarray, pz: np.ndarray, front: np.ndarray, back: np.ndarray
) -> None:
    """Rasterise one triangle (cell centres inside it), keeping the largest and smallest
    upstream distance in every cell. Triangles thinner than a cell still mark the cells
    their corners fall in, so thin parts keep their outline."""
    h, w = front.shape
    for x, y, z in zip(px, py, pz, strict=True):
        # A corner on a cell boundary belongs to the cell below it.
        i = min(max(math.ceil(y) - 1, 0), h - 1)
        j = min(max(math.ceil(x) - 1, 0), w - 1)
        front[i, j] = max(front[i, j], z)
        back[i, j] = min(back[i, j], z)
    x0, x1 = max(math.floor(px.min()), 0), min(math.ceil(px.max()), w - 1)
    y0, y1 = max(math.floor(py.min()), 0), min(math.ceil(py.max()), h - 1)
    (ax, bx, cx), (ay, by, cy) = px, py
    area = (bx - ax) * (cy - ay) - (cx - ax) * (by - ay)
    if abs(area) < 1e-12 or x0 > x1 or y0 > y1:
        return
    xs = np.arange(x0, x1 + 1) + 0.5
    ys = np.arange(y0, y1 + 1)[:, None] + 0.5
    w0 = ((bx - xs) * (cy - ys) - (cx - xs) * (by - ys)) / area
    w1 = ((cx - xs) * (ay - ys) - (ax - xs) * (cy - ys)) / area
    w2 = 1 - w0 - w1
    inside = (w0 >= -1e-9) & (w1 >= -1e-9) & (w2 >= -1e-9)
    if not inside.any():
        return
    z = w0 * pz[0] + w1 * pz[1] + w2 * pz[2]
    f = front[y0 : y1 + 1, x0 : x1 + 1]
    b = back[y0 : y1 + 1, x0 : x1 + 1]
    np.maximum(f, np.where(inside, z, -np.inf), out=f)
    np.minimum(b, np.where(inside, z, np.inf), out=b)


def weights(prints: list[Footprint], wake_factor: float, wake_length: float) -> list[np.ndarray]:
    """Per part and cell: 1 if exposed, ``wake_factor`` if in another part's wake, 0 if
    inside another part's outline."""
    out = []
    for b, fb in enumerate(prints):
        hb, wb = fb.front.shape
        hidden = np.zeros((hb, wb), dtype=bool)
        shaded = np.zeros((hb, wb), dtype=bool)
        for a, fa in enumerate(prints):
            if a == b:
                continue
            i0, i1 = max(fa.i0, fb.i0), min(fa.i0 + fa.front.shape[0], fb.i0 + hb)
            j0, j1 = max(fa.j0, fb.j0), min(fa.j0 + fa.front.shape[1], fb.j0 + wb)
            if i0 >= i1 or j0 >= j1:
                continue
            sa = (slice(i0 - fa.i0, i1 - fa.i0), slice(j0 - fa.j0, j1 - fa.j0))
            sb = (slice(i0 - fb.i0, i1 - fb.i0), slice(j0 - fb.j0, j1 - fb.j0))
            front_a, back_a, front_b = fa.front[sa], fa.back[sa], fb.front[sb]
            with np.errstate(invalid="ignore"):
                # b's front face inside a (a reaches further upstream and further down):
                # ties go to the part listed first, so two equal parts do not hide each other.
                inside = (front_a >= front_b) & (back_a <= front_b)
                if a > b:
                    inside &= front_a > front_b
                ahead = (back_a > front_b) & (back_a - front_b <= wake_length)
            hidden[sb] |= inside
            shaded[sb] |= ahead
        weight = np.where(shaded, wake_factor, 1.0)
        weight = np.where(hidden, 0.0, weight)
        out.append(np.where(fb.covered, weight, 0.0))
    return out


def drag(parts: list[Part], a: dict[str, Any]) -> dict[str, Any]:
    """Cd A (m^2) of ``parts`` and its split by shape class and by part."""
    axes = flow_axes(a["pitch_deg"])
    prints = [footprint(pt, axes) for pt in parts]
    w = weights(prints, a["wake_factor"], a["wake_length"])
    cd = {
        "faired": a["cd_faired"],
        "cylinder": a["cd_cylinder"],
        "plate": a["cd_plate"],
        "bluff": a["cd_bluff"],
    }
    cell = PIXEL * PIXEL * 1e-6
    by_part = {}
    by_class = dict.fromkeys(cd, 0.0)
    for pt, wt in zip(parts, w, strict=True):
        kind = shape_class(pt)
        cda = cd[kind] * float(wt.sum()) * cell
        by_part[pt.name] = cda
        by_class[kind] += cda
    covered = np.zeros(1)
    if prints:
        i0 = min(f.i0 for f in prints)
        j0 = min(f.j0 for f in prints)
        h = max(f.i0 + f.front.shape[0] for f in prints) - i0
        wd = max(f.j0 + f.front.shape[1] for f in prints) - j0
        covered = np.zeros((h, wd), dtype=bool)
        for f in prints:
            covered[
                f.i0 - i0 : f.i0 - i0 + f.front.shape[0], f.j0 - j0 : f.j0 - j0 + f.front.shape[1]
            ] |= f.covered
    return {
        "area_m2": float(covered.sum()) * cell,
        "cda_m2": sum(by_part.values()),
        "by_class": by_class,
        "by_part": by_part,
    }


def aero(drone: Drone, p: dict[str, Any]) -> dict[str, Any]:
    """Frontal area, Cd A and drag at ``aero.speed`` without and with the printed covers
    (propeller discs left out), the largest contributors, and the comparison with the
    measured range for whole quadcopter bodies."""
    a = p["aero"]
    body = [pt for pt in drone.parts if pt.group != "prop"]
    bare = [pt for pt in body if pt.group != "cover"]
    before, after = drag(bare, a), drag(body, a)
    q = 0.5 * a["air_density"] * a["speed"] ** 2
    top = sorted(after["by_part"].items(), key=lambda kv: (-kv[1], kv[0]))[:8]
    lo, hi = a["reference_cda_m2"]
    return {
        "speed_m_s": a["speed"],
        "pitch_deg": a["pitch_deg"],
        "frontal_area_before_m2": round(before["area_m2"], 6),
        "frontal_area_after_m2": round(after["area_m2"], 6),
        "cda_before_m2": round(before["cda_m2"], 6),
        "cda_after_m2": round(after["cda_m2"], 6),
        "drag_before_n": round(q * before["cda_m2"], 3),
        "drag_after_n": round(q * after["cda_m2"], 3),
        "cda_by_class_m2": {k: round(v, 6) for k, v in after["by_class"].items()},
        "cda_top_parts_m2": {k: round(v, 6) for k, v in top},
        "reference_cda_m2": [lo, hi],
    }
