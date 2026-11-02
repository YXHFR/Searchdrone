"""Probability of detection (POD) per cell: from one still, one pass and all passes.

One still: ``p_still(c) = p_det(px_person) * (1 - canopy(c)) * overlap(c)``, where
``px_person`` is a person's shoulder width (0.5 m) in pixels, ``p_det`` the active
detector profile's recall at that size, and ``overlap`` the share of the cell inside the
still's footprint. A still the detector failed on sees nothing.

Stills in a row see the same thing through the same canopy, so they are not independent:
within one pass (one order) ``POD_pass = 1 - (1 - max_i p_still_i)^min(k, 2)`` with ``k``
the number of stills that saw the cell, at most two independent looks per pass. Between
passes ``POD_tot = 1 - prod(1 - POD_pass)``.

A drone with a tele camera beside the main one takes two stills of one moment: they are
one look, not two, and each camera counts with its own ``p``: ``1 - (1 - p_main)(1 -
p_tele)`` inside the tele still's footprint, ``p_main`` outside it
(``same_moment_groups``).
"""

from __future__ import annotations

import math
from collections.abc import Iterable
from dataclasses import dataclass
from datetime import datetime
from typing import Final

import numpy as np
import shapely
from shapely.geometry import Polygon

from sd_search.probmap.grid import Floats, Grid

__all__ = [
    "DETECTORS",
    "PERSON_WIDTH_M",
    "PLAN_CAMERA",
    "SIDELAP",
    "Camera",
    "DetectorProfile",
    "Still",
    "pass_pod",
    "plan_pod",
    "same_moment_groups",
    "still_pod",
]

PERSON_WIDTH_M: Final = 0.5  # shoulder width: standing or sitting, the hardest case
MAX_LOOKS_PER_PASS: Final = 2
# A tele still and a main still of the same moment are one look (``same_moment_groups``):
# taken at most this far apart (both are processed in one round of the onboard loop),
# the tele still with at least this much more detail and this share of its view inside
# the main still's.
SAME_MOMENT_S: Final = 1.0
PAIR_DETAIL_RATIO: Final = 1.5
PAIR_INSIDE_SHARE: Final = 0.8
SIDELAP: Final = 0.2  # survey lines overlap by a fifth of a still's ground width


@dataclass(frozen=True)
class DetectorProfile:
    """A detector's recall against a person's size in pixels (piecewise linear between
    measured points, flat outside them)."""

    profile_id: str
    px: tuple[float, ...]
    recall: tuple[float, ...]

    def p_det(self, px_person: float) -> float:
        return float(np.interp(px_person, self.px, self.recall))


# The oracle's starting profile, at the level published for aerial person detection
# (recall 0.889 above 32 px, 0.6 at 16-32 px, 0.2 under 16 px). A starting value, to be
# replaced by measured profiles; nothing below 4 px.
DETECTORS: Final[dict[str, DetectorProfile]] = {
    "det_oracle_v0": DetectorProfile(
        "det_oracle_v0", px=(4.0, 8.0, 24.0, 40.0), recall=(0.0, 0.2, 0.6, 0.889)
    ),
}
DEFAULT_DETECTOR: Final = "det_oracle_v0"


@dataclass(frozen=True)
class Camera:
    width_px: int
    hfov_deg: float

    def px_per_m(self, alt_m: float) -> float:
        return self.width_px / self.ground_width_m(alt_m)

    def ground_width_m(self, alt_m: float) -> float:
        return 2 * alt_m * math.tan(math.radians(self.hfov_deg) / 2)


PLAN_CAMERA: Final = Camera(width_px=1280, hfov_deg=66.0)  # the onboard stills camera


@dataclass(frozen=True)
class Still:
    still_id: str
    seq: int
    pass_id: str  # the order it was taken during ("" for none)
    footprint: tuple[tuple[float, float], ...]  # corners in the grid's plane
    px_per_m: float
    detector_ok: bool
    captured_utc: datetime | None = None  # when it was taken (the map's clock)


def still_pod(
    grid: Grid,
    still: Still,
    canopy: Floats,
    detector: DetectorProfile,
) -> tuple[tuple[slice, slice], Floats] | None:
    """The still's ``p_still`` on the cells its footprint touches: (window, values)."""
    if not still.detector_ok:
        return None
    poly = Polygon(still.footprint)
    if not poly.is_valid or poly.area <= 0:
        return None
    xmin, ymin, xmax, ymax = poly.bounds
    win = grid.window(xmin, ymin, xmax, ymax)
    if win is None:
        return None
    sx, sy = grid.subpoints
    inside = shapely.contains_xy(poly, sx[win], sy[win])
    overlap = inside.mean(axis=-1)
    p = detector.p_det(PERSON_WIDTH_M * still.px_per_m)
    return win, p * (1 - canopy[win]) * overlap


def _inside_share(inner: Still, outer: Still) -> float:
    a, b = Polygon(inner.footprint), Polygon(outer.footprint)
    if not (a.is_valid and b.is_valid) or a.area <= 0:
        return 0.0
    return float(a.intersection(b).area / a.area)


def _pairs_with(tele: Still, main: Still) -> bool:
    """``tele`` and ``main`` are one moment seen by two cameras: taken at most
    ``SAME_MOMENT_S`` apart, ``tele`` with clearly more detail and its view inside
    ``main``'s. (One camera's stills have the same detail at one height, so two stills of
    one camera never pair.)"""
    if _gap_s(tele, main) > SAME_MOMENT_S:
        return False
    if tele.px_per_m < PAIR_DETAIL_RATIO * main.px_per_m:
        return False
    return _inside_share(tele, main) >= PAIR_INSIDE_SHARE


def same_moment_groups(stills: Iterable[Still]) -> list[list[Still]]:
    """The pass's stills as looks: each still alone, except a tele still with the main
    still of the same moment (``_pairs_with``, the nearest in time), which make one look
    together. The report says nothing about the camera, so the pair is recognised by its
    time, detail and footprint."""
    ordered = sorted(stills, key=lambda s: s.px_per_m)  # main stills first
    groups: list[list[Still]] = []
    taken: set[int] = set()
    for tele in reversed(ordered):  # the most detailed first
        if id(tele) in taken:
            continue
        partners = [
            m for m in ordered if id(m) not in taken and m is not tele and _pairs_with(tele, m)
        ]
        if partners:
            main = min(partners, key=lambda m: _gap_s(tele, m))
            taken |= {id(tele), id(main)}
            groups.append([main, tele])
    groups += [[s] for s in ordered if id(s) not in taken]
    return groups


def _gap_s(a: Still, b: Still) -> float:
    if a.captured_utc is None or b.captured_utc is None:
        return math.inf
    return abs((a.captured_utc - b.captured_utc).total_seconds())


def _look_pod(
    grid: Grid, group: list[Still], canopy: Floats, detector: DetectorProfile
) -> tuple[tuple[slice, slice], Floats] | None:
    """One look's ``p`` on the cells it touches: ``1 - prod(1 - p_still)`` over the
    look's stills, each with its own camera's ``p`` (inside the tele still's footprint
    both cameras, outside it the main camera alone)."""
    parts = [got for s in group if (got := still_pod(grid, s, canopy, detector)) is not None]
    if not parts:
        return None
    if len(parts) == 1:
        return parts[0]
    j0 = min(w[0].start for w, _ in parts)
    j1 = max(w[0].stop for w, _ in parts)
    i0 = min(w[1].start for w, _ in parts)
    i1 = max(w[1].stop for w, _ in parts)
    miss = np.ones((j1 - j0, i1 - i0))
    for (rows, cols), p in parts:
        miss[rows.start - j0 : rows.stop - j0, cols.start - i0 : cols.stop - i0] *= 1 - p
    return (slice(j0, j1), slice(i0, i1)), 1 - miss


def pass_pod(
    grid: Grid,
    stills: Iterable[Still],
    canopy: Floats,
    detector: DetectorProfile,
    hold: Iterable[tuple[int, int]] = (),
) -> Floats:
    """``POD_pass`` of one pass. ``hold``: cells this pass must not count, because a
    candidate seen there in this pass still waits for a person's decision."""
    best = np.zeros(grid.shape)
    looks = np.zeros(grid.shape)
    for group in same_moment_groups(stills):
        got = _look_pod(grid, group, canopy, detector)
        if got is None:
            continue
        win, p = got
        best[win] = np.maximum(best[win], p)
        looks[win] += p > 0
    k = np.minimum(looks, MAX_LOOKS_PER_PASS)
    pod = 1 - (1 - best) ** k
    for j, i in hold:
        pod[j, i] = 0.0
    return pod


def plan_pod(
    canopy: Floats,
    alt_m: float,
    detector: DetectorProfile,
    camera: Camera = PLAN_CAMERA,
) -> Floats:
    """The POD one more survey pass at ``alt_m`` would give each cell (full overlap, two
    looks)."""
    p = detector.p_det(PERSON_WIDTH_M * camera.px_per_m(alt_m)) * (1 - canopy)
    return 1 - (1 - p) ** MAX_LOOKS_PER_PASS
