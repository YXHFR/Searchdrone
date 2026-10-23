"""The charter's sectors on the map: what is left in each, and which to search next.

A sector's POA is the sum over the cells whose centre lies in it. Its planned POS is
``sum p(c) POD_plan(c)`` for one more survey pass at the charter's search height, and its
PSR (probability of success rate) that POS per minute of planned flight: line length
``area / spacing`` at the survey speed. ``suggest_sector_order`` orders the sectors by
PSR, highest first. It only suggests: nothing here starts a flight, the AI proposes and
the gate and a person approve as before.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from typing import Final

import shapely
from shapely.geometry import Polygon

from sd_schemas.probmap import SectorScore
from sd_search.probmap.bayes import MapState
from sd_search.probmap.grid import Floats
from sd_search.probmap.pod import PLAN_CAMERA, SIDELAP, Camera, DetectorProfile, plan_pod

__all__ = ["COVERED_POD", "PlanSector", "sector_scores", "situation_line", "suggest_sector_order"]

COVERED_POD: Final = 0.1  # a cell counts as covered from this cumulative POD


@dataclass(frozen=True)
class PlanSector:
    sector_id: str
    corners: tuple[tuple[float, float], ...]  # in the grid's plane


def spacing_m(alt_m: float, camera: Camera = PLAN_CAMERA) -> float:
    """The suggested line spacing: a still's ground width less the side overlap."""
    return camera.ground_width_m(alt_m) * (1 - SIDELAP)


def sector_scores(
    state: MapState,
    sectors: Sequence[PlanSector],
    canopy: Floats,
    detector: DetectorProfile,
    *,
    alt_m: float,
    speed_ms: float,
) -> list[SectorScore]:
    cx, cy = state.grid.centers
    pod_plan = plan_pod(canopy, alt_m, detector)
    spacing = spacing_m(alt_m)
    out: list[SectorScore] = []
    for s in sectors:
        poly = Polygon(s.corners)
        if not poly.is_valid or poly.area <= 0:
            continue
        inside = shapely.contains_xy(poly, cx, cy)
        n = int(inside.sum())
        poa = float(state.p[inside].sum())
        pos = float((state.p * pod_plan)[inside].sum())
        est_min = poly.area / spacing / speed_ms / 60
        out.append(
            SectorScore(
                sector_id=s.sector_id,
                poa=round(min(1.0, poa), 6),
                covered_pct=round(100 * float((state.pod_cum[inside] >= COVERED_POD).mean()), 1)
                if n
                else 0.0,
                mean_pod=round(float(state.pod_cum[inside].mean()), 4) if n else 0.0,
                pos_plan=round(min(1.0, pos), 6),
                est_min=round(est_min, 2),
                psr=round(pos / est_min, 6) if est_min > 0 else 0.0,
                spacing_m=round(spacing, 1),
            )
        )
    return out


def suggest_sector_order(scores: Sequence[SectorScore]) -> list[SectorScore]:
    """Greedy by PSR, highest first (ties by sector id)."""
    return sorted(scores, key=lambda s: (-s.psr, s.sector_id))


def situation_line(scores: Sequence[SectorScore]) -> str:
    """The AI's situation line, in the suggested order."""
    parts = [
        f"{s.sector_id} {_sv(s.poa, 2)}/{s.covered_pct:.0f}%/{_sv(s.mean_pod, 2)}/{_sv(s.psr, 3)}"
        for s in suggest_sector_order(scores)
    ]
    return "Sektorer (POA kvar / täckt / medel-POD / PSR): " + "  ".join(parts)


def _sv(x: float, n: int) -> str:
    return f"{x:.{n}f}".replace(".", ",")
