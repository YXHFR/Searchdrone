"""The quality check of one farm round. A round that fails is kept as an error, never as data.

The checks:

* **took off**: the drone (the camera) was at least ``MIN_FLIGHT_M`` above the ground;
* **pictures**: at least ``MIN_PICTURES`` usable pictures from the air, and at most
  ``MAX_BLACK_SHARE`` of them black or one flat colour;
* **facit**: every person box the simulator's bounding-box camera reports lies inside a
  projected box (``labels.CONTAINED``), the median IoU of the open (unhidden) people is
  at least ``MIN_MEDIAN_IOU``, at most ``MAX_STRAY_SHARE`` of the camera's person boxes
  match no projected person, and the round has at least ``MIN_LABELS`` labels.
"""

from __future__ import annotations

import statistics
from dataclasses import dataclass, field
from typing import Any

from sd_sim.farm.labels import CONTAINED, UNHIDDEN, ImageFacit

MIN_FLIGHT_M = 10.0
MIN_PICTURES = 5
MAX_BLACK_SHARE = 0.1
MIN_MEDIAN_IOU = 0.5
MIN_LABELS = 1
MAX_STRAY_SHARE = 0.1  # person boxes from the camera that match no projected person


@dataclass
class RoundTally:
    """What a round produced, picture by picture."""

    max_height_m: float = 0.0
    pictures: int = 0  # pictures from the air
    black: int = 0
    on_ground: int = 0
    labels: int = 0
    seen: int = 0  # person boxes from the bounding-box camera
    stray: int = 0  # of those, matched to no projected person
    uncontained: int = 0  # matched, but not inside the projected box
    open_ious: list[float] = field(default_factory=list)

    def add(self, facit: ImageFacit) -> None:
        self.labels += len(facit.labels)
        self.stray += len(facit.unmatched_seen)
        self.seen += len(facit.unmatched_seen)
        for p in facit.people:
            if p.seen is None:
                continue
            self.seen += 1
            contain = p.containment
            if contain is not None and contain < CONTAINED:
                self.uncontained += 1
            iou = p.iou
            if iou is not None and p.visible >= UNHIDDEN and p.in_view >= 0.99:
                self.open_ious.append(iou)


@dataclass(frozen=True)
class QcResult:
    ok: bool
    reasons: tuple[str, ...]
    numbers: dict[str, Any]

    def as_dict(self) -> dict[str, Any]:
        return {"ok": self.ok, "reasons": list(self.reasons), **self.numbers}


def check(t: RoundTally) -> QcResult:
    reasons: list[str] = []
    if t.max_height_m < MIN_FLIGHT_M:
        reasons.append(f"drönaren lyfte inte (högst {t.max_height_m:.1f} m över marken)")
    usable = t.pictures - t.black
    if usable < MIN_PICTURES:
        reasons.append(f"för få användbara bilder ({usable}, minst {MIN_PICTURES})")
    if t.pictures and t.black / t.pictures > MAX_BLACK_SHARE:
        reasons.append(f"{t.black} av {t.pictures} bilder är svarta eller enfärgade")
    if t.uncontained:
        reasons.append(f"{t.uncontained} personrutor från etikettkameran ligger utanför facit")
    if t.seen and t.stray / t.seen > MAX_STRAY_SHARE:
        reasons.append(f"{t.stray} av {t.seen} personrutor från etikettkameran saknar facit")
    median = statistics.median(t.open_ious) if t.open_ious else None
    if median is not None and median < MIN_MEDIAN_IOU:
        reasons.append(f"facitrutorna träffar dåligt (median-IoU {median:.2f})")
    if t.labels < MIN_LABELS:
        reasons.append("inga personer i bild under varvet")
    numbers = {
        "max_height_m": round(t.max_height_m, 1),
        "pictures": t.pictures,
        "black": t.black,
        "on_ground": t.on_ground,
        "labels": t.labels,
        "seen": t.seen,
        "stray": t.stray,
        "uncontained": t.uncontained,
        "median_iou": None if median is None else round(median, 3),
        "open_people": len(t.open_ious),
    }
    return QcResult(not reasons, tuple(reasons), numbers)
