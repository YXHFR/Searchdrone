"""The detector measures: matching, recall at a fixed false-alarm rate, AP50.

Per picture, detections are matched to the person boxes greedily, highest score first,
each to the free box it overlaps most (IoU >= 0.5); a detection with no box is a false
alarm. The score threshold is then the lowest one at which the false alarms stay at or
below ``fp_per_12mp`` per 12 megapixels of picture area (a picture of another size
counts by its area). Recall is given at that threshold, per size class of the person
box's longest side (``lt16`` < 16 px, ``16to32`` 16-32 px, ``gt32`` > 32 px) and per
dataset, each with a 95 % Wilson interval.
"""

from __future__ import annotations

import math
from collections.abc import Sequence
from dataclasses import dataclass, field

import numpy as np

from sd_detect.boxes import iou_matrix

MATCH_IOU = 0.5
FP_PER_12MP = 0.5
SCORE_MIN = 0.25  # DETECTOR_SCORE_MIN on board
MP12 = 12_000_000
SIZE_CLASSES = ("lt16", "16to32", "gt32")


def size_class(box: Sequence[float]) -> str:
    side = max(box[2] - box[0], box[3] - box[1])
    if side < 16:
        return "lt16"
    return "16to32" if side <= 32 else "gt32"


@dataclass
class PictureResult:
    """One picture after matching."""

    dataset: str
    pixels: int
    truth_class: list[str]  # size class of each person box
    truth_hit_score: list[float | None]  # score of the detection matched to it
    fp_scores: list[float]  # scores of detections matched to nothing
    latency_ms: float = 0.0


def match_picture(
    dataset: str,
    width: int,
    height: int,
    truth: Sequence[Sequence[float]],
    dets: Sequence[tuple[float, Sequence[float]]],
    latency_ms: float = 0.0,
) -> PictureResult:
    """``dets``: ``(score, (x0, y0, x1, y1))``."""
    order = sorted(range(len(dets)), key=lambda i: -dets[i][0])
    t = np.asarray(truth, dtype=np.float64).reshape(-1, 4)
    d = np.asarray([dets[i][1] for i in order], dtype=np.float64).reshape(-1, 4)
    iou = iou_matrix(d, t)
    hit: list[float | None] = [None] * len(t)
    fps: list[float] = []
    for row, i in enumerate(order):
        score = float(dets[i][0])
        best, best_iou = -1, MATCH_IOU
        for j in range(len(t)):
            if hit[j] is None and iou[row, j] >= best_iou:
                best, best_iou = j, float(iou[row, j])
        if best >= 0:
            hit[best] = score
        else:
            fps.append(score)
    return PictureResult(
        dataset,
        width * height,
        [size_class(tuple(float(v) for v in b)) for b in t],
        hit,
        fps,
        latency_ms,
    )


def wilson(hits: int, n: int, z: float = 1.96) -> tuple[float, float]:
    if n == 0:
        return (0.0, 1.0)
    p = hits / n
    centre = (p + z * z / (2 * n)) / (1 + z * z / n)
    half = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / (1 + z * z / n)
    return (max(0.0, centre - half), min(1.0, centre + half))


def threshold_for(results: Sequence[PictureResult], fp_per_12mp: float = FP_PER_12MP) -> float:
    """The lowest score threshold whose false alarms stay within the allowance
    (``inf``: not even the best detection may be kept; ``0``: nothing is ever wrong)."""
    allowed = fp_per_12mp * sum(r.pixels for r in results) / MP12
    fps = sorted((s for r in results for s in r.fp_scores), reverse=True)
    if len(fps) <= allowed:
        return 0.0
    # keeping every detection scored above fps[k] gives k false alarms; ties go together
    k = math.floor(allowed)
    return math.nextafter(fps[k], math.inf)


def ap50(results: Sequence[PictureResult]) -> float:
    """Average precision at IoU 0.5, every-point interpolation."""
    n_truth = sum(len(r.truth_class) for r in results)
    if n_truth == 0:
        return 0.0
    marks = [(s, 1) for r in results for s in r.truth_hit_score if s is not None]
    marks += [(s, 0) for r in results for s in r.fp_scores]
    marks.sort(key=lambda m: -m[0])
    tp = np.cumsum([m[1] for m in marks]) if marks else np.zeros(0)
    fp = np.cumsum([1 - m[1] for m in marks]) if marks else np.zeros(0)
    recall = np.concatenate([[0.0], tp / n_truth, [tp[-1] / n_truth if len(tp) else 0.0]])
    precision = np.concatenate([[1.0], tp / np.maximum(tp + fp, 1), [0.0]])
    for i in range(len(precision) - 2, -1, -1):
        precision[i] = max(precision[i], precision[i + 1])
    return float(np.sum((recall[1:] - recall[:-1]) * precision[1:]))


def _beta_moments(values: Sequence[float]) -> dict[str, float] | None:
    """Beta(a, b) with the sample's mean and variance (``None``: too few values)."""
    v = np.clip(np.asarray(values, dtype=np.float64), 1e-3, 1 - 1e-3)
    if len(v) < 5:
        return None
    mean, var = float(v.mean()), float(v.var(ddof=1))
    if var <= 0 or var >= mean * (1 - mean):
        return None
    common = mean * (1 - mean) / var - 1
    return {"a": round(mean * common, 3), "b": round((1 - mean) * common, 3)}


def _pct(values: Sequence[float], q: float) -> float:
    return float(np.percentile(values, q)) if len(values) else 0.0


@dataclass
class Summary:
    threshold: float
    n_pictures: int
    area_12mp: float
    fp_per_12mp: float  # achieved at the threshold
    recall: dict[str, dict[str, float]]  # size class -> n, hits, recall, lo, hi
    recall_by_dataset: dict[str, dict[str, float]]
    ap50: float
    precision_at_score_min: float | None
    latency_ms: dict[str, float]
    score_hit: dict[str, float] | None = None
    score_fp: dict[str, float] | None = None
    notes: list[str] = field(default_factory=list[str])

    def as_dict(self) -> dict[str, object]:
        return {
            "threshold": None if math.isinf(self.threshold) else round(self.threshold, 4),
            "n_pictures": self.n_pictures,
            "area_12mp": round(self.area_12mp, 3),
            "fp_per_12mp": round(self.fp_per_12mp, 4),
            "recall": self.recall,
            "recall_by_dataset": self.recall_by_dataset,
            "ap50": round(self.ap50, 4),
            "precision_at_score_min": self.precision_at_score_min,
            "latency_ms": self.latency_ms,
            "score_hit": self.score_hit,
            "score_fp": self.score_fp,
            "notes": self.notes,
        }


def _recall_row(hits: int, n: int) -> dict[str, float]:
    lo, hi = wilson(hits, n)
    return {
        "n": n,
        "hits": hits,
        "recall": round(hits / n, 4) if n else 0.0,
        "lo": round(lo, 4),
        "hi": round(hi, 4),
    }


def summarise(results: Sequence[PictureResult], fp_per_12mp: float = FP_PER_12MP) -> Summary:
    t = threshold_for(results, fp_per_12mp)
    area = sum(r.pixels for r in results) / MP12

    def kept(score: float | None) -> bool:
        return score is not None and score >= t

    per_class: dict[str, list[int]] = {c: [0, 0] for c in (*SIZE_CLASSES, "all")}
    per_ds: dict[str, list[int]] = {}
    for r in results:
        ds = per_ds.setdefault(r.dataset, [0, 0])
        for cls, score in zip(r.truth_class, r.truth_hit_score, strict=True):
            for row in (per_class[cls], per_class["all"], ds):
                row[1] += 1
                row[0] += int(kept(score))
    fp_kept = sum(1 for r in results for s in r.fp_scores if s >= t)
    tp_min = sum(1 for r in results for s in r.truth_hit_score if s is not None and s >= SCORE_MIN)
    fp_min = sum(1 for r in results for s in r.fp_scores if s >= SCORE_MIN)
    lat = [r.latency_ms for r in results]
    hit_scores = [s for r in results for s in r.truth_hit_score if s is not None and s >= t]
    fp_scores = [s for r in results for s in r.fp_scores if s >= t]
    return Summary(
        threshold=t,
        n_pictures=len(results),
        area_12mp=area,
        fp_per_12mp=fp_kept / area if area else 0.0,
        recall={c: _recall_row(h, n) for c, (h, n) in per_class.items()},
        recall_by_dataset={d: _recall_row(h, n) for d, (h, n) in sorted(per_ds.items())},
        ap50=ap50(results),
        precision_at_score_min=round(tp_min / (tp_min + fp_min), 4) if tp_min + fp_min else None,
        latency_ms={"p50": round(_pct(lat, 50), 1), "p95": round(_pct(lat, 95), 1)},
        score_hit=_beta_moments(hit_scores),
        score_fp=_beta_moments(fp_scores),
    )
