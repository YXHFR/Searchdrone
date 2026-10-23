"""AP50 on the val tiles, to keep the best epoch while training.

Only a guide for picking the epoch: the measure that counts is ``just bench-detector``
on whole pictures (``tools/bench/metrics.py``), with the same matching rule (greedy,
highest score first, IoU >= 0.5).
"""

from __future__ import annotations

from collections.abc import Sequence

import numpy as np

MATCH_IOU = 0.5
NMS_IOU = 0.5


def _iou(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    if len(a) == 0 or len(b) == 0:
        return np.zeros((len(a), len(b)))
    x0 = np.maximum(a[:, None, 0], b[None, :, 0])
    y0 = np.maximum(a[:, None, 1], b[None, :, 1])
    x1 = np.minimum(a[:, None, 2], b[None, :, 2])
    y1 = np.minimum(a[:, None, 3], b[None, :, 3])
    inter = np.clip(x1 - x0, 0, None) * np.clip(y1 - y0, 0, None)
    area_a = (a[:, 2] - a[:, 0]) * (a[:, 3] - a[:, 1])
    area_b = (b[:, 2] - b[:, 0]) * (b[:, 3] - b[:, 1])
    return inter / np.maximum(area_a[:, None] + area_b[None, :] - inter, 1e-9)


def nms(boxes: np.ndarray, scores: np.ndarray, iou: float = NMS_IOU, top: int = 300) -> list[int]:
    order = list(np.argsort(-scores)[: top * 10])
    keep: list[int] = []
    while order and len(keep) < top:
        i = order.pop(0)
        keep.append(int(i))
        if order:
            overlap = _iou(boxes[[i]], boxes[order])[0]
            order = [j for j, o in zip(order, overlap, strict=True) if o < iou]
    return keep


def ap50(results: Sequence[tuple[np.ndarray, np.ndarray, np.ndarray]]) -> float:
    """``results``: per tile, predicted boxes, their scores and the true boxes."""
    marks: list[tuple[float, int]] = []
    n_truth = 0
    for boxes, scores, truth in results:
        n_truth += len(truth)
        keep = nms(boxes, scores)
        boxes, scores = boxes[keep], scores[keep]
        iou = _iou(boxes, truth)
        taken = np.zeros(len(truth), dtype=bool)
        for row in np.argsort(-scores):
            hit = -1
            if len(truth):
                cand = np.where(~taken & (iou[row] >= MATCH_IOU), iou[row], -1)
                hit = int(np.argmax(cand)) if cand.max() >= MATCH_IOU else -1
            if hit >= 0:
                taken[hit] = True
            marks.append((float(scores[row]), int(hit >= 0)))
    if n_truth == 0:
        return 0.0
    marks.sort(key=lambda m: -m[0])
    tp = np.cumsum([m[1] for m in marks]) if marks else np.zeros(0)
    fp = np.cumsum([1 - m[1] for m in marks]) if marks else np.zeros(0)
    recall = np.concatenate([[0.0], tp / n_truth])
    precision = np.concatenate([[1.0], tp / np.maximum(tp + fp, 1)])
    for i in range(len(precision) - 2, -1, -1):
        precision[i] = max(precision[i], precision[i + 1])
    return float(np.sum((recall[1:] - recall[:-1]) * precision[1:]))
