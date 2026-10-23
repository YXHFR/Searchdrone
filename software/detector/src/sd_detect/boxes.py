"""Box arithmetic shared by the tiling and the image bank.

Boxes are ``(x0, y0, x1, y1)`` in pixels as float arrays of shape ``(n, 4)``; ``BBox``
(top-left corner, width, height) is converted at the edges.
"""

from __future__ import annotations

import numpy as np
import numpy.typing as npt

from sd_detect.api import BBox

__all__ = ["Boxes", "from_bbox", "iou_matrix", "nms", "to_bbox"]

Boxes = npt.NDArray[np.float64]


def from_bbox(boxes: list[BBox]) -> Boxes:
    if not boxes:
        return np.zeros((0, 4), dtype=np.float64)
    return np.array([[b.x, b.y, b.x + b.w, b.y + b.h] for b in boxes], dtype=np.float64)


def to_bbox(box: npt.ArrayLike) -> BBox:
    x0, y0, x1, y1 = (float(v) for v in np.asarray(box, dtype=np.float64))
    return BBox(x=max(0.0, x0), y=max(0.0, y0), w=max(x1 - x0, 1e-6), h=max(y1 - y0, 1e-6))


def iou_matrix(a: Boxes, b: Boxes) -> npt.NDArray[np.float64]:
    """Intersection over union of every box in ``a`` with every box in ``b``."""
    if len(a) == 0 or len(b) == 0:
        return np.zeros((len(a), len(b)), dtype=np.float64)
    x0 = np.maximum(a[:, None, 0], b[None, :, 0])
    y0 = np.maximum(a[:, None, 1], b[None, :, 1])
    x1 = np.minimum(a[:, None, 2], b[None, :, 2])
    y1 = np.minimum(a[:, None, 3], b[None, :, 3])
    inter = np.clip(x1 - x0, 0, None) * np.clip(y1 - y0, 0, None)
    area_a = (a[:, 2] - a[:, 0]) * (a[:, 3] - a[:, 1])
    area_b = (b[:, 2] - b[:, 0]) * (b[:, 3] - b[:, 1])
    union = area_a[:, None] + area_b[None, :] - inter
    return np.where(union > 0, inter / np.maximum(union, 1e-12), 0.0)


def nms(boxes: Boxes, scores: npt.ArrayLike, iou: float) -> list[int]:
    """Indices of the boxes kept by greedy non-maximum suppression, highest score first:
    a box is dropped when it overlaps a kept one by ``iou`` or more."""
    order = list(np.argsort(-np.asarray(scores, dtype=np.float64), kind="stable"))
    keep: list[int] = []
    while order:
        best = order.pop(0)
        keep.append(int(best))
        if not order:
            break
        overlap = iou_matrix(boxes[[best]], boxes[order])[0]
        order = [i for i, o in zip(order, overlap, strict=True) if o < iou]
    return keep
