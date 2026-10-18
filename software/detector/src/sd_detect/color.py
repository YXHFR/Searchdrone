"""``ColorDetector``: finds patches of one colour in the real pixels of a picture.

For every register class whose source is ``"color"`` it marks the pixels inside the
class's hue, saturation and value ranges, joins touching pixels (8-neighbourhood) into
patches and reports each patch of at least ``min_area_px`` pixels as a box. A patch that
touches the picture's edge is not reported: part of it is outside, so its size cannot be
judged (the next still, which overlaps this one, sees it whole). It knows nothing about
size on the ground: the pipeline drops patches that are too small or too
large once their size in metres is known, so a red car never counts as a red block.

The score says how sure the colour is (how saturated the patch is and how much of its
box it fills); it is not a probability that the thing is what the class says.
"""

from __future__ import annotations

from collections import deque
from dataclasses import dataclass
from typing import Any, ClassVar, Self

import numpy as np
import numpy.typing as npt

from sd_detect.api import BBox, Detection, Frame, SourceKind
from sd_detect.classes import ClassRegistry, ClassSpec, RegistryError
from sd_detect.still import Still

__all__ = ["MAX_PATCHES", "ColorDetector", "ColorRule", "hsv"]

MAX_PATCHES = 64  # per class and picture; more is noise, not finds
_NEIGHBOURS = ((-1, -1), (-1, 0), (-1, 1), (0, -1), (0, 1), (1, -1), (1, 0), (1, 1))

FloatImage = npt.NDArray[np.float32]


@dataclass(frozen=True)
class ColorRule:
    """The colour ranges of one class (hue in degrees 0-360, the rest 0-1)."""

    label: str
    hue_deg: tuple[tuple[float, float], ...]
    min_saturation: float
    min_value: float
    min_area_px: int

    @classmethod
    def from_spec(cls, spec: ClassSpec) -> Self:
        p: dict[str, Any] = spec.params
        try:
            hue = tuple((float(lo), float(hi)) for lo, hi in p["hue_deg"])
            rule = cls(
                label=spec.id,
                hue_deg=hue,
                min_saturation=float(p["min_saturation"]),
                min_value=float(p["min_value"]),
                min_area_px=int(p["min_area_px"]),
            )
        except (KeyError, TypeError, ValueError) as exc:
            raise RegistryError(f"class {spec.id!r}: bad colour settings ({exc})") from exc
        if not hue or any(not 0 <= lo < hi <= 360 for lo, hi in hue):
            raise RegistryError(f"class {spec.id!r}: hue_deg ranges must lie in 0-360")
        if rule.min_area_px < 1:
            raise RegistryError(f"class {spec.id!r}: min_area_px must be at least 1")
        return rule


def hsv(frame: Frame) -> tuple[FloatImage, FloatImage, FloatImage]:
    """Hue in degrees (0-360), saturation and value (0-1) of every pixel."""
    rgb = np.frombuffer(frame.rgb, dtype=np.uint8).reshape(frame.height, frame.width, 3)
    f = rgb.astype(np.float32) / 255.0
    r, g, b = f[..., 0], f[..., 1], f[..., 2]
    top = f.max(axis=2)
    span = top - f.min(axis=2)
    safe = np.where(span > 0, span, 1.0)
    hue = np.where(
        top == r,
        ((g - b) / safe) % 6.0,
        np.where(top == g, (b - r) / safe + 2.0, (r - g) / safe + 4.0),
    )
    hue = np.where(span > 0, hue * 60.0, 0.0).astype(np.float32)
    sat = np.where(top > 0, span / np.where(top > 0, top, 1.0), 0.0).astype(np.float32)
    return hue, sat, top.astype(np.float32)


def _patches(mask: npt.NDArray[np.bool_]) -> list[tuple[int, int, int, int, int]]:
    """Touching pixels joined: (x0, y0, x1, y1, pixel count) per patch, x1/y1 inclusive."""
    height, width = mask.shape
    seen = np.zeros_like(mask)
    out: list[tuple[int, int, int, int, int]] = []
    ys, xs = np.nonzero(mask)
    for y0, x0 in zip(ys.tolist(), xs.tolist(), strict=True):
        if seen[y0, x0]:
            continue
        seen[y0, x0] = True
        queue = deque([(y0, x0)])
        lo_x = hi_x = x0
        lo_y = hi_y = y0
        count = 0
        while queue:
            y, x = queue.popleft()
            count += 1
            lo_x, hi_x = min(lo_x, x), max(hi_x, x)
            lo_y, hi_y = min(lo_y, y), max(hi_y, y)
            for dy, dx in _NEIGHBOURS:
                ny, nx = y + dy, x + dx
                if 0 <= ny < height and 0 <= nx < width and mask[ny, nx] and not seen[ny, nx]:
                    seen[ny, nx] = True
                    queue.append((ny, nx))
        out.append((lo_x, lo_y, hi_x, hi_y, count))
    return out


class ColorDetector:
    name: ClassVar[str] = "color"
    license: ClassVar[str] = "Apache-2.0"
    source: ClassVar[SourceKind] = "color"

    def __init__(self, rules: list[ColorRule]) -> None:
        self.rules = rules

    @classmethod
    def from_registry(cls, registry: ClassRegistry) -> Self:
        return cls([ColorRule.from_spec(spec) for spec in registry.by_source("color")])

    def detect(self, image: Frame) -> list[Detection]:
        if not self.rules:
            return []
        hue, sat, val = hsv(image)
        out: list[Detection] = []
        for rule in self.rules:
            in_hue = np.zeros(hue.shape, dtype=np.bool_)
            for lo, hi in rule.hue_deg:
                in_hue |= (hue >= lo) & (hue <= hi)
            mask = in_hue & (sat >= rule.min_saturation) & (val >= rule.min_value)
            found: list[Detection] = []
            for x0, y0, x1, y1, count in _patches(mask):
                if count < rule.min_area_px:
                    continue
                if x0 == 0 or y0 == 0 or x1 == image.width - 1 or y1 == image.height - 1:
                    continue
                w, h = x1 - x0 + 1, y1 - y0 + 1
                fill = count / (w * h)
                patch_sat = float(
                    sat[y0 : y1 + 1, x0 : x1 + 1][mask[y0 : y1 + 1, x0 : x1 + 1]].mean()
                )
                score = min(0.99, max(0.05, 0.5 * fill + 0.5 * patch_sat))
                found.append(
                    Detection(
                        label=rule.label,
                        score=round(score, 3),
                        bbox=BBox(x=x0, y=y0, w=w, h=h),
                        source="color",
                    )
                )
            found.sort(key=lambda d: d.score, reverse=True)
            out.extend(found[:MAX_PATCHES])
        return out

    def detect_still(self, still: Still) -> list[Detection]:
        return self.detect(still.frame)
