"""Where stills come from.

``DirStillSource`` reads the folder a camera process writes: ``<name>.ppm`` (8-bit RGB)
plus ``<name>.json`` with the size and, in the simulator, the bounding-box camera's
labelled boxes (``boxes``: ``label``, ``x0``, ``y0``, ``x1``, ``y1``). The JSON file is
written last, so a still is complete once its JSON exists. The source returns the newest
complete still and deletes it and every older one: a detector that falls behind skips
stills instead of building a backlog. A still larger than the configured camera is
refused before it is read.
"""

from __future__ import annotations

import json
import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Protocol

from sd_detect import BBox, Frame, SimBox, Still

__all__ = ["MAX_SIDE_PX", "CapturedStill", "DirStillSource", "StillSource", "read_ppm"]

log = logging.getLogger("sd_onboard.perception")
MAX_SIDE_PX = 8192


@dataclass(frozen=True)
class CapturedStill:
    still: Still
    name: str
    synthetic: bool  # rendered by the simulator, not a camera


class StillSource(Protocol):
    def poll(self) -> CapturedStill | None: ...


Size = tuple[int, int]
_HEADER_MAX = 64  # bytes a PPM header may take


def read_ppm(data: bytes, max_size: Size = (MAX_SIDE_PX, MAX_SIDE_PX)) -> Frame:
    """A binary PPM (``P6``, maxval 255, at most ``max_size``) as a frame."""
    parts: list[bytes] = []
    pos = 0
    while len(parts) < 4:
        while pos < len(data) and data[pos : pos + 1].isspace():
            pos += 1
        start = pos
        while pos < len(data) and not data[pos : pos + 1].isspace():
            pos += 1
        if start == pos:
            raise ValueError("truncated PPM header")
        parts.append(data[start:pos])
    pos += 1  # the single whitespace byte after maxval
    if parts[0] != b"P6" or parts[3] != b"255":
        raise ValueError("only binary 8-bit PPM (P6, 255) is read")
    width, height = int(parts[1]), int(parts[2])
    if not (0 < width <= max_size[0] and 0 < height <= max_size[1]):
        raise ValueError(f"PPM size {width}x{height} out of range")
    rgb = data[pos : pos + width * height * 3]
    if len(rgb) != width * height * 3:
        raise ValueError("truncated PPM pixel data")
    return Frame(width=width, height=height, rgb=rgb, ts_mono_ms=0)


class DirStillSource:
    def __init__(
        self, folder: Path, *, synthetic: bool, max_size: Size = (MAX_SIDE_PX, MAX_SIDE_PX)
    ) -> None:
        self.folder = folder
        self.synthetic = synthetic
        self.max_size = max_size
        self.skipped = 0

    def poll(self) -> CapturedStill | None:
        if not self.folder.is_dir():
            return None
        ready = sorted(self.folder.glob("[!.]*.json"))
        if not ready:
            return None
        newest = ready[-1]
        for old in ready[:-1]:
            self._remove(old)
            self.skipped += 1
        try:
            meta = json.loads(newest.read_text(encoding="utf-8"))
            ppm = newest.with_suffix(".ppm")
            if ppm.stat().st_size > self.max_size[0] * self.max_size[1] * 3 + _HEADER_MAX:
                raise ValueError("still larger than the camera")
            frame = read_ppm(ppm.read_bytes(), self.max_size)
            boxes = meta.get("boxes")
            sim_boxes = (
                None
                if boxes is None
                else tuple(
                    SimBox(
                        gz_label=int(b["label"]),
                        bbox=BBox(
                            x=max(0.0, float(b["x0"])),
                            y=max(0.0, float(b["y0"])),
                            w=max(1.0, float(b["x1"]) - float(b["x0"])),
                            h=max(1.0, float(b["y1"]) - float(b["y0"])),
                        ),
                    )
                    for b in boxes
                )
            )
            return CapturedStill(
                Still(frame=frame, sim_boxes=sim_boxes), newest.stem, self.synthetic
            )
        except (OSError, ValueError, KeyError, TypeError) as exc:
            log.warning("unreadable still %s: %s", newest.name, exc)
            return None
        finally:
            self._remove(newest)

    @staticmethod
    def _remove(meta: Path) -> None:
        for path in (meta.with_suffix(".ppm"), meta):
            path.unlink(missing_ok=True)
