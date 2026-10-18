"""A still: one full-resolution picture for the detector sources, plus what only the
simulator knows about it.

``sim_boxes`` carries the simulator's bounding-box camera output for the same moment
(the label number of every labelled object in view and its box). Only the simulator
fills it; a real camera never does, and the oracle source refuses to run outside it.
"""

from __future__ import annotations

from typing import ClassVar, Protocol, runtime_checkable

from pydantic import Field

from sd_detect.api import BBox, Detection, Frame, SourceKind
from sd_schemas.base import SdModel

__all__ = ["MAX_SIM_BOXES", "SimBox", "Still", "StillDetector"]

MAX_SIM_BOXES = 512


class SimBox(SdModel):
    """One labelled object in view, from the simulator's bounding-box camera."""

    gz_label: int = Field(ge=1, le=65535)
    bbox: BBox


class Still(SdModel):
    frame: Frame
    sim_boxes: tuple[SimBox, ...] | None = Field(default=None, max_length=MAX_SIM_BOXES)


@runtime_checkable
class StillDetector(Protocol):
    """A detector source: it reports the register's classes that name it as ``source``."""

    name: ClassVar[str]
    license: ClassVar[str]
    source: ClassVar[SourceKind]

    def detect_still(self, still: Still) -> list[Detection]: ...
