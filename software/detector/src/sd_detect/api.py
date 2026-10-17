"""The person-detector interface: one image in, a list of detections out.

Only the names ``Detector`` and ``detect()`` are stable. The fields of ``Frame`` and
``Detection`` are preliminary and may grow before stage 2. The interface assumes no model
type: a class-and-score detector (RF-DETR, YOLOX) and an embedding model both fit.

Nothing here steers the vehicle or reaches the safety gate: a detection is only a hint
for a person to look at, and only a person confirms a find. There is no face recognition
and no search for anything but people.
"""

from __future__ import annotations

from typing import ClassVar, Literal, Protocol, Self, runtime_checkable

from pydantic import Field, model_validator

from sd_schemas.base import SdModel

__all__ = [
    "EMBEDDING_MAX_LEN",
    "FRAME_MAX_SIDE_PX",
    "LABELS",
    "BBox",
    "Detection",
    "Detector",
    "Frame",
]

FRAME_MAX_SIDE_PX = 8192
EMBEDDING_MAX_LEN = 4096
LABELS: tuple[Literal["person"], ...] = ("person",)
"""The only label a detector may report: Searchdrone searches for people, not objects."""


class Frame(SdModel):
    """One camera image: 8-bit RGB, row by row, no padding."""

    width: int = Field(ge=1, le=FRAME_MAX_SIDE_PX)
    height: int = Field(ge=1, le=FRAME_MAX_SIDE_PX)
    rgb: bytes
    ts_mono_ms: int = Field(ge=0)

    @model_validator(mode="after")
    def _size_matches(self) -> Self:
        if len(self.rgb) != self.width * self.height * 3:
            raise ValueError("rgb must hold exactly width * height * 3 bytes")
        return self


class BBox(SdModel):
    """A box in image pixels: top-left corner, width and height."""

    x: float = Field(ge=0)
    y: float = Field(ge=0)
    w: float = Field(gt=0)
    h: float = Field(gt=0)


class Detection(SdModel):
    """Something that may be a person in one image. Never a decision: a person confirms."""

    label: str
    score: float = Field(ge=0, le=1)
    bbox: BBox
    embedding: list[float] | None = Field(default=None, max_length=EMBEDDING_MAX_LEN)

    @model_validator(mode="after")
    def _person_only(self) -> Self:
        if self.label not in LABELS:
            raise ValueError("a detector may only report people (label 'person')")
        return self


@runtime_checkable
class Detector(Protocol):
    """A detector engine. ``license`` is an SPDX id, e.g. ``"Apache-2.0"``."""

    name: ClassVar[str]
    license: ClassVar[str]

    def detect(self, image: Frame) -> list[Detection]: ...
