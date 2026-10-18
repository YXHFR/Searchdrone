"""The detector interface: one image in, a list of detections out.

Only the names ``Detector`` and ``detect()`` are stable. The interface assumes no model
type: a class-and-score detector (RF-DETR, YOLOX), a colour threshold and an embedding
model all fit. A detection's label is a class id from the class register
(``sd_detect.classes``); ``source`` says what kind of source found it, so a simulator
label (``"oracle"``) is never mistaken for a real model.

Nothing here steers the vehicle or reaches the safety gate: a detection is only a hint
for a person to look at, and only a person confirms a find. There is no face recognition.
"""

from __future__ import annotations

from typing import ClassVar, Literal, Protocol, Self, runtime_checkable

from pydantic import Field, field_validator, model_validator

from sd_detect.classes import CLASS_ID_PATTERN, is_forbidden_class
from sd_schemas.base import SdModel

__all__ = [
    "EMBEDDING_MAX_LEN",
    "FRAME_MAX_SIDE_PX",
    "SOURCES",
    "BBox",
    "Detection",
    "Detector",
    "Frame",
    "SourceKind",
]

FRAME_MAX_SIDE_PX = 8192
EMBEDDING_MAX_LEN = 4096
SourceKind = Literal["model", "color", "oracle"]
SOURCES: tuple[SourceKind, ...] = ("model", "color", "oracle")
"""What found a detection: a trained model, a colour threshold on real pixels, or the
simulator's own labels (only in the simulator; never a measure of real ability)."""


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
    """Something of one class in one image. Never a decision: a person confirms.

    ``label`` is a class id (``sd_detect.classes``). The register is the guard: the
    pipeline (``sd_detect.sources.detect_all``) keeps only labels the register gives to
    the source that reported them, so register membership is checked where detections
    leave a source. The words refused here are only a backstop. ``embedding`` is a
    generic appearance vector for matching against a description. It is never a face or
    other biometric identity vector.
    """

    label: str = Field(pattern=CLASS_ID_PATTERN)
    score: float = Field(ge=0, le=1)
    bbox: BBox
    embedding: list[float] | None = Field(default=None, max_length=EMBEDDING_MAX_LEN)
    source: SourceKind | None = None

    @field_validator("label")
    @classmethod
    def _no_forbidden_class(cls, value: str) -> str:
        if is_forbidden_class(value):
            raise ValueError("a detector never reports faces, plates or other identities")
        return value


@runtime_checkable
class Detector(Protocol):
    """A detector engine. ``license`` is an SPDX id, e.g. ``"Apache-2.0"``."""

    name: ClassVar[str]
    license: ClassVar[str]

    def detect(self, image: Frame) -> list[Detection]: ...
