"""``SimLabelDetector``: the simulator's own labels as detections (source ``"oracle"``).

The simulator's bounding-box camera knows the label number of every labelled object in
view. For each register class whose source is ``"oracle"`` this source turns those boxes
into detections with score 1 and ``source = "oracle"``. It is ground truth from the
simulator, not a model: nothing it reports says anything about what a real detector
would see, and it refuses to start unless ``SD_SIM=1`` (the simulator sets it).
"""

from __future__ import annotations

import os
from collections.abc import Mapping
from typing import ClassVar, Self

from sd_detect.api import Detection, SourceKind
from sd_detect.classes import ClassRegistry
from sd_detect.still import Still

__all__ = ["NOT_IN_SIMULATOR", "SimLabelDetector"]

NOT_IN_SIMULATOR = (
    "Facitkällan (simulatorns etiketter) körs bara i simulatorn (SD_SIM=1). "
    "The oracle source runs only in the simulator."
)


class SimLabelDetector:
    name: ClassVar[str] = "sim_labels"
    license: ClassVar[str] = "Apache-2.0"
    source: ClassVar[SourceKind] = "oracle"

    def __init__(self, labels: Mapping[int, str], env: Mapping[str, str] | None = None) -> None:
        if (os.environ if env is None else env).get("SD_SIM") != "1":
            raise RuntimeError(NOT_IN_SIMULATOR)
        self.labels = dict(labels)

    @classmethod
    def from_registry(cls, registry: ClassRegistry, env: Mapping[str, str] | None = None) -> Self:
        oracle = {spec.id for spec in registry.by_source("oracle")}
        labels = {n: spec.id for n, spec in registry.by_gz_label().items() if spec.id in oracle}
        return cls(labels, env)

    def detect_still(self, still: Still) -> list[Detection]:
        out: list[Detection] = []
        for box in still.sim_boxes or ():
            label = self.labels.get(box.gz_label)
            if label is not None:
                out.append(Detection(label=label, score=1.0, bbox=box.bbox, source="oracle"))
        return out
