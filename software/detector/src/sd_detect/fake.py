"""``NullDetector``: the only detector in v0.1.0. It never finds anything.

It is a stand-in so that later stages can plug a real engine in behind ``Detector``
without changing the rest. It does not pretend to work: ``built`` is ``False`` and the
reason says so.
"""

from __future__ import annotations

from typing import ClassVar

from sd_detect.api import Detection, Frame

__all__ = ["NOT_BUILT", "NullDetector"]

NOT_BUILT = "Persondetektorn är inte byggd än (kommer i etapp 2). Inget letas i bilden."


class NullDetector:
    name: ClassVar[str] = "null"
    license: ClassVar[str] = "Apache-2.0"
    built: ClassVar[bool] = False
    reason: ClassVar[str] = NOT_BUILT

    def detect(self, image: Frame) -> list[Detection]:
        del image  # deliberately unused: nothing is looked for
        return []
