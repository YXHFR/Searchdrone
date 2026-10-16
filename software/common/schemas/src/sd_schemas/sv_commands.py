"""Chat messages that mean STOP or FLY HOME.

Skeleton only. The stop check that fills this in comes with the AI commander; it may
only ever return ``"stop"`` or ``"return_home"``, never a release of a stop.
"""

from __future__ import annotations

from typing import Literal

__all__ = ["OverrideIntent", "classify_override"]

OverrideIntent = Literal["stop", "return_home"]


def classify_override(text: str) -> OverrideIntent | None:
    """Not built yet: no chat text triggers an override. The STOP and FLY HOME buttons do."""
    del text
    return None
