"""The review queue's priority of a candidate: an order for people, never a filter.

``sqrt(person_p) * (0.35 + 0.65 m) * (1.0 if multi_frame else 0.75)
* (0.7 + 0.3 poa_rank) * (0.8 if refined < 0.3 else 1.0)``, with ``m = 0.5`` when the
match score is unknown and ``poa_rank = 0.5`` until there is a probability map. Never 0:
every candidate stays in the queue, a low one only further down (``PRIORITY_MIN``).
"""

from __future__ import annotations

import math
from typing import Final

__all__ = ["POA_RANK_DEFAULT", "PRIORITY_MIN", "priority"]

POA_RANK_DEFAULT: Final = 0.5  # no probability map yet
PRIORITY_MIN: Final = 0.01


def priority(
    person_p: float,
    match: float | None,
    *,
    multi_frame: bool,
    poa_rank: float = POA_RANK_DEFAULT,
    refined: float | None = None,
) -> float:
    m = 0.5 if match is None else min(1.0, max(0.0, match))
    p = (
        math.sqrt(min(1.0, max(0.0, person_p)))
        * (0.35 + 0.65 * m)
        * (1.0 if multi_frame else 0.75)
        * (0.7 + 0.3 * min(1.0, max(0.0, poa_rank)))
        * (1.0 if refined is None or refined >= 0.3 else 0.8)
    )
    return round(min(1.0, max(PRIORITY_MIN, p)), 4)
