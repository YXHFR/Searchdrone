"""What the simulator farm and the AI's grade agree on about a round's files.

``sd_sim.farm.round`` writes ``round.json`` with this schema and ``sd_brain.simeval.farm``
grades it; neither imports the other, so the name lives here. So does the rule for when
the search flew on from a raised candidate, which the exam's world and the farm both apply.
"""

from __future__ import annotations

from typing import Final

__all__ = ["CLOSE_UP_RADIUS_M", "FLEW_ON_M", "ROUND_SCHEMA", "SEARCH_SKILLS"]

ROUND_SCHEMA: Final = "simfarm.round/v1"
# The search flew on from a raised candidate: this much flown on a search order after the
# raise, a goto that ends this close to the candidate not counted.
FLEW_ON_M: Final = 30.0
CLOSE_UP_RADIUS_M: Final = 15.0
SEARCH_SKILLS: Final = frozenset({"goto", "survey_sector"})
