"""What the simulator farm and the AI's grade agree on about a round's files.

``sd_sim.farm.round`` writes ``round.json`` with this schema and ``sd_brain.simeval.farm``
grades it; neither imports the other, so the name lives here.
"""

from __future__ import annotations

from typing import Final

__all__ = ["ROUND_SCHEMA"]

ROUND_SCHEMA: Final = "simfarm.round/v1"
