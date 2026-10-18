"""The farm round in Gazebo: fly a seeded search with the chosen commander and keep the stills.

Not built yet: it needs the stills camera with its bounding-box camera in Gazebo and the
search scenarios that use it. Until then a round stops with "inte byggt än" (exit 2) and
``just sim-farm`` stops with it.
"""

from __future__ import annotations

from pathlib import Path

from sd_sim.farm.round import NotBuiltError, RawRound, RoundSpec

TICKET = "simfarmens Gazebo-varv, kräver stillbildskameran i Gazebo"


class GazeboRunner:
    def run(self, spec: RoundSpec, work: Path) -> RawRound:
        raise NotBuiltError(f"inte byggt än ({TICKET}); varv {spec.round_id}")
