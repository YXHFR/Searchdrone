"""Settings for the 2D search simulator, and the run sizes for training and evaluation.

Units: metres, minutes. Every value is a model default, not a measured one; the
scenario generator randomises the ones the real world would not tell us exactly.
"""

from __future__ import annotations

import math
from dataclasses import dataclass


@dataclass(frozen=True)
class SimConfig:
    cell_m: float = 25.0
    grid_cells: int = 80  # 80 x 80 cells = 2 x 2 km
    sector_cells: int = 10  # 10 x 10 cells = 250 x 250 m, so 8 x 8 sectors
    transit_speed_mps: float = 10.0
    search_speed_mps: float = 6.0
    swath_m: float = 50.0  # ground width one survey lane sees, overlap already taken off
    turn_s: float = 10.0  # per lane change in a sector survey
    battery_min: float = 25.0
    reserve_frac: float = 0.15  # never planned below this; the gate's own limit still applies
    swap_min: float = 4.0  # land, change battery, take off
    horizon_min: float = 180.0
    # What the belief assumes; the scenario draws the truth around these.
    belief_p_det: float = 0.75
    belief_diffusion_m: float = 15.0  # Gaussian spread per sqrt(minute)
    track_weight: float = 2.0  # prior weight near a track: 1 + track_weight * exp(-d / 50 m)

    @property
    def n_sectors(self) -> int:
        return self.grid_cells // self.sector_cells

    @property
    def sector_m(self) -> float:
        return self.sector_cells * self.cell_m

    @property
    def sweep_min(self) -> float:
        lanes = math.ceil(self.sector_m / self.swath_m)
        path_m = lanes * self.sector_m
        return (path_m / self.search_speed_mps + (lanes - 1) * self.turn_s) / 60.0

    def __post_init__(self) -> None:
        if self.grid_cells % self.sector_cells:
            raise ValueError("grid_cells must be a multiple of sector_cells")


@dataclass(frozen=True)
class RunSize:
    """How much work one training or evaluation run does."""

    iterations: int  # ES iterations
    population: int  # ES candidates per iteration (even: antithetic pairs)
    train_scenarios: int  # scenarios per ES iteration
    val_scenarios: int  # scenarios for picking the best checkpoint
    eval_scenarios: int  # held-out scenarios in the comparison
    val_every: int


SIZES: dict[str, RunSize] = {
    # About a minute on a CI runner; shows the loop works, not a good policy.
    "ci": RunSize(
        iterations=20,
        population=16,
        train_scenarios=16,
        val_scenarios=100,
        eval_scenarios=300,
        val_every=5,
    ),
    # The owner's computer; about two CPU hours, so 15-30 minutes on 4-8 cores.
    "full": RunSize(
        iterations=300,
        population=48,
        train_scenarios=48,
        val_scenarios=400,
        eval_scenarios=1000,
        val_every=10,
    ),
}

# Seed ranges never overlap, so the comparison runs on scenarios training never saw.
TRAIN_SEED_BASE = 0
VAL_SEED_BASE = 10_000_000
EVAL_SEED_BASE = 20_000_000
