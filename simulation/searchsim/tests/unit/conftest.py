"""Shared test helpers for the search simulator."""

from __future__ import annotations

import numpy as np
import pytest

from sd_motion import negative
from sd_searchsim.env import Episode


class IndependentEpisode(Episode):
    """The episode as it was before ``miss_correlation``: every sweep a fresh chance with
    the scenario's own pod, in the truth and the belief. Kept as written then, so the tests
    can check that ``miss_correlation=0`` still gives the same numbers bit for bit."""

    def pod_belief_now(self) -> np.ndarray:
        return self.sc.pod_belief

    def step(self, sector: int) -> None:
        if self.done:
            raise RuntimeError("episode is over")
        if not self.feasible()[sector]:
            raise ValueError(f"sector {sector} is not permitted or does not fit the battery")
        cfg = self.cfg
        transit = float(self._transit_min(self.centres[sector]))
        t_mid = self.t + transit + cfg.sweep_min / 2
        cell = self._true_cell(t_mid)
        if cell is not None and self._sector_of(cell) == sector:
            q = float(self.sc.pod_true[cell])
            found_now = self.survive * q
            self.expected_minutes += found_now * t_mid
            self.survive -= found_now
        self._advance(transit + cfg.sweep_min)
        k, s = cfg.sector_cells, cfg.n_sectors
        r0, c0 = (sector // s) * k, (sector % s) * k
        if self.motion is not None:
            pod = np.zeros_like(self.belief)
            pod[r0 : r0 + k, c0 : c0 + k] = self.sc.pod_belief[r0 : r0 + k, c0 : c0 + k]
            self.motion = negative(self.motion, pod)
            self.belief = self.motion.p
        else:
            b = self.belief.copy()
            b[r0 : r0 + k, c0 : c0 + k] *= 1.0 - self.sc.pod_belief[r0 : r0 + k, c0 : c0 + k]
            self.belief = b / b.sum()
        self.battery -= transit + cfg.sweep_min
        self.pos = self.centres[sector].copy()
        self.swept[sector] += 1
        self.surveyed += 1
        self._ensure_feasible()


@pytest.fixture
def independent_episode() -> type[Episode]:
    return IndependentEpisode
