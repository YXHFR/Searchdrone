"""The search simulator plans with the same return-home reserve as the safety rules.

sd_searchsim may not import sd_schemas (an import contract keeps it standalone), so the
two values are compared here instead.
"""

from __future__ import annotations

from sd_schemas.safety import DEFAULT_BATTERY_RTL_PCT, MIN_BATTERY_RTL_PCT
from sd_searchsim.config import SimConfig


def test_reserve_matches_the_return_home_default() -> None:
    reserve_pct = SimConfig().reserve_frac * 100
    assert reserve_pct == DEFAULT_BATTERY_RTL_PCT
    assert reserve_pct >= MIN_BATTERY_RTL_PCT
