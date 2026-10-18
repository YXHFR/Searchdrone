"""Reference values for hardware/drone.yaml (safety manifest id HW-01).

The committed drone and limits files must give exactly these numbers. A change to the
formula, a unit or the files that moves any value fails here, so it cannot happen
silently.
"""

from pathlib import Path

import pytest

from sd_hwcalc.calc import check, compute
from sd_hwcalc.model import load_drone, load_limits

HARDWARE = Path(__file__).resolve().parents[4] / "hardware"


def test_reference_values() -> None:
    perf = compute(load_drone(HARDWARE / "drone.yaml"))
    assert perf.total_mass_kg == pytest.approx(2.300, abs=0.001)
    assert perf.hover_power_ideal_w == pytest.approx(108.58, abs=0.1)
    assert perf.hover_power_w == pytest.approx(283.45, abs=0.1)
    assert perf.usable_energy_wh == pytest.approx(172.80, abs=0.1)
    assert perf.hover_time_min == pytest.approx(36.58, abs=0.05)
    assert perf.hover_thrust_fraction == pytest.approx(0.359, abs=0.005)
    assert perf.thrust_to_weight == pytest.approx(2.78, abs=0.005)


def test_committed_design_passes_committed_limits() -> None:
    report = check(load_drone(HARDWARE / "drone.yaml"), load_limits(HARDWARE / "limits.yaml"))
    assert report.passed, report.failures
