"""The numbers a closer-look run reports (§16.3), from the world's truth."""

from __future__ import annotations

import math

import pytest

from sd_sim.inspect_metrics import InspectMeter
from sd_sim.obstacles import Ground, Scene, SimTree

TARGET = (10.0, 0.0)


def _meter() -> InspectMeter:
    scene = Scene((-35.0, 149.0), 584.0, 200.0, Ground(), (SimTree(0.0, 0.0, 20.0),))
    return InspectMeter(scene, TARGET)


def test_a_descent_beside_a_tree() -> None:
    meter = _meter()
    meter.sample(0.0, 0.0, 0.0, 60.0, floor_z=37.0)
    meter.sample(10.0, 5.0, 0.0, 40.0, floor_z=37.0)
    meter.sample(20.0, 8.0, 0.0, 35.5, floor_z=37.0)  # 1.5 m under the floor
    meter.sample(30.0, 8.0, 0.0, 50.0, floor_z=None)
    m = meter.result(outcome="floor_reached", px_first={"main": 30.0}, px_last={"main": 80.0})
    # Nearest: 8 m out and 35.5 m up from the upper crown (radius 1.6 m, top 19.5 m).
    assert m.min_clearance_m == pytest.approx(math.hypot(8.0 - 1.6, 35.5 - 19.5), abs=0.01)
    assert m.min_target_horizontal_m == pytest.approx(2.0)
    assert m.floor_breach_m == pytest.approx(1.5)
    assert m.lowest_above_ground_m == pytest.approx(35.5)
    assert m.inspect_duration_s == 30.0
    assert (m.outcome, m.target_px_last_step) == ("floor_reached", {"main": 80.0})


def test_never_under_the_floor_is_no_breach_and_no_samples_is_an_error() -> None:
    meter = _meter()
    meter.sample(0.0, 0.0, 0.0, 30.0, floor_z=25.0)
    assert meter.result().floor_breach_m == 0.0
    assert meter.result().min_clearance_m == pytest.approx(10.0, abs=0.01)  # over the trunk
    with pytest.raises(ValueError):
        _meter().result()
