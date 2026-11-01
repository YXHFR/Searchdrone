"""Each hardware limit fails on its own, and only when it is broken."""

from pathlib import Path
from typing import Any

import pytest
import yaml
from pydantic import ValidationError

from sd_hwcalc.calc import check, compute
from sd_hwcalc.model import Drone, HardwareLimits

HARDWARE = Path(__file__).resolve().parents[4] / "hardware"


def raw(name: str) -> dict[str, Any]:
    return yaml.safe_load((HARDWARE / name).read_text(encoding="utf-8"))


def drone(**changes: Any) -> Drone:
    data = raw("drone.yaml")
    for dotted, value in changes.items():
        *parents, leaf = dotted.split("__")
        node = data
        for key in parents:
            node = node[key]
        node[leaf] = value
    return Drone.model_validate(data)


LIMITS = HardwareLimits.model_validate(raw("limits.yaml"))


def failed(d: Drone, limits: HardwareLimits = LIMITS) -> set[str]:
    return {f.limit for f in check(d, limits).failures}


def frame_for_total(total_kg: float) -> float:
    """The frame mass that brings the committed drone to ``total_kg`` in all."""
    d = drone()
    return total_kg - (compute(d).total_mass_kg - d.frame_mass_kg)


def test_mass_above_takeoff_limit_fails() -> None:
    heavy = drone(frame_mass_kg=frame_for_total(LIMITS.max_takeoff_mass_kg + 0.01))
    assert compute(heavy).total_mass_kg > LIMITS.max_takeoff_mass_kg
    assert "max_takeoff_mass_kg" in failed(heavy)


def test_mass_exactly_at_limit_passes() -> None:
    # The limit is set to the drone's own total, so "at the limit" is exact in floats.
    d = drone()
    total = compute(d).total_mass_kg
    at_limit = LIMITS.model_copy(update={"max_takeoff_mass_kg": total})
    assert "max_takeoff_mass_kg" not in failed(d, at_limit)
    just_below = LIMITS.model_copy(update={"max_takeoff_mass_kg": total - 1e-9})
    assert "max_takeoff_mass_kg" in failed(d, just_below)


def test_short_hover_time_fails() -> None:
    assert failed(drone(battery__capacity_mah=6000)) == {"min_hover_time_min"}


def test_weak_motors_fail_thrust_checks() -> None:
    assert failed(drone(motors__max_thrust_kg_per_motor=1.0)) == {
        "max_hover_thrust_fraction",
        "min_thrust_to_weight",
    }


def test_failure_messages_are_swedish_and_name_the_values() -> None:
    (failure,) = check(drone(battery__capacity_mah=6000), LIMITS).failures
    assert "Svävtid" in failure.message
    assert failure.bound == 25


def test_hover_time_scales_with_battery() -> None:
    small = compute(drone(battery__capacity_mah=5000)).usable_energy_wh
    assert small == pytest.approx(compute(drone()).usable_energy_wh / 2)


@pytest.mark.parametrize(
    "changes",
    [
        {"props__count": 6},  # one propeller per motor
        {"frame_mass_kg": -0.1},
        {"frame_mass_kg": "0.55"},  # strings are not numbers
        {"motors__count": True},  # booleans are not counts
        {"propulsion__efficiency_total": 1.2},
        {"propulsion__efficiency_total": 0.81},  # above the plausibility bound
        {"environment__air_density_kg_m3": 1000.0},
        {"environment__air_density_kg_m3": 0.5},
        {"battery__capacity_mah": 30000},  # 648 Wh in the committed pack: > 300 Wh/kg
        {"schema_version": True},
        {"battery__usable_fraction": 0},
        {"environment__air_density_kg_m3": float("nan")},
        {"schema_version": 2},
        {"unknown_field": 1},
    ],
)
def test_invalid_drone_file_is_rejected(changes: dict[str, Any]) -> None:
    with pytest.raises(ValidationError):
        drone(**changes)


def test_missing_field_is_rejected() -> None:
    data = raw("drone.yaml")
    del data["battery"]["mass_kg"]
    with pytest.raises(ValidationError):
        Drone.model_validate(data)


def test_limits_reject_unknown_fields() -> None:
    with pytest.raises(ValidationError):
        HardwareLimits.model_validate({**raw("limits.yaml"), "max_takeoff_mass": 3})


def test_non_finite_result_fails_closed() -> None:
    # Inputs that pass the schema but overflow in the formula must never pass.
    d = drone().model_copy(
        update={"battery": drone().battery.model_copy(update={"capacity_mah": 1.0e308})}
    )
    report = check(d, LIMITS)
    assert not report.passed
    assert [f.limit for f in report.failures] == ["finite_values"]


def test_duplicate_key_is_rejected(tmp_path: Path) -> None:
    from sd_hwcalc.model import load_drone

    text = (HARDWARE / "drone.yaml").read_text(encoding="utf-8")
    bad = tmp_path / "drone.yaml"
    bad.write_text(text + "frame_mass_kg: 0.1\n", encoding="utf-8")
    with pytest.raises(yaml.YAMLError, match="duplicate key"):
        load_drone(bad)


def test_unhashable_key_is_rejected(tmp_path: Path) -> None:
    from sd_hwcalc.model import load_drone

    bad = tmp_path / "drone.yaml"
    bad.write_text("? [a]\n: 1\n", encoding="utf-8")
    with pytest.raises(yaml.YAMLError, match="unhashable key"):
        load_drone(bad)
