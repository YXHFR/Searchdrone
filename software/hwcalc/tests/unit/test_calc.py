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


def test_mass_above_takeoff_limit_fails() -> None:
    # 0.55 + 0.21 kg frame takes the total to 2.51 kg.
    assert "max_takeoff_mass_kg" in failed(drone(frame_mass_kg=0.76))


def test_mass_exactly_at_limit_passes() -> None:
    d = drone(frame_mass_kg=0.75)
    assert compute(d).total_mass_kg == pytest.approx(2.5)
    assert "max_takeoff_mass_kg" not in failed(d)


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
    small = compute(drone(battery__capacity_mah=4500)).usable_energy_wh
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
        {"battery__capacity_mah": 30000},  # 648 Wh in 0.87 kg
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
