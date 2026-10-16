"""The file formats of ``hardware/drone.yaml`` and ``hardware/limits.yaml``.

Every field is required and unknown fields are rejected, so a typo can never fall back to
a silent default. Units are in the field names. Values that can never be physical (a
negative mass, an efficiency above 1) are rejected here; whether a valid design is good
enough is decided by the limits in :mod:`sd_hwcalc.calc`.
"""

from __future__ import annotations

from pathlib import Path
from typing import Annotated, Literal, Self

import yaml
from pydantic import BaseModel, ConfigDict, Field, model_validator

__all__ = [
    "Battery",
    "Drone",
    "Environment",
    "HardwareLimits",
    "Motors",
    "Propellers",
    "Propulsion",
    "load_drone",
    "load_limits",
]

Positive = Annotated[float, Field(gt=0)]
NonNegative = Annotated[float, Field(ge=0)]
Fraction = Annotated[float, Field(gt=0, le=1)]
Count = Annotated[int, Field(ge=1)]


class _Strict(BaseModel):
    # strict: no "1.0" strings or true/false where numbers belong; no inf or nan.
    model_config = ConfigDict(extra="forbid", frozen=True, strict=True, allow_inf_nan=False)


class Motors(_Strict):
    count: Count
    mass_kg: Positive  # per motor
    max_thrust_kg_per_motor: Positive


class Propellers(_Strict):
    count: Count
    diameter_in: Positive
    mass_kg: Positive  # per propeller


class Battery(_Strict):
    chemistry: Annotated[str, Field(min_length=1)]
    cells_series: Count
    cells_parallel: Count
    capacity_mah: Positive  # whole pack
    nominal_cell_v: Positive
    mass_kg: Positive
    usable_fraction: Fraction


class Propulsion(_Strict):
    efficiency_total: Fraction  # rotor figure of merit x motor x ESC


class Environment(_Strict):
    air_density_kg_m3: Positive


class Drone(_Strict):
    schema_version: Literal[1]
    name: Annotated[str, Field(min_length=1)]
    revision: Annotated[str, Field(min_length=1)]
    frame_mass_kg: Positive
    avionics_mass_kg: NonNegative
    avionics_power_w: NonNegative
    payload_mass_kg: NonNegative
    motors: Motors
    props: Propellers
    battery: Battery
    propulsion: Propulsion
    environment: Environment

    @model_validator(mode="after")
    def _one_propeller_per_motor(self) -> Self:
        if self.props.count != self.motors.count:
            msg = f"props.count ({self.props.count}) must equal motors.count ({self.motors.count})"
            raise ValueError(msg)
        return self


class HardwareLimits(_Strict):
    schema_version: Literal[1]
    max_takeoff_mass_kg: Positive
    min_hover_time_min: Positive
    max_hover_thrust_fraction: Fraction
    min_thrust_to_weight: Positive


def _read_yaml(path: Path) -> object:
    with path.open(encoding="utf-8") as f:
        return yaml.safe_load(f)


def load_drone(path: Path) -> Drone:
    return Drone.model_validate(_read_yaml(path))


def load_limits(path: Path) -> HardwareLimits:
    return HardwareLimits.model_validate(_read_yaml(path))
