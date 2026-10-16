"""The file formats of ``hardware/drone.yaml`` and ``hardware/limits.yaml``.

Every field is required and unknown fields are rejected, so a typo can never fall back to
a silent default. Units are in the field names. Values that can never be physical or
plausible (a negative mass, an efficiency above 0.8, a battery better than today's cells)
are rejected here; whether a valid design is good
enough is decided by the limits in :mod:`sd_hwcalc.calc`.
"""

from __future__ import annotations

from pathlib import Path
from typing import Annotated, Any, Self

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
SchemaVersion = Annotated[int, Field(ge=1, le=1)]  # strict int: true is not 1

# Plausibility bounds: values outside them are typos or wishful thinking, and each one
# alone could turn a failing airframe into a passing one without moving any mass.
MAX_PROPULSION_EFFICIENCY = 0.8  # rotor figure of merit x motor x ESC, realistically <= 0.7
MIN_AIR_DENSITY_KG_M3 = 0.9  # about 3 000 m altitude
MAX_AIR_DENSITY_KG_M3 = 1.4  # cold air at sea level
MAX_PACK_ENERGY_WH_PER_KG = 300.0  # nominal pack energy per pack mass, best cells today


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

    @model_validator(mode="after")
    def _plausible_energy_density(self) -> Self:
        wh = self.cells_series * self.nominal_cell_v * self.capacity_mah / 1000
        if not wh / self.mass_kg <= MAX_PACK_ENERGY_WH_PER_KG:
            msg = (
                f"battery: {wh:.0f} Wh in {self.mass_kg} kg is above "
                f"{MAX_PACK_ENERGY_WH_PER_KG:.0f} Wh/kg"
            )
            raise ValueError(msg)
        return self


class Propulsion(_Strict):
    efficiency_total: Annotated[float, Field(gt=0, le=MAX_PROPULSION_EFFICIENCY)]


class Environment(_Strict):
    air_density_kg_m3: Annotated[float, Field(ge=MIN_AIR_DENSITY_KG_M3, le=MAX_AIR_DENSITY_KG_M3)]


class Drone(_Strict):
    schema_version: SchemaVersion
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
    schema_version: SchemaVersion
    max_takeoff_mass_kg: Positive
    min_hover_time_min: Positive
    max_hover_thrust_fraction: Fraction
    min_thrust_to_weight: Positive


class _UniqueKeyLoader(yaml.SafeLoader):
    """SafeLoader that rejects a key given twice (YAML would keep the last one silently)."""


def _unique_mapping(loader: yaml.SafeLoader, node: yaml.MappingNode, deep: bool = False) -> Any:
    keys: set[Any] = set()
    for key_node, _ in node.value:
        key = loader.construct_object(key_node, deep=deep)
        try:
            hash(key)
        except TypeError:
            raise yaml.constructor.ConstructorError(
                None, None, "unhashable key", key_node.start_mark
            ) from None
        if key in keys:
            raise yaml.constructor.ConstructorError(
                None, None, f"duplicate key {key!r}", key_node.start_mark
            )
        keys.add(key)
    return loader.construct_mapping(node, deep=deep)


_UniqueKeyLoader.add_constructor(yaml.resolver.BaseResolver.DEFAULT_MAPPING_TAG, _unique_mapping)


def _read_yaml(path: Path) -> object:
    with path.open(encoding="utf-8") as f:
        return yaml.load(f, Loader=_UniqueKeyLoader)  # noqa: S506  (SafeLoader subclass)


def load_drone(path: Path) -> Drone:
    return Drone.model_validate(_read_yaml(path))


def load_limits(path: Path) -> HardwareLimits:
    return HardwareLimits.model_validate(_read_yaml(path))
