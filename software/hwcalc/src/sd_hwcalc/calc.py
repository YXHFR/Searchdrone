"""Mass, hover power and hover time of a multirotor, checked against the hardware limits.

Hover power comes from momentum theory: the ideal power of one rotor is
``T^1.5 / sqrt(2 * rho * A)``, divided by the total propulsion efficiency, plus the
avionics' own power. The usable energy is the pack's nominal energy times the usable
fraction. Hover time is energy over power. This is an estimate for comparing designs,
not a promise of flight time; the reference values are pinned by tests.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

from sd_hwcalc.model import Drone, HardwareLimits

__all__ = ["Failure", "G", "Performance", "Report", "check", "compute"]

G = 9.80665  # standard gravity, m/s^2
INCH_M = 0.0254


@dataclass(frozen=True)
class Performance:
    total_mass_kg: float
    hover_power_ideal_w: float
    hover_power_w: float
    usable_energy_wh: float
    hover_time_min: float
    hover_thrust_fraction: float  # hover thrust / maximum thrust ("hover throttle")
    thrust_to_weight: float


@dataclass(frozen=True)
class Failure:
    limit: str
    value: float
    bound: float
    message: str  # Swedish, for the owner


@dataclass(frozen=True)
class Report:
    performance: Performance
    failures: tuple[Failure, ...]

    @property
    def passed(self) -> bool:
        return not self.failures


def compute(drone: Drone) -> Performance:
    n = drone.motors.count
    mass = (
        drone.frame_mass_kg
        + drone.avionics_mass_kg
        + drone.payload_mass_kg
        + drone.motors.count * drone.motors.mass_kg
        + drone.props.count * drone.props.mass_kg
        + drone.battery.mass_kg
    )
    thrust_per_rotor_n = mass * G / n
    disk_area_m2 = math.pi * (drone.props.diameter_in * INCH_M) ** 2 / 4
    rho = drone.environment.air_density_kg_m3
    p_ideal = n * thrust_per_rotor_n**1.5 / math.sqrt(2 * rho * disk_area_m2)
    p_hover = p_ideal / drone.propulsion.efficiency_total + drone.avionics_power_w
    b = drone.battery
    energy_wh = b.cells_series * b.nominal_cell_v * b.capacity_mah / 1000 * b.usable_fraction
    max_thrust_kg = n * drone.motors.max_thrust_kg_per_motor
    return Performance(
        total_mass_kg=mass,
        hover_power_ideal_w=p_ideal,
        hover_power_w=p_hover,
        usable_energy_wh=energy_wh,
        hover_time_min=60 * energy_wh / p_hover,
        hover_thrust_fraction=mass / max_thrust_kg,
        thrust_to_weight=max_thrust_kg / mass,
    )


def check(drone: Drone, limits: HardwareLimits) -> Report:
    """Every limit passes only when its value is finite and on the good side (fail closed)."""
    perf = compute(drone)
    failures: list[Failure] = []
    bad = [name for name, value in vars(perf).items() if not math.isfinite(value)]
    if bad:
        failures.append(
            Failure(
                "finite_values",
                math.nan,
                math.nan,
                f"Beräkningen gav ogiltiga värden ({', '.join(bad)}); kontrollera drone.yaml.",
            )
        )
        return Report(perf, tuple(failures))
    if not perf.total_mass_kg <= limits.max_takeoff_mass_kg:
        failures.append(
            Failure(
                "max_takeoff_mass_kg",
                perf.total_mass_kg,
                limits.max_takeoff_mass_kg,
                f"Total massa {perf.total_mass_kg:.3f} kg är över gränsen "
                f"{limits.max_takeoff_mass_kg} kg.",
            )
        )
    if not perf.hover_time_min >= limits.min_hover_time_min:
        failures.append(
            Failure(
                "min_hover_time_min",
                perf.hover_time_min,
                limits.min_hover_time_min,
                f"Svävtid {perf.hover_time_min:.1f} min är under gränsen "
                f"{limits.min_hover_time_min} min.",
            )
        )
    if not perf.hover_thrust_fraction <= limits.max_hover_thrust_fraction:
        failures.append(
            Failure(
                "max_hover_thrust_fraction",
                perf.hover_thrust_fraction,
                limits.max_hover_thrust_fraction,
                f"Svävgas {perf.hover_thrust_fraction:.0%} av maxdragkraften är över gränsen "
                f"{limits.max_hover_thrust_fraction:.0%}.",
            )
        )
    if not perf.thrust_to_weight >= limits.min_thrust_to_weight:
        failures.append(
            Failure(
                "min_thrust_to_weight",
                perf.thrust_to_weight,
                limits.min_thrust_to_weight,
                f"Dragkraft/vikt {perf.thrust_to_weight:.2f} är under gränsen "
                f"{limits.min_thrust_to_weight}.",
            )
        )
    return Report(perf, tuple(failures))
