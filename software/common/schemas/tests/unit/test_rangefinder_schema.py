"""The downward laser's state: a surface exactly when it is confirmed, finite values."""

from __future__ import annotations

import math

import pytest

from sd_schemas.rangefinder import RANGEFINDER_TYPES, LaserState


def test_a_confirmed_laser_carries_its_surface() -> None:
    state = LaserState(confirmed=True, surface_amsl_m=31.5, min_range_m=10.0, distance_m=12.0)
    assert state.surface_amsl_m == 31.5
    assert LaserState(confirmed=False, surface_amsl_m=None, min_range_m=0.0).distance_m is None


@pytest.mark.parametrize(
    "kwargs",
    [
        {"confirmed": True, "surface_amsl_m": None, "min_range_m": 0.0},
        {"confirmed": False, "surface_amsl_m": 20.0, "min_range_m": 0.0},
        {"confirmed": True, "surface_amsl_m": math.nan, "min_range_m": 0.0},
        {"confirmed": False, "surface_amsl_m": None, "min_range_m": math.inf},
        {"confirmed": False, "surface_amsl_m": None, "min_range_m": -1.0},
        {"confirmed": False, "surface_amsl_m": None, "min_range_m": 0.0, "distance_m": -0.5},
    ],
)
def test_an_impossible_state_is_refused(kwargs: dict[str, object]) -> None:
    with pytest.raises(ValueError):
        LaserState(**kwargs)  # type: ignore[arg-type]  # pyright: ignore[reportArgumentType]


def test_the_driver_numbers_are_arducopter_47s() -> None:
    # RNGFND1_TYPE: Benewake serial 27, LightWare GRF serial 45 (research 31).
    assert RANGEFINDER_TYPES == {"tf02_pro": 27, "grf250": 45}
