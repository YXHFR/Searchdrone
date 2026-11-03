"""The downward laser (rangefinder, "laseravståndsmätare"): its numbers and its state.

The laser is optional. It hangs under the drone, is wired to the flight controller
(ArduPilot's own driver) and measures the distance straight down to the first thing it
hits: a treetop or the ground. The onboard computer reads it as MAVLink
``DISTANCE_SENSOR`` and decides whether to trust it (``sd_onboard.rangefinder``); the
flight controller never steers by it (``WP_RFND_USE = 0``, ``SURFTRAK_MODE = 0``).

A reading is valid when the flight controller reports it inside both its own range and
the range tested for this laser, the drone tilts less than ``LASER_TILT_MAX_DEG`` (the
distance is then corrected for the tilt), and the drone's height and attitude are fresh.
The laser is **confirmed** after ``LASER_CONFIRM_S`` of valid readings in a row with no
gap longer than ``LASER_GAP_S``. The surface it sees is then the **highest** surface of
the last ``LASER_CONFIRM_S``, never an average: the nearest treetop counts.

``LaserState`` is what the height floor (``sd_safety.floor``) and the closer look read.
No laser at all (``[rangefinder] model = "none"``) is ``None`` there, not a state.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Final, Literal

__all__ = [
    "LASER_CONFIRM_S",
    "LASER_GAP_S",
    "LASER_STREAM_HZ",
    "LASER_STUCK_ALT_M",
    "LASER_TILT_MAX_DEG",
    "RANGEFINDER_ORIENT_DOWN",
    "RANGEFINDER_TYPES",
    "LaserState",
    "RangefinderModel",
]

LASER_CONFIRM_S: Final = 2.0  # valid readings in a row before the laser is trusted
LASER_TILT_MAX_DEG: Final = 20.0  # a reading taken at this tilt or more is not used
# A gap this long between valid readings (or since the last one) breaks the run. Five
# readings at the 10 Hz stream: one or two lost packets do not, a silent laser does.
LASER_GAP_S: Final = 0.5
LASER_STREAM_HZ: Final = 10  # DISTANCE_SENSOR, asked for with SET_MESSAGE_INTERVAL
# A distance that stays exactly the same (to the centimetre the message carries) while
# the drone moved this much up or down is a stuck laser, not a measurement.
LASER_STUCK_ALT_M: Final = 1.0

# ArduPilot's MAV_SENSOR_ROTATION_PITCH_270: pointing straight down (RNGFND1_ORIENT).
RANGEFINDER_ORIENT_DOWN: Final = 25

# Which laser is fitted (``[rangefinder] model`` in config/onboard.toml).
RangefinderModel = Literal["none", "tf02_pro", "grf250", "other"]

# ArduPilot 4.7's RNGFND1_TYPE for each model: Benewake serial (27), LightWare GRF serial
# (45). "other" is any driver the flight controller has (the simulator's is MAVLink, 10).
RANGEFINDER_TYPES: Final[dict[RangefinderModel, int]] = {"tf02_pro": 27, "grf250": 45}


@dataclass(frozen=True, slots=True)
class LaserState:
    """What a fitted laser says right now.

    ``surface_amsl_m`` is set exactly when ``confirmed``: the highest surface (drone's
    height above sea level minus the tilt-corrected distance) over the last
    ``LASER_CONFIRM_S``. ``min_range_m`` is the laser's tested shortest range
    (``[rangefinder] min_tested_m``): the floor keeps that plus a margin above the
    surface. ``distance_m`` is the newest valid tilt-corrected distance, for the panel
    and the logs only; it is ``None`` when the newest reading was not valid.
    """

    confirmed: bool
    surface_amsl_m: float | None
    min_range_m: float
    distance_m: float | None = None

    def __post_init__(self) -> None:
        if self.confirmed != (self.surface_amsl_m is not None):
            raise ValueError("a surface is given exactly when the laser is confirmed")
        values = [self.min_range_m]
        values += [v for v in (self.surface_amsl_m, self.distance_m) if v is not None]
        if not all(math.isfinite(v) for v in values):
            raise ValueError(f"laser values must be finite: {values}")
        if self.min_range_m < 0 or (self.distance_m is not None and self.distance_m < 0):
            raise ValueError("a range or distance cannot be negative")
