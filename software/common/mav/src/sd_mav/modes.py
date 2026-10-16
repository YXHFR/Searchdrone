"""ArduCopter flight modes and the mode steps of the STOP and FLY HOME path.

The mode numbers are ArduCopter's ``custom_mode`` values. ``STOP_STEPS`` and
``HOME_STEPS`` are tried in order: when the flight controller refuses a step, the next
one is sent.
"""

from __future__ import annotations

from enum import IntEnum, StrEnum
from typing import Final

__all__ = [
    "AIRBORNE_ALT_M",
    "GROUND_DISARM_MAX_ALT_M",
    "HOME_MODES",
    "HOME_STEPS",
    "ONBOARD_MODES",
    "STOP_MODES",
    "STOP_STEPS",
    "CopterMode",
    "OverridePhase",
    "mode_name",
    "override_phase",
]


class CopterMode(IntEnum):
    STABILIZE = 0
    ACRO = 1
    ALT_HOLD = 2
    AUTO = 3
    GUIDED = 4
    LOITER = 5
    RTL = 6
    CIRCLE = 7
    LAND = 9
    DRIFT = 11
    SPORT = 13
    FLIP = 14
    AUTOTUNE = 15
    POSHOLD = 16
    BRAKE = 17
    THROW = 18
    AVOID_ADSB = 19
    GUIDED_NOGPS = 20
    SMART_RTL = 21
    FLOWHOLD = 22
    FOLLOW = 23
    ZIGZAG = 24
    SYSTEMID = 25
    AUTOROTATE = 26
    AUTO_RTL = 27
    TURTLE = 28


STOP_STEPS: Final = (CopterMode.BRAKE, CopterMode.LOITER, CopterMode.LAND)
HOME_STEPS: Final = (CopterMode.SMART_RTL, CopterMode.RTL, CopterMode.LAND)
STOP_MODES: Final = frozenset(STOP_STEPS)
HOME_MODES: Final = frozenset(HOME_STEPS)

# Every mode the onboard computer may ask for. BRAKE and LAND only on the emergency
# fallback path (sd_mav.allowlist checks the origin).
ONBOARD_MODES: Final = frozenset(
    {
        CopterMode.GUIDED,
        CopterMode.AUTO,
        CopterMode.LOITER,
        CopterMode.BRAKE,
        CopterMode.SMART_RTL,
        CopterMode.RTL,
        CopterMode.LAND,
    }
)

# Below this height an armed vehicle that reports "on ground" can be disarmed.
AIRBORNE_ALT_M: Final = 0.3
# The ground station's disarm is refused above this height (a second check in the
# allowlist, after override_phase).
GROUND_DISARM_MAX_ALT_M: Final = 0.5


def mode_name(custom_mode: int) -> str:
    """The ArduCopter name of a mode number, ``"MODE_<n>"`` for unknown numbers."""
    try:
        return CopterMode(custom_mode).name
    except ValueError:
        return f"MODE_{custom_mode}"


class OverridePhase(StrEnum):
    ON_GROUND_DISARMED = "on_ground_disarmed"
    ON_GROUND_ARMED = "on_ground_armed"
    AIRBORNE = "airborne"


def override_phase(armed: bool, landed_state: str, alt_rel_home_m: float) -> OverridePhase:
    """Which steps STOP and FLY HOME use. Only the emergency path uses this.

    Anything that is not clearly "armed and standing on the ground" counts as airborne,
    including the ``takeoff``, ``landing`` and ``unknown`` landed states.
    """
    if not armed:
        return OverridePhase.ON_GROUND_DISARMED
    if landed_state == "on_ground" and alt_rel_home_m < AIRBORNE_ALT_M:
        return OverridePhase.ON_GROUND_ARMED
    return OverridePhase.AIRBORNE
