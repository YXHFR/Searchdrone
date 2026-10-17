"""Values the scenario tests compare with, read from their source at run time.

No scenario writes a mode number or a failsafe time itself: the modes come from
``sd_mav.modes``, the times and actions from the safety parameter file. A value missing
from its source is an ``InfraError``, never a guessed default.
"""

from __future__ import annotations

from functools import cache
from types import MappingProxyType
from typing import Final

from sd_mav.failsafe_signals import PATTERNS
from sd_mav.modes import HOME_STEPS, STOP_STEPS, CopterMode
from sd_sim import params, paths
from sd_sim.errors import InfraError

__all__ = [
    "FS_CHAIN",
    "HOME_MODE",
    "SKILL_RTL_MODE",
    "SRTL_UNAVAILABLE",
    "STOP_MODE",
    "fs_chain",
    "safety_params",
    "value",
]

STOP_MODE: Final = STOP_STEPS[0]  # BRAKE, then LOITER, then LAND
HOME_MODE: Final = HOME_STEPS[0]  # SMART_RTL, then RTL, then LAND
SKILL_RTL_MODE: Final = CopterMode.SMART_RTL  # the return_home skill (fallback RTL)
SRTL_UNAVAILABLE: Final = PATTERNS["srtl_unavailable"]

# ArduCopter 4.7.1's failsafe actions (FS_GCS_ENABLE, BATT_FS_LOW_ACT, FENCE_ACTION) as
# the modes they may end in, first choice first.
FS_CHAIN: Final = MappingProxyType(
    {
        1: (CopterMode.RTL, CopterMode.LAND),
        2: (CopterMode.LAND,),
        3: (CopterMode.SMART_RTL, CopterMode.RTL, CopterMode.LAND),
        4: (CopterMode.SMART_RTL, CopterMode.LAND),
        5: (CopterMode.BRAKE, CopterMode.LAND),
    }
)


@cache
def safety_params() -> dict[str, float]:
    path = paths.REPO / "simulation" / "sitl" / "params" / "searchdrone-safety.parm"
    if not path.is_file():
        raise InfraError(f"{path.name} saknas")
    return params.parse_parm(path.read_text(encoding="utf-8"), path.name)


def value(name: str) -> float:
    found = safety_params().get(name)
    if found is None:
        raise InfraError(f"säkerhetsparameterfilen saknar värdet {name}")
    return found


def fs_chain(param: str) -> tuple[CopterMode, ...]:
    """The modes the failsafe ``param`` (e.g. ``FS_GCS_ENABLE``) may end in."""
    action = int(value(param))
    chain = FS_CHAIN.get(action)
    if chain is None:
        raise InfraError(f"{param} = {action} har ingen känd kedja")
    return chain


# --- the limits table: the only source of numbers for the scenario tests ---
# host = the host's monotonic clock, sim = the simulator's own time (real time here).
L_STOP_ACK_S: Final = 1.0  # STOP/HOME call -> COMMAND_ACK(DO_SET_MODE) (host)
L_STOP_HB_S: Final = 2.0  # STOP/HOME call -> a HEARTBEAT shows the mode (host)
L_STOP_STILL_S: Final = 5.0  # STOP call -> ground speed < 0.5 m/s (sim)
L_STOP_HOLD_S: Final = 20.0  # then, for this long: within 3 m and 1 m (sim)
L_HOLD_H_M: Final = 3.0
L_HOLD_V_M: Final = 1.0
STILL_MS: Final = 0.5
L_GATE_S: Final = 2.0  # order sent -> gate decision (host)
L_DRIFT_S: Final = 15.0  # after a refused order, for this long: within 3 m and 1 m (sim)
L_CHAT_S: Final = 10.0  # operator line or gate refusal -> the commander answers (host)
L_TAKEOFF_S: Final = 30.0  # takeoff accepted -> 40 +- 2 m above home (sim)
L_ARRIVE_S: Final = 60.0  # goto accepted -> within 5 m / 2 m and still for 2 s (sim)
L_LAND_S: Final = 180.0  # flying home started -> landed and disarmed (sim)
L_LAND_M: Final = 3.0  # ... this close to home
L_SKILL_RTL_S: Final = 3.0  # an onboard rule fired -> SKILL_RTL_MODE in the HEARTBEAT (sim)
L_FS_S: Final = 2.0  # a failsafe condition met -> the failsafe mode (sim)
L_BRAIN_DOWN_S: Final = 6.0  # sd-brain dead -> brain down and a system line (host)
L_AI_SILENT_S: Final = 150.0  # AI dead after done -> holds this long, never home (sim)
L_WATCHDOG_STOP_S: Final = 4.0  # sd-onboard dead in GUIDED -> STOP_MODE (sim)
L_WATCHDOG_RTL_S: Final = 21.0  # ... -> HOME_MODE (sim)
L_LINK_DEGRADED_S: Final = 3.0  # SD-link broken -> degraded (host, +- 1 s)
L_LINK_LOST_S: Final = 10.0  # ... -> lost (host, +- 1 s)
L_ABSENT_S: Final = 32.0  # last pilot WS closed -> OPERATOR_ABSENT (30 s + 2 s, host)
