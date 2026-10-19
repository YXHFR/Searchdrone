"""Every time and margin the safety logic uses, in one place.

Never read from configuration or environment variables. The limits themselves (how
high, how fast, how far) are in :mod:`sd_safety.limits`, not here.
"""

from __future__ import annotations

from types import MappingProxyType
from typing import Final

# --- the gate ---------------------------------------------------------------------------

CLEARANCE_VALID_S: Final = 120  # "clear around the drone" is valid this long, then expires
APPROVAL_TIMEOUT_S: Final = 60  # an approval card that nobody answers expires
TAKEOFF_BATTERY_MARGIN_PCT: Final = 20.0  # takeoff needs battery >= rtl level + this
BRAIN_BATTERY_MARGIN_PCT: Final = 5.0  # the AI may not start a move this close to the rtl level
# A drawn path (follow_path) must leave the reserve after the path and the flight home.
# Assumed use: 5 % per minute in the air, i.e. 20 min on a full pack, well below the
# simulation model's computed hover time. A real energy model replaces this later.
ROUTE_BATTERY_PCT_PER_MIN: Final = 5.0
ROUTE_DESCENT_MS: Final = 1.0  # the landing at home, counted slowly
AIRBORNE_MIN_ALT_M: Final = 2.0  # "airborne" for the gate: armed, in the air and above this
STATE_MAX_AGE_MS: Final = 2000  # older vehicle state or snapshot -> STATE_STALE
RATE_ACTION_MIN_GAP_MS: Final = 3000  # at most one AI action per 3 s ...
RATE_WINDOW_MS: Final = 60_000  # ... and per minute at most:
RATE_ACTIONS_PER_WINDOW: Final = 15
RATE_REPORTS_PER_WINDOW: Final = 6

# --- onboard executor --------------------------------------------------------------------

FC_HEARTBEAT_TIMEOUT_S: Final = 3.0
MAV_ACK_TIMEOUT_S: Final = 1.0
MODE_CONFIRM_TIMEOUT_S: Final = 2.0
MISSION_UPLOAD_TIMEOUT_S: Final = 10.0
MISSION_MAX_ITEMS: Final = 400
MISSION_END_LOITER_S: Final = 30
EXECUTOR_DEFAULT_SPEED_MS: Final = 5.0  # mirrors WP_SPD; used when an order gives no speed
GOTO_ARRIVE_RADIUS_M: Final = 2.0
GOTO_ALT_STEP_TOL_M: Final = 1.5
DEFAULT_SURVEY_SPACING_M: Final = 30.0
SURVEY_MAX_AREA_M2: Final = 250_000.0
HOME_MATCH_M: Final = 10.0
ALT_OVERSHOOT_M: Final = 2.0
SPEED_OVERSHOOT_MS: Final = 2.0
RANGE_HARD_MARGIN_M: Final = 50.0
PILOT_POS_STALE_S: Final = 5.0
PILOT_POS_MAX_JUMP_MS: Final = 50.0
BATTERY_UNKNOWN_S: Final = 5.0
BATTERY_UNKNOWN_RTL_S: Final = 30.0
LIMIT_CLEAR_S: Final = 2.0
HEARTBEAT_INTERVAL_US: Final = 200_000
COMMANDER_STATUS_HZ: Final = 1
EXECUTOR_TICK_HZ: Final = 10
SNAPSHOT_HZ: Final = 2
# Real flight: FS_THR_ENABLE on the flight controller (RC link lost in the air -> SmartRTL,
# else RTL). The real-flight parameter layer sets it; the onboard config check requires it.
FIELD_RC_FAILSAFE: Final = 4.0

# --- links, watchdogs and the emergency path --------------------------------------------

LINK_LOSS_DETECT_S: Final = 3.0
SDLINK_GRACE_S: Final = 20.0
SDLINK_HOLD_S: Final = 30.0
FS_GCS_TIMEOUT_S: Final = 20.0  # mirrors FS_GCS_TIMEOUT in the flight controller
BRAIN_HEARTBEAT_TIMEOUT_S: Final = 5.0
OPERATOR_ABSENT_S: Final = 30.0
COMPANION_WATCHDOG_S: Final = 3.0
COMPANION_WATCHDOG_RTL_S: Final = 20.0
OVERRIDE_NOTICE_WINDOW_S: Final = 2.0
OVERRIDE_ONBOARD_FALLBACK_S: Final = 1.5
OVERRIDE_RESEND_MS: Final = 200
OVERRIDE_MAX_S: Final = 10.0
GCS_SEEN_FOR_RELEASE_S: Final = 3.0

# --- the flight controller's fence --------------------------------------------------------

FENCE_POLYGON_MARGIN_M: Final = 10.0  # FC polygon = area + geofence_buffer_m + this
FENCE_ALT_MARGIN_M: Final = 5.0  # FENCE_ALT_MAX = max height + this
FENCE_RADIUS_MARGIN_M: Final = 30.0  # FENCE_RADIUS = max range + this
FENCE_MARGIN_M: Final = 5.0  # the flight controller's own FENCE_MARGIN
FC_FENCE_AREA_MARGIN_M: Final = 10.0  # every corner of an area stays this far inside the margin

FC_SAFETY_EXPECTED: Final = MappingProxyType(
    {
        # The flight controller's safety parameters with the standard limits. The ground
        # station only reads them; when a pilot raises the limits it sets the fence to
        # sd_safety.limits.fc_fence_params(limits) instead.
        "FENCE_ENABLE": 1,
        "FENCE_TYPE": 7,  # height + circle + polygon
        "FENCE_ACTION": 3,  # SmartRTL, else RTL, else LAND
        "FENCE_ALT_MAX": 125,
        "FENCE_RADIUS": 530,
        "FENCE_MARGIN": 5,
        "LOIT_SPEED_MS": 15,
        "FS_GCS_ENABLE": 3,
        "FS_GCS_TIMEOUT": 20,
        "MAV_GCS_SYSID": 250,
        "BATT_FS_LOW_ACT": 3,
        "BATT_FS_CRT_ACT": 1,
        # Read by the ground station's wind warning (sd_ground.services.wind).
        "BATT_CAPACITY": 10000,  # mAh, the 6S2P Li-ion pack
        "BATT_LOW_MAH": 1500,  # the flight controller's low-battery reserve
        "RTL_SPEED_MS": 8,  # ground speed on the way home
    }
)
# ATC_ANGLE_MAX: the safety file leaves ArduPilot's default (30 degrees in Copter-4.7.1).
FC_ANGLE_MAX_DEG: Final = 30.0

# What the flight controller can represent (ArduPilot Copter 4.7 parameter ranges).
FC_FENCE_ALT_MAX_RANGE_M: Final = (10.0, 1000.0)
FC_FENCE_RADIUS_RANGE_M: Final = (30.0, 10_000.0)
FC_LOIT_SPEED_RANGE_MS: Final = (0.2, 35.0)
