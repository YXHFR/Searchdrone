"""Flight modes, the emergency-path phase and the failsafe signals."""

from __future__ import annotations

from typing import Any

import pytest
from pymavlink.dialects.v20 import ardupilotmega as mavlink

from sd_mav.failsafe_signals import PATTERNS, notice_text, signal_from, text_signal
from sd_mav.modes import (
    HOME_MODES,
    HOME_STEPS,
    STOP_MODES,
    STOP_STEPS,
    CopterMode,
    OverridePhase,
    mode_name,
    override_phase,
)


def test_mode_numbers_and_steps() -> None:
    expected = {
        "STABILIZE": 0,
        "ALT_HOLD": 2,
        "AUTO": 3,
        "GUIDED": 4,
        "LOITER": 5,
        "RTL": 6,
        "LAND": 9,
        "BRAKE": 17,
        "SMART_RTL": 21,
    }
    for name, number in expected.items():
        assert CopterMode[name] == number
    assert STOP_STEPS == (CopterMode.BRAKE, CopterMode.LOITER, CopterMode.LAND)
    assert HOME_STEPS == (CopterMode.SMART_RTL, CopterMode.RTL, CopterMode.LAND)
    assert set(STOP_STEPS) == STOP_MODES and set(HOME_STEPS) == HOME_MODES
    assert mode_name(5) == "LOITER" and mode_name(99) == "MODE_99"


@pytest.mark.parametrize("landed", ["on_ground", "in_air", "takeoff", "landing", "unknown"])
@pytest.mark.parametrize("alt", [0.0, 0.29, 0.31, 3.0])
@pytest.mark.parametrize("armed", [False, True])
def test_override_phase(armed: bool, landed: str, alt: float) -> None:
    phase = override_phase(armed, landed, alt)
    if not armed:
        assert phase is OverridePhase.ON_GROUND_DISARMED
    elif landed == "on_ground" and alt < 0.3:
        assert phase is OverridePhase.ON_GROUND_ARMED
    else:
        assert phase is OverridePhase.AIRBORNE


def _text(text: str) -> Any:
    return mavlink.MAVLink_statustext_message(mavlink.MAV_SEVERITY_WARNING, text.encode())


@pytest.mark.parametrize(
    ("text", "name"),
    [
        ("Failsafe: battery", "failsafe"),
        ("Fence Breach", "fence_breach"),
        ("Battery 1 is critical 13.2V", "battery"),
        ("GCS Failsafe On", "failsafe"),
        ("gcs lost", "gcs"),
        ("EKF variance", "ekf"),
        ("Crash: Disarming", "crash"),
        ("SmartRTL Unavailable, Trying RTL Mode", "srtl_unavailable"),
        # Copter-4.7.1 events.cpp failsafe_deadreckon_check()
        ("Dead Reckoning started", "dead_reckoning"),
        ("Dead Reckoning timeout", "dead_reckoning"),
        # sd_gcs_loiter_cap.lua
        ("SD failsafe: no GCS 150 s in LOITER, SmartRTL", "failsafe"),
    ],
)
def test_failsafe_texts(text: str, name: str) -> None:
    assert text_signal(text) == name
    signal = signal_from(_text(text))
    assert signal is not None and signal.source == "statustext" and signal.text == text


def test_not_failsafe() -> None:
    for text in ("Mode changed", "Arming motors", "Reached waypoint #3", "EKF3 IMU0 is using GPS"):
        assert signal_from(_text(text)) is None
    assert set(PATTERNS) >= {
        "failsafe",
        "fence_breach",
        "battery",
        "gcs",
        "ekf",
        "crash",
        "srtl_unavailable",
    }


def test_battery_and_heartbeat_signals() -> None:
    def battery(state: int) -> Any:
        return mavlink.MAVLink_battery_status_message(
            0, 0, 0, 3000, [16000] + [65535] * 9, -1, -1, -1, 20, 0, state
        )

    for state in (2, 3, 4):
        sig = signal_from(battery(state))
        assert sig is not None and sig.source == "battery"
    for state in (0, 1, 5, 6, 7):
        assert signal_from(battery(state)) is None
    hb = mavlink.MAVLink_heartbeat_message(2, 3, 0, 5, mavlink.MAV_STATE_CRITICAL, 3)
    assert signal_from(hb) is not None
    hb_ok = mavlink.MAVLink_heartbeat_message(2, 3, 0, 5, mavlink.MAV_STATE_ACTIVE, 3)
    assert signal_from(hb_ok) is None
    assert signal_from(mavlink.MAVLink_vfr_hud_message(0, 0, 0, 0, 0, 0)) is None


@pytest.mark.parametrize(
    ("text", "name"),
    [
        # Copter-4.7.1 events.cpp failsafe_gcs_on_event() with FS_OPTIONS bit 4 in LOITER
        ("GCS Failsafe - Continuing Pilot Control", "gcs_continue"),
        # Copter-4.7.1 events.cpp gpsglitch_check()
        ("GPS Glitch or Compass error", "gps_glitch"),
    ],
)
def test_failsafe_text_patterns_notices(text: str, name: str) -> None:
    """The flight controller keeps its mode: a notice, never a failsafe signal."""
    assert notice_text(text) == name
    assert text_signal(text) is None
    assert signal_from(_text(text)) is None
