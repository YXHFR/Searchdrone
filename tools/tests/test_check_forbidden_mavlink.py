"""tools/check_forbidden_mavlink.py: allowed and refused examples, one per rule."""

from __future__ import annotations

from pathlib import Path

import check_forbidden_mavlink as cfm
import pytest

ROOT = Path(__file__).resolve().parents[2]
GROUND = "software/ground/src/sd_ground/mav/example.py"
SIM = "simulation/sd_sim/src/sd_sim/example.py"


def rules(text: str, path: str = GROUND) -> list[str]:
    return [f.rule for f in cfm.check_source(text, path)]


@pytest.mark.parametrize(
    "line",
    [
        "msg = mavlink.MAVLink_param_set_message(1, 1, b'FENCE_ENABLE', 0, 2)",
        "conn.param_ext_set_send(1, 1, b'FENCE_ENABLE', b'0', 2)",
        "conn.rc_channels_override_send(1, 1, *([1500] * 8))",
        "msg = mavlink.MAVLink_set_mode_message(1, 1, 4)",
        "kind = 'PARAM_SET'",
        "kind = b'RC_CHANNELS_OVERRIDE'",
    ],
)
def test_denied_messages_fail(line: str) -> None:
    assert "forbidden-message" in rules(line)


@pytest.mark.parametrize(
    "line",
    [
        "cmd = mavlink.MAV_CMD_DO_FENCE_ENABLE",
        "cmd = MAV_CMD_DO_FLIGHTTERMINATION",
        "cmd = mavlink.MAV_CMD_PREFLIGHT_REBOOT_SHUTDOWN",
        "m = mavlink.MAVLink_command_long_message(1, 1, 207, 0, 0, 0, 0, 0, 0, 0, 0)",
        "conn.command_int_send(1, 1, 6, 185, 0, 0, 0, 0, 0, 0, 0, 0, 0)",
        "m = mavlink.MAVLink_command_long_message(1, 1, command=180, confirmation=0)",
    ],
)
def test_denied_commands_fail(line: str) -> None:
    assert "forbidden-command" in rules(line)


def test_forced_arming_fails() -> None:
    line = "m = mavlink.MAVLink_command_long_message(1, 1, 400, 0, 1, 21196, 0, 0, 0, 0, 0)"
    assert rules(line) == ["forced-arming"]
    assert rules("FORCE = 21196.0") == ["forced-arming"]


def test_raw_send_in_production_fails_but_not_in_the_simulator() -> None:
    line = "self._conn.mav.heartbeat_send(6, 8, 0, 0, 0)"
    assert rules(line) == ["raw-send"]
    assert rules(line, SIM) == []


@pytest.mark.parametrize(
    "line",
    [
        # What the ground station and the onboard computer really send, via Vehicle.send.
        "m = mavlink.MAVLink_command_long_message(1, 1, 400, 0, 1, 0, 0, 0, 0, 0, 0)",
        "m = mavlink.MAVLink_command_long_message(1, 1, mavlink.MAV_CMD_DO_SET_MODE, 0, 1, 4)",
        "m = mavlink.MAVLink_mission_count_message(1, 1, 3, mavlink.MAV_MISSION_TYPE_FENCE)",
        "await vehicle.send(m, origin=SendOrigin.GROUND)",
        "name = 'FENCE_ENABLE'  # read back with PARAM_REQUEST_READ",
        "conn.param_request_read_send(1, 1, b'FENCE_ENABLE', -1)",
        "event = 'mav.forbidden_mavlink_blocked'",
        "self.mav.submit(action)",
        "limit = 21195",
    ],
)
def test_allowed_traffic_passes(line: str) -> None:
    assert rules(line) == []


def test_docstrings_are_prose() -> None:
    text = '"""PARAM_SET"""\n\ndef f() -> None:\n    """RC_CHANNELS_OVERRIDE"""\n'
    assert rules(text) == []


def test_the_gate_and_the_harness_are_allowed() -> None:
    gate = "software/common/mav/src/sd_mav/vehicle.py"
    harness = "simulation/sd_sim/src/sd_sim/harness.py"
    assert rules("m = mavlink.MAVLink_param_set_message(1, 1, b'FENCE_RADIUS', 600, 9)", gate) == []
    assert rules("self._conn.mav.param_set_send(1, 1, name.encode(), v, 9)", harness) == []
    # The harness may set SIM_* parameters, never switch off the fence.
    assert rules("c = mavlink.MAV_CMD_DO_FENCE_ENABLE", harness) == ["forbidden-command"]


def test_tests_are_not_read() -> None:
    assert cfm.is_test_file("software/common/mav/tests/unit/test_forbidden.py")
    assert cfm.is_test_file("simulation/tests/onboard/test_onboard_sitl.py")
    assert cfm.is_test_file("software/onboard/conftest.py")
    assert not cfm.is_test_file("software/onboard/src/sd_onboard/executor/executor.py")


def test_every_denied_command_has_a_name() -> None:
    assert set(cfm.DENIED_COMMAND_NAMES.values()) == set(cfm.DENIED_COMMANDS)
    assert "MAV_CMD_DO_SET_PARAMETER" in cfm.DENIED_COMMAND_NAMES


def test_findings_name_file_and_line() -> None:
    (finding,) = cfm.check_source("x = 1\ncmd = mavlink.MAV_CMD_DO_PARACHUTE\n", GROUND)
    assert str(finding).startswith(f"{GROUND}:2: forbidden-command: MAV_CMD_DO_PARACHUTE")


def test_main_fails_on_a_finding(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    rel = "software/x/src/x/bad.py"
    (tmp_path / rel).parent.mkdir(parents=True)
    (tmp_path / rel).write_text("c = MAV_CMD_DO_MOTOR_TEST\n", encoding="utf-8")
    monkeypatch.setattr(cfm, "REPO", tmp_path)
    assert cfm.main([rel]) == 1


def test_the_repository_is_clean() -> None:
    files = cfm.candidate_files(ROOT)
    assert "software/ground/src/sd_ground/mav/gcs.py" in files
    assert [str(f) for f in cfm.check_files(ROOT, files)] == []
