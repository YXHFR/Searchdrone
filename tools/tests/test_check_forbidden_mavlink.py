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
        # By message id: the constant, its name as a string, or the number in mavlink_map.
        "i = mavlink.MAVLINK_MSG_ID_PARAM_SET",
        "i = getattr(mavlink, 'MAVLINK_MSG_ID_RC_CHANNELS_OVERRIDE')",
        "cls = mavlink.mavlink_map[23]",
        "cls = mavlink.mavlink_map.get(23)",
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
    assert rules(line) == ["raw-send"] * 3  # _conn, heartbeat_send, and the send on mav
    assert rules(line, SIM) == []


@pytest.mark.parametrize(
    "line",
    [
        # Ways to own a connection, or to write on one, without Vehicle.send.
        "from pymavlink import mavutil",
        "import pymavlink.mavutil as mu",
        "from pymavlink.mavutil import mavlink_connection",
        "c = mavutil.mavlink_connection('udpout:127.0.0.1:14550')",
        "enc = mavlink.MAVLink(sock)",
        "conn.reboot_autopilot()",
        "conn.arducopter_arm()",
        "conn.set_mode_apm(6)",
        "conn.set_mode_rtl()",
        # Importing mavutil by name at run time.
        "mu = importlib.import_module('pymavlink.mavutil')",
        "mu = __import__('pymavlink', fromlist=['mavutil'])",
        # Any message class from the id table, the gate's connection or its writer.
        "cls = mavlink.mavlink_map[msg_id]",
        "vehicle._conn.close()",
        "vehicle._transport.write_message(m)",
        # mavfile's set_mode, and a helper taken without a call (called later).
        "conn.set_mode('GUIDED')",
        "conn.set_mode(4, custom_mode=4)",
        "switch = conn.set_mode",
        "boot = conn.reboot_autopilot",
        "send = conn.write_message",
        # Any message's send method, also on a mav taken into another name, and the
        # helpers spelled as getattr strings.
        "conn.param_request_read_send(1, 1, b'FENCE_ENABLE', -1)",
        "m = conn.mav\nm.heartbeat_send(6, 8, 0, 0, 0)",
        "c = getattr(vehicle._transport, '_conn')",
        "getattr(conn, 'reboot_autopilot')()",
        "getattr(conn, 'write_message')(msg)",
    ],
)
def test_owning_a_connection_in_production_fails(line: str) -> None:
    assert "raw-send" in rules(line)
    assert "raw-send" not in rules(line, SIM)


@pytest.mark.parametrize(
    ("line", "rule"),
    [
        # A denied name under another name, or spelled as a string for getattr.
        (
            "from pymavlink.dialects.v20.ardupilotmega import MAVLink_param_set_message as M",
            "forbidden-message",
        ),
        (
            "from pymavlink.dialects.v20.ardupilotmega import MAV_CMD_DO_FENCE_ENABLE as F",
            "forbidden-command",
        ),
        ("getattr(conn.mav, 'param_set_send')(1, 1, b'FENCE_ENABLE', 0, 2)", "forbidden-message"),
        ("cmd = getattr(mavlink, 'MAV_CMD_DO_FLIGHTTERMINATION')", "forbidden-command"),
    ],
)
def test_renamed_or_spelled_out_denied_names_fail(line: str, rule: str) -> None:
    assert rule in rules(line, SIM)


@pytest.mark.parametrize(
    "line",
    [
        # What the ground station and the onboard computer really send, via Vehicle.send.
        "from pymavlink.dialects.v20 import ardupilotmega as mavlink",
        "await io.set_mode(CopterMode.GUIDED)",
        "await self.set_mode(mode, SendOrigin.LIMIT)",
        "i = mavlink.MAVLINK_MSG_ID_HEARTBEAT",
        "skills = importlib.import_module('sd_onboard.skills')",
        "self._connection.close()",
        "await v.set_mode(mode=CopterMode.GUIDED, origin=SendOrigin.GROUND)",
        "pkg = __import__('sd_onboard.skills')",
        "m = mavlink.MAVLink_command_long_message(1, 1, 400, 0, 1, 0, 0, 0, 0, 0, 0)",
        "m = mavlink.MAVLink_command_long_message(1, 1, mavlink.MAV_CMD_DO_SET_MODE, 0, 1, 4)",
        "m = mavlink.MAVLink_mission_count_message(1, 1, 3, mavlink.MAV_MISSION_TYPE_FENCE)",
        "await vehicle.send(m, origin=SendOrigin.GROUND)",
        "name = 'FENCE_ENABLE'  # read back with PARAM_REQUEST_READ",
        "event = 'mav.forbidden_mavlink_blocked'",
        "MavAction(kind='set_mode', params={'mode': 'GUIDED'})",
        "await self.mav.submit(action)",
        "self.mav.submit(action)",
        "limit = 21195",
    ],
)
def test_allowed_traffic_passes(line: str) -> None:
    assert rules(line) == []


def test_docstrings_are_prose() -> None:
    text = '"""PARAM_SET"""\n\ndef f() -> None:\n    """RC_CHANNELS_OVERRIDE"""\n'
    assert rules(text) == []


@pytest.mark.parametrize(
    "line",
    [
        # Outside production the simulator and tools may own a connection.
        "self._conn.write_message(m)",
        "send = conn.write_message",
        "harness.set_mode('GUIDED')",
        "conn.param_request_read_send(1, 1, b'FENCE_ENABLE', -1)",
    ],
)
def test_connection_helpers_pass_in_the_simulator(line: str) -> None:
    assert set(rules(line)) == {"raw-send"}
    assert rules(line, SIM) == []


@pytest.mark.parametrize(
    "line",
    [
        # Helpers that send denied traffic themselves: refused everywhere outside the gate.
        "self._conn.reboot_autopilot()",
        "self._conn.set_servo(9, 1900)",
        "self._conn.calibrate_level()",
        "self._conn.set_mode_rtl()",
        "boot = conn.reboot_autopilot",
    ],
)
def test_denied_helpers_fail_in_the_simulator(line: str) -> None:
    assert rules(line, SIM) == ["forbidden-command"]
    assert rules(line, HARNESS) == ["forbidden-command"]


def test_mavlink_map_outside_production_only_for_allowed_ids() -> None:
    assert rules("cls = mavlink.mavlink_map[0]", SIM) == []
    assert rules("cls = mavlink.mavlink_map.get(0)", SIM) == []
    assert rules("cls = mavlink.mavlink_map[23]", SIM) == ["forbidden-message"]


HARNESS = "simulation/sd_sim/src/sd_sim/harness.py"


def test_the_gate_is_allowed() -> None:
    gate = "software/common/mav/src/sd_mav/vehicle.py"
    assert rules("m = mavlink.MAVLink_param_set_message(1, 1, b'FENCE_RADIUS', 600, 9)", gate) == []
    assert rules("self._conn.mav.param_set_send(1, 1, name.encode(), v, 9)", gate) == []


@pytest.mark.parametrize(
    "line",
    [
        # set_sim_param refuses any name but SIM_* at run time.
        "def set_sim_param(self, name, v):\n"
        "    'Docs.'\n"
        "    if not name.startswith('SIM_'):\n"
        "        raise ValueError(name)\n"
        "    self._conn.mav.param_set_send(1, 1, name.encode(), v, 9)",
        "self._conn.mav.param_set_send(1, 1, b'SIM_RC_FAIL', 1, 9)",
        "self._conn.mav.param_set_send(1, 1, 'SIM_WIND_DIR'.encode(), 90, 9)",
        "self._conn.mav.param_set_send(1, 1, param_id=b'SIM_WIND_SPD', param_value=3)",
        "self._conn.mav.heartbeat_send(6, 8, 0, 0, 0)",
        # In the writer: the guarded name, or a literal SIM_ one.
        "def set_sim_param(self, name, v):\n"
        "    if not name.startswith('SIM_'):\n"
        "        raise ValueError(name)\n"
        "    self._conn.mav.param_set_send(1, 1, b'SIM_SPEEDUP', v, 9)",
        # mavfile's own param_set_send(parm_name, parm_value).
        "self._conn.param_set_send('SIM_WIND_DIR', 90)",
        "self._conn.param_set_send(parm_name=b'SIM_WIND_SPD', parm_value=3)",
        # The guarded name, sent as is.
        "def set_sim_param(self, name, v):\n"
        "    if not name.startswith('SIM_'):\n"
        "        raise ValueError(name)\n"
        "    self._conn.mav.param_set_send(1, 1, name.encode(), v, 9)",
    ],
)
def test_the_harness_may_set_sim_parameters(line: str) -> None:
    assert rules(line, HARNESS) == []


@pytest.mark.parametrize(
    "line",
    [
        # PARAM_SET of a literal non-SIM_ name, and every other denied message or command.
        "self._conn.mav.param_set_send(1, 1, b'FENCE_ENABLE', 0, 9)",
        "m = mavlink.MAVLink_param_set_message(1, 1, 'ARMING_SKIPCHK', 1, 9)",
        "self._conn.mav.param_set_send(1, 1, param_id=b'FENCE_ENABLE', param_value=0)",
        "self._conn.mav.param_set_send(1, 1, 'FENCE_ENABLE'.encode(), 0, 9)",
        "self._conn.mav.param_set_send(1, 1, name.encode(), v, 9)",
        "self._conn.param_set_send('FS_THR_ENABLE', 0)",
        # mavfile's param_set_send takes the name first, not third.
        "self._conn.param_set_send('FENCE_ENABLE', 0, b'SIM_RC_FAIL')",
        # The guarded writer sends only the guarded name or a SIM_ literal.
        "def set_sim_param(self, name, v):\n"
        "    if not name.startswith('SIM_'):\n"
        "        raise ValueError(name)\n"
        "    self._conn.mav.param_set_send(1, 1, b'FENCE_ENABLE', 0, 9)",
        "def set_sim_param(self, name, v):\n"
        "    if not name.startswith('SIM_'):\n"
        "        raise ValueError(name)\n"
        "    self._conn.mav.param_set_send(1, 1, other.encode(), v, 9)",
        # A mav taken into another name, or an owner of unknown kind: the third
        # argument is the dialect's param_id, whatever the first says.
        "m = self._conn.mav\nm.param_set_send(b'SIM_X', 1, b'FENCE_ENABLE', 0, 9)",
        "getattr(self._conn, 'mav').param_set_send(b'SIM_X', 1, b'ARMING_CHECK', 0, 9)",
        # The guarded name bound again after the guard is not trusted.
        "def set_sim_param(self, name, v):\n"
        "    if not name.startswith('SIM_'):\n"
        "        raise ValueError(name)\n"
        "    name = 'FENCE_ENABLE'\n"
        "    self._conn.mav.param_set_send(1, 1, name.encode(), v, 9)",
        "def set_sim_param(self, name, v):\n"
        "    if not name.startswith('SIM_'):\n"
        "        raise ValueError(name)\n"
        "    for name in ['FENCE_ENABLE']:\n"
        "        self._conn.mav.param_set_send(1, 1, name.encode(), v, 9)",
        # Bound again by import, match or with.
        "def set_sim_param(self, name, v):\n"
        "    if not name.startswith('SIM_'):\n"
        "        raise ValueError(name)\n"
        "    from consts import FENCE as name\n"
        "    self._conn.mav.param_set_send(1, 1, name.encode(), v, 9)",
        "def set_sim_param(self, name, v):\n"
        "    if not name.startswith('SIM_'):\n"
        "        raise ValueError(name)\n"
        "    match b'FENCE_ENABLE':\n"
        "        case name:\n"
        "            self._conn.mav.param_set_send(1, 1, name, v, 9)",
        "def set_sim_param(self, name, v):\n"
        "    if not name.startswith('SIM_'):\n"
        "        raise ValueError(name)\n"
        "    with open(p) as name:\n"
        "        self._conn.mav.param_set_send(1, 1, name, v, 9)",
        # A default, decorator or annotation runs outside the guard.
        "name = b'FENCE_ENABLE'\n"
        "def set_sim_param(self, name, v, _s=c.mav.param_set_send(1, 1, name, 0, 9)):\n"
        "    if not name.startswith('SIM_'):\n"
        "        raise ValueError(name)",
        "name = b'FENCE_ENABLE'\n"
        "@deco(c.mav.param_set_send(1, 1, name, 0, 9))\n"
        "def set_sim_param(self, name, v):\n"
        "    if not name.startswith('SIM_'):\n"
        "        raise ValueError(name)",
        # A bare param_set_send (a bound method passed in) is of unknown kind.
        "def f(param_set_send):\n    param_set_send('FS_THR_ENABLE', 0, b'SIM_X')",
        # A *args or **kwargs may fill the name's place.
        "self._conn.mav.param_set_send(1, *[1, b'ARMING_CHECK'], b'SIM_X', 0, 9)",
        "def set_sim_param(self, name, v):\n"
        "    if not name.startswith('SIM_'):\n"
        "        raise ValueError(name)\n"
        "    self._conn.mav.param_set_send(1, *[1, b'FENCE_ENABLE'], name, v, 9)",
        "self._conn.mav.param_set_send(1, 1, b'SIM_X', **extra)",
        "c.param_set_send(b'SIM_X', *rest)",
        "c.param_set_send(b'SIM_X', **k)",
        # Any owner but a literal <x>.mav is unknown: both places must be SIM_.
        "m = self._conn.mav\nm.param_set_send(1, 1, b'SIM_RC_FAIL', 1, 9)",
        "def a(self):\n    m = self._conn.mav\n\n\ndef b(self, m):\n"
        "    m.param_set_send('FS_THR_ENABLE', 0, b'SIM_X')",
        "import pymavlink.mavutil as m\nm.param_set_send('FS_THR_ENABLE', 0, b'SIM_X')",
        "def a(self):\n    m = self._conn.mav\n"
        "def b(self):\n    m = self._conn\n    m.param_set_send('FS_THR_ENABLE', 0, b'SIM_X')",
        # mavfile's three-argument form: the third (parm_type) is not a SIM_ name.
        "self._conn.param_set_send('SIM_WIND_DIR', 90, 9)",
        "(m := self._conn.mav).param_set_send(b'SIM_X', 1, b'FENCE_ENABLE', 0, 9)",
        "self._conn.mav.send(mavlink.mavlink_map[23](1, 1, b'FENCE_ENABLE', 0, 9))",
        "cls = mavlink.mavlink_map.get(23)",
        "self._conn.mav.param_ext_set_send(1, 1, b'SIM_RC_FAIL', b'1', 9)",
        "self._conn.mav.rc_channels_override_send(1, 1, *([1500] * 8))",
        "m = mavlink.MAVLink_set_mode_message(1, 1, 4)",
        # A set_sim_param without the SIM_ guard first gets no exemption.
        "def set_sim_param(self, name, v):\n"
        "    self._conn.mav.param_set_send(1, 1, name.encode(), v, 9)",
        "def set_sim_param(self, name, v):\n"
        "    self._conn.mav.param_set_send(1, 1, name.encode(), v, 9)\n"
        "    if not name.startswith('SIM_'):\n"
        "        raise ValueError(name)",
        "def set_sim_param(self, name, v):\n"
        "    if not other.startswith('SIM_'):\n"
        "        raise ValueError(name)\n"
        "    self._conn.mav.param_set_send(1, 1, name.encode(), v, 9)",
        # A function nested in the writer, or a second writer, gets no exemption.
        "def set_sim_param(self, name, v):\n"
        "    if not name.startswith('SIM_'):\n"
        "        raise ValueError(name)\n"
        "    def inner():\n"
        "        self._conn.mav.param_set_send(1, 1, b'FENCE_ENABLE', 0, 9)",
        "def set_sim_param(name, v):\n"
        "    if not name.startswith('SIM_'):\n"
        "        raise ValueError(name)\n"
        "    conn.mav.param_set_send(1, 1, name.encode(), v, 9)\n"
        "class Evil:\n"
        "    def set_sim_param(self, name, v):\n"
        "        if not name.startswith('SIM_'):\n"
        "            raise ValueError(name)\n"
        "        self._conn.mav.param_set_send(1, 1, name.encode(), v, 9)",
    ],
)
def test_the_harness_may_build_nothing_else(line: str) -> None:
    assert set(rules(line, HARNESS)) == {"forbidden-message"}


def test_the_harness_may_send_no_denied_command() -> None:
    assert rules("c = mavlink.MAV_CMD_DO_FENCE_ENABLE", HARNESS) == ["forbidden-command"]


def test_the_real_harness_guards_set_sim_param() -> None:
    # The exemption rests on this guard: without it, the harness's PARAM_SET is refused.
    text = (ROOT / HARNESS).read_text(encoding="utf-8")
    assert cfm.check_source(text, HARNESS) == []
    guard = '        if not name.startswith("SIM_"):\n'
    assert guard in text
    unguarded = cfm.check_source(text.replace(guard, "        if False:\n"), HARNESS)
    assert [f.rule for f in unguarded] == ["forbidden-message"]


def test_tests_are_not_read() -> None:
    assert cfm.is_test_file("software/common/mav/tests/unit/test_forbidden.py")
    assert cfm.is_test_file("simulation/tests/onboard/test_onboard_sitl.py")
    assert cfm.is_test_file("software/onboard/conftest.py")
    assert cfm.is_test_file("software/panel/e2e/helpers.py")
    assert not cfm.is_test_file("software/onboard/src/sd_onboard/executor/executor.py")


@pytest.mark.parametrize(
    "path",
    [
        # Inside a package, a folder named tests or e2e is shipped code and is read.
        "software/onboard/src/sd_onboard/tests/helpers.py",
        "software/onboard/src/sd_onboard/e2e/driver.py",
        "software/onboard/src/sd_onboard/conftest.py",
    ],
)
def test_code_in_a_package_is_read(path: str) -> None:
    assert not cfm.is_test_file(path)


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


def test_this_check_may_name_param_set_only() -> None:
    me = "tools/check_forbidden_mavlink.py"
    assert rules("CALLS = {'param_set_send', 'MAVLink_param_set_message'}", me) == []
    assert rules("kind = 'RC_CHANNELS_OVERRIDE'", me) == ["forbidden-message"]


@pytest.mark.parametrize(
    "path",
    [SIM, "simulation/sd_sim/src/sd_sim/faults.py", "tools/example.py", GROUND],
)
def test_a_sim_param_set_outside_the_harness_is_refused(path: str) -> None:
    # The SIM_ exemption belongs to SIM_PARAM_WRITERS' files only.
    line = "conn.mav.param_set_send(1, 1, b'SIM_RC_FAIL', 1, 9)"
    assert "forbidden-message" in rules(line, path)


@pytest.mark.parametrize(
    "line",
    [
        # The mav taken into another name first, then a plain send.
        "m = conn.mav\nm.send(msg)",
        "m: object = self._conn.mav\nm.send(msg)",
        "self._m = conn.mav\nself._m.send(msg)",
        "(m := conn.mav).send(msg)",
        # pymavlink.mavutil through sys.modules.
        "mavutil = sys.modules['pymavlink.mavutil']",
        # Another object's _conn or write_message is still the gate's.
        "vehicle._conn.close()",
        "conn.write_message(msg)",
        "self.link._conn.recv_msg()",
        # Any _conn or write_message counts, also the class's own: a subclass of
        # sd_mav's MavConnection has the real connection under these names.
        "class S(MavConnection):\n    def go(self, msg):\n        self.write_message(msg)",
        "self._conn.write(msgbuf)",
        "self._conn = MavConnection(url)",
        "w = self.write_message\nw(msg)",
    ],
)
def test_more_raw_sends_in_production(line: str) -> None:
    assert "raw-send" in rules(line)


@pytest.mark.parametrize(
    "line",
    [
        # A mav that is not a connection's, and a dotted name without mavutil.
        "mav = self.mav\nmav.submit(work)",
        "name = 'sd_ground.mav.thread'",
    ],
)
def test_lookalikes_in_production_are_allowed(line: str) -> None:
    assert rules(line) == []
