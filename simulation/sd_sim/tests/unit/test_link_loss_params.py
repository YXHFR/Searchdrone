"""The link-loss and GPS constants against the flight controller's parameter files and the
Lua script: the onboard timings only hold if the flight controller agrees with them."""

import re
from pathlib import Path

from sd_safety import constants as c
from sd_sim import params, paths, sitl

PARAMS = paths.REPO / "simulation" / "sitl" / "params"
SAFETY = params.parse_parm((PARAMS / "searchdrone-safety.parm").read_text(encoding="utf-8"))
LUA = paths.REPO / "simulation" / "sitl" / "scripts" / "sd_gcs_loiter_cap.lua"


def test_link_loss_constants() -> None:
    assert SAFETY["FS_GCS_TIMEOUT"] == c.FS_GCS_TIMEOUT_S
    # In LOITER at least 5 s before the flight controller's GCS failsafe looks at the mode.
    assert c.LINK_LOSS_DETECT_S + c.FIND_MOVE_DEADLINE_S <= c.FS_GCS_TIMEOUT_S - 5
    # The Lua backstop never cuts a full find hover short (20 s margin).
    assert c.FC_LOITER_CAP_S >= c.LINK_LOSS_DETECT_S + c.FIND_HOVER_S + 20
    assert SAFETY["EK3_GLITCH_RAD"] < c.SPOOF_JUMP_M
    assert SAFETY["RTL_SPEED_MS"] >= c.FIND_MOVE_SPEED_MS
    assert SAFETY["AHRS_GPS_MINSATS"] == c.GPS_WEAK_SATS
    assert SAFETY["FS_DR_TIMEOUT"] == c.FS_DR_TIMEOUT_S
    # FS_OPTIONS: bit 3 and bit 4, never bit 0-2 (continue in AUTO and friends).
    assert int(SAFETY["FS_OPTIONS"]) == 8 | 16
    assert SAFETY["SCR_ENABLE"] == 1


def test_lua_constants_match() -> None:
    text = LUA.read_text(encoding="utf-8")
    cap = re.search(r"^local CAP_MS = (\d+)", text, re.MULTILINE)
    assert cap is not None
    assert int(cap.group(1)) == round(c.FC_LOITER_CAP_S * 1000)
    assert f"no GCS {c.FC_LOITER_CAP_S:.0f} s" in text
    # Its text must classify as a failsafe onboard (sd_mav.failsafe_signals "failsafe").
    assert re.search(r'send_text\(2, "SD failsafe', text)


def _crc32_bitwise(data: bytes) -> int:
    """ArduPilot's crc_crc32(0, data) written out bit by bit (reflected 0xEDB88320, start
    0, no final inversion), independent of the zlib shortcut in sd_sim.sitl."""
    crc = 0
    for byte in data:
        crc ^= byte
        for _ in range(8):
            crc = (crc >> 1) ^ 0xEDB88320 if crc & 1 else crc >> 1
    return crc


def test_script_crc_is_ardupilots() -> None:
    for data in (b"", b"123456789", LUA.read_bytes()):
        assert sitl.script_crc32(data) == _crc32_bitwise(data)
    assert sitl.script_crc32(b"123456789") == 0x2DFD2D88  # zlib (0xCBF43926) inverts twice


def test_only_our_lua_script_can_arm() -> None:
    """SCR_LD_CHECKSUM and SCR_RUN_CHECKSUM lock the flight controller to exactly the
    scripts in sitl.LUA_SCRIPTS. A changed script fails here until the safety file (and
    sd_sim.params.SAFETY_RULES) carry its new checksum."""
    scripts = [LUA.parent / name for name in sitl.LUA_SCRIPTS]
    want = sitl.scripts_checksum(scripts)
    assert SAFETY["SCR_LD_CHECKSUM"] == want, f"SCR_LD_CHECKSUM ska vara {want}"
    assert SAFETY["SCR_RUN_CHECKSUM"] == want, f"SCR_RUN_CHECKSUM ska vara {want}"
    rules = {name: value for name, _, value in params.SAFETY_RULES}
    assert rules["SCR_LD_CHECKSUM"] == rules["SCR_RUN_CHECKSUM"] == want
    # A float parameter holds the 23-bit value exactly, and -1 (off) is never it.
    assert 0 <= want <= sitl.SCRIPT_CHECKSUM_MASK and float(want) == want
    # Any other byte in the script gives another checksum.
    changed = LUA.read_bytes() + b"\n"
    assert (sitl.script_crc32(changed) & sitl.SCRIPT_CHECKSUM_MASK) != want


def test_model_files_set_the_drag_for_dead_reckoning() -> None:
    for layer in ("sd_quad14-sim.parm", "sd_fpv7-sim.parm"):
        values = params.parse_parm((PARAMS / layer).read_text(encoding="utf-8"))
        for name in ("EK3_DRAG_BCOEF_X", "EK3_DRAG_BCOEF_Y", "EK3_DRAG_MCOEF"):
            assert values.get(name, 0) > 0, (layer, name)


def test_field_profile_has_the_source_switch() -> None:
    field = params.parse_parm(
        Path(PARAMS / "searchdrone-safety.field.parm").read_text(encoding="utf-8")
    )
    assert SAFETY["RC8_OPTION"] == 0
    assert field["RC8_OPTION"] == 90
