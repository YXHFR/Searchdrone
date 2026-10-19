"""The link-loss and GPS constants against the flight controller's parameter files and the
Lua script: the onboard timings only hold if the flight controller agrees with them."""

import re
from pathlib import Path

from sd_safety import constants as c
from sd_sim import params, paths

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
