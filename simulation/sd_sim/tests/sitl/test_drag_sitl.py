"""The model files' EK3_DRAG_* match what SITL suggests for the same frame (within 10 %).

SITL prints ``Suggested EK3_DRAG_BCOEF_* = <b>, EK3_DRAG_MCOEF = <m>`` when it loads a
frame; dead reckoning without GPS needs drag values close to the frame's."""

import json
import re
import subprocess
from pathlib import Path

import pytest

from sd_sim import fpv7, params, paths

pytestmark = [
    pytest.mark.sitl,
    pytest.mark.budget_s(30),
    pytest.mark.skipif(not paths.arducopter_bin().is_file(), reason="ArduCopter SITL is not built"),
]
SUGGESTED = re.compile(r"Suggested EK3_DRAG_BCOEF_\* = ([\d.]+), EK3_DRAG_MCOEF = ([\d.]+)")
PARAMS = paths.REPO / "simulation" / "sitl" / "params"


def _suggestion(model: str, workdir: Path) -> tuple[float, float]:
    try:
        out = subprocess.run(
            [str(paths.arducopter_bin()), "-I", "9", "--model", model, "--wipe"],
            cwd=workdir,
            capture_output=True,
            text=True,
            timeout=8,
        ).stdout
    except subprocess.TimeoutExpired as exc:  # SITL runs until stopped; its start is enough
        raw = exc.stdout or b""
        out = raw.decode() if isinstance(raw, bytes) else raw
    found = SUGGESTED.search(out)
    assert found, out[-2000:]
    return float(found.group(1)), float(found.group(2))


@pytest.mark.parametrize("vehicle", ["sd_quad14", fpv7.MODEL_NAME])
def test_drag_matches_sitl_suggestion(vehicle: str, tmp_path: Path) -> None:
    if vehicle == fpv7.MODEL_NAME:
        frame = f"{vehicle}.json"
        (tmp_path / frame).write_text(json.dumps(fpv7.frame_json()), encoding="utf-8")
        model = f"x:{frame}"
    else:
        frame = "sd_quad14.json"
        (tmp_path / frame).write_bytes(
            (paths.REPO / "simulation" / "sitl" / "frames" / frame).read_bytes()
        )
        model = f"quad:{frame}"
    bcoef, mcoef = _suggestion(model, tmp_path)
    layer = params.parse_parm((PARAMS / f"{vehicle}-sim.parm").read_text(encoding="utf-8"))
    for name in ("EK3_DRAG_BCOEF_X", "EK3_DRAG_BCOEF_Y"):
        assert abs(layer[name] - bcoef) <= 0.1 * bcoef, (name, layer[name], bcoef)
    assert abs(layer["EK3_DRAG_MCOEF"] - mcoef) <= 0.1 * mcoef, (layer["EK3_DRAG_MCOEF"], mcoef)
