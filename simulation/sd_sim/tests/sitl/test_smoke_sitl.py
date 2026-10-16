"""SITL smoke flight with builtin physics (needs ArduCopter SITL, see just sim-setup)."""

import json

import pytest

from sd_sim import config, paths
from sd_sim.smoke import ALT_TOLERANCE_M, TAKEOFF_ALT_M, run

pytestmark = [
    pytest.mark.sitl,
    pytest.mark.slow,
    pytest.mark.skipif(not paths.arducopter_bin().is_file(), reason="ArduCopter SITL is not built"),
]


def test_smoke_builtin_guided_takeoff_land() -> None:
    cfg = config.load({"physics": "builtin", "video": "off"})
    cfg.vehicle = "stock"
    code, run_dir = run(cfg)
    result = json.loads((run_dir / "result.json").read_text(encoding="utf-8"))
    assert code == 0, result
    assert abs(result["max_alt_m"] - TAKEOFF_ALT_M) <= ALT_TOLERANCE_M
    assert result["non_loopback_listeners"] == []
    assert result["duration_s"] <= 180
