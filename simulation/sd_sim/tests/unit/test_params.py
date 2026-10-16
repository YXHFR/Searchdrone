"""sd_sim.params: parameter files, layer order, comparison and the safety rules."""

from __future__ import annotations

from pathlib import Path

import pytest

from sd_sim import params


def test_parse_spaces_commas_and_comments() -> None:
    text = "# header\nFENCE_ENABLE 1\nBATT_LOW_MAH,1350  # 15 %\n\n  MAV_GCS_SYSID\t250\n"
    assert params.parse_parm(text) == {
        "FENCE_ENABLE": 1.0,
        "BATT_LOW_MAH": 1350.0,
        "MAV_GCS_SYSID": 250.0,
    }


@pytest.mark.parametrize("bad", ["FENCE_ENABLE", "FENCE_ENABLE 1 2", "FENCE_ENABLE yes"])
def test_parse_rejects_bad_lines(bad: str) -> None:
    with pytest.raises(params.ParamFileError, match=r"x\.parm:1"):
        params.parse_parm(bad, "x.parm")


def test_later_layer_wins(tmp_path: Path) -> None:
    a, b = tmp_path / "a.parm", tmp_path / "b.parm"
    a.write_text("X 1\nY 2\n")
    b.write_text("Y 3\n")
    assert params.merge_layers([a, b]) == {"X": 1.0, "Y": 3.0}


def test_mismatches_tolerance_and_missing() -> None:
    expected = {"A": 1.0, "B": 2.0, "C": 3.0}
    actual = {"A": 1.00005, "B": 2.5}
    out = params.mismatches(expected, actual)
    assert out == ["B: 2.5 (väntat 2)", "C: saknas i flygkontrollen (väntat 3)"]


def test_safety_rules() -> None:
    good = {
        "MAV_GCS_SYSID": 250.0,
        "FENCE_ENABLE": 1.0,
        "BATT_FS_LOW_ACT": 3.0,
        "BATT_LOW_MAH": 1350.0,
        "FS_GCS_ENABLE": 3.0,
    }
    assert params.rule_violations(good) == []
    bad = {**good, "FENCE_ENABLE": 0.0, "BATT_LOW_MAH": 0.0}
    del bad["FS_GCS_ENABLE"]
    out = params.rule_violations(bad)
    assert len(out) == 3
    assert any(v.startswith("FENCE_ENABLE") for v in out)
    assert "FS_GCS_ENABLE: saknas i flygkontrollen" in out
