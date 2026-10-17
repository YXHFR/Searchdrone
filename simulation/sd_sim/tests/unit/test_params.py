"""sd_sim.params: parameter files, layer order, comparison and the safety rules."""

from __future__ import annotations

from pathlib import Path

import pytest

from sd_sim import params, paths


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


SAFETY_FILE = paths.REPO / "simulation" / "sitl" / "params" / "searchdrone-safety.parm"


def test_safety_rules() -> None:
    good = {name: value for name, _, value in params.SAFETY_RULES}
    assert params.rule_violations(good) == []
    bad = {**good, "FENCE_ENABLE": 0.0, "BATT_LOW_MAH": 0.0, "FS_GCS_ENABLE": 1.0}
    del bad["FS_GCS_TIMEOUT"]
    out = params.rule_violations(bad)
    assert len(out) == 4
    assert any(v.startswith("FENCE_ENABLE") for v in out)
    assert any(v.startswith("FS_GCS_ENABLE") for v in out)  # any other action is wrong too
    assert "FS_GCS_TIMEOUT: saknas i flygkontrollen" in out


def test_rule_operators() -> None:
    rules = (("A", "!=", 0.0), ("B", ">", 1.0))
    assert params.rule_violations({"A": 2.0, "B": 2.0}, rules) == []
    assert len(params.rule_violations({"A": 0.0, "B": 1.0}, rules)) == 2


def test_safety_file_satisfies_the_rules() -> None:
    values = params.parse_parm(SAFETY_FILE.read_text(encoding="utf-8"))
    assert params.rule_violations(values) == []


def test_every_failsafe_fence_and_battery_line_has_a_rule() -> None:
    values = params.parse_parm(SAFETY_FILE.read_text(encoding="utf-8"))
    ruled = {name for name, _, _ in params.SAFETY_RULES}
    unruled = [n for n in values if n.startswith(params.RULED_PREFIXES) and n not in ruled]
    assert unruled == []


def test_rules_agree_with_the_ground_station() -> None:
    from sd_safety.constants import FC_SAFETY_EXPECTED

    exact = {name: value for name, op, value in params.SAFETY_RULES if op == "=="}
    for name, value in FC_SAFETY_EXPECTED.items():
        assert name in exact, f"{name} saknar en exakt regel"
        assert exact[name] == value, name


FIELD_FILE = SAFETY_FILE.with_name("searchdrone-safety.field.parm")


def test_field_layer_requires_the_rc_transmitter() -> None:
    """Real flight: the field layer on top of the safety file keeps every rule, but the
    RC failsafe must act; the simulator file alone does not pass for real flight."""
    sim = params.merge_layers([SAFETY_FILE])
    real = params.merge_layers([SAFETY_FILE, FIELD_FILE])
    assert params.rule_violations(real, params.FIELD_RULES) == []
    assert any(
        v.startswith("FS_THR_ENABLE") for v in params.rule_violations(sim, params.FIELD_RULES)
    )
    assert any(v.startswith("FS_THR_ENABLE") for v in params.rule_violations(real))
    assert {n for n, _, _ in params.FIELD_RULES} == {n for n, _, _ in params.SAFETY_RULES}
    # The field layer changes only rows the safety file has, and keeps the arming checks.
    field = params.parse_parm(FIELD_FILE.read_text(encoding="utf-8"))
    assert set(field) <= set(sim)
    assert field["ARMING_SKIPCHK"] == 0.0


def test_field_rules_pin_the_rc_failsafe_value() -> None:
    real = params.merge_layers([SAFETY_FILE, FIELD_FILE])
    assert params.rule_violations({**real, "FS_THR_ENABLE": 2.0}, params.FIELD_RULES)


def test_a_test_layer_can_only_make_the_rules_stricter() -> None:
    rules = {n: (op, v) for n, op, v in params.layer_rules({"FS_THR_ENABLE": 4.0})}
    assert rules["FS_THR_ENABLE"] == ("==", 4.0)
    assert rules["FENCE_ENABLE"] == ("==", 1.0)  # every other rule still applies
    assert params.layer_rules({"SIM_WIND_SPD": 3.0}) == list(params.SAFETY_RULES)
    with pytest.raises(params.ParamFileError, match="FENCE_ALT_MAX"):
        params.layer_rules({"FENCE_ALT_MAX": 500.0})
    with pytest.raises(params.ParamFileError, match="FS_THR_ENABLE"):
        params.layer_rules({"FS_THR_ENABLE": 2.0})
