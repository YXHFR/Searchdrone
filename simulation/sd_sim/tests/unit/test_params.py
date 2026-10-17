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


FLIGHT_FILE = SAFETY_FILE.with_name("searchdrone-safety-flight.parm")


def test_flight_profile_requires_the_rc_transmitter() -> None:
    sim = params.parse_parm(SAFETY_FILE.read_text(), SAFETY_FILE.name)
    flight = params.parse_parm(FLIGHT_FILE.read_text(), FLIGHT_FILE.name)
    assert params.rule_violations(sim) == []
    assert params.rule_violations(flight, params.FLIGHT_SAFETY_RULES) == []
    # Same parameters; only the RC failsafe differs between simulator and real flight.
    assert set(sim) == set(flight)
    assert {k for k in sim if sim[k] != flight[k]} == params.PROFILE_DIFFERENCES
    assert flight["FS_THR_ENABLE"] != 0
    rc_check = 1 << 6  # ArduPilot's arming check bit for the RC channels
    assert int(flight["ARMING_SKIPCHK"]) & rc_check == 0
