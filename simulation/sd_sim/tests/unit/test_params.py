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


# A vehicle's sim layer may raise ATC_ANGLE_MAX this far above FC_ANGLE_MAX_DEG and no
# further. The ground station reads ATC_ANGLE_MAX from the flight controller, but keeps
# FC_ANGLE_MAX_DEG - 10 for its tilt warning when the read fails, which then comes at
# most 15 degrees before the limit instead of 10.
SIM_LEAN_HEADROOM_DEG = 5.0


def lean_limit_problem(name: str, values: dict[str, float], sim_layers: set[str]) -> str | None:
    """Why ``name`` moves the lean limit in a way the tilt warning cannot follow, or None."""
    from sd_safety.constants import FC_ANGLE_MAX_DEG

    angle = values.get("ATC_ANGLE_MAX", FC_ANGLE_MAX_DEG)
    if name in sim_layers:
        if FC_ANGLE_MAX_DEG <= angle <= FC_ANGLE_MAX_DEG + SIM_LEAN_HEADROOM_DEG:
            return None
    elif angle == FC_ANGLE_MAX_DEG:
        return None
    return f"{name}: ATC_ANGLE_MAX {angle:g}"


def test_no_parameter_file_moves_the_lean_limit() -> None:
    """The ground station's tilt warning assumes ArduPilot's default ATC_ANGLE_MAX; a file
    that changes it must change FC_ANGLE_MAX_DEG too. Only a vehicle's declared sim layer
    may raise it, by at most SIM_LEAN_HEADROOM_DEG: the warning then comes early, never
    late. The safety and field files keep the default."""
    from sd_sim.sitl import SIM_PARAM_FILES

    sim_layers = set(SIM_PARAM_FILES.values())
    for path in sorted(SAFETY_FILE.parent.rglob("*.parm")):
        values = params.parse_parm(path.read_text(encoding="utf-8"), path.name)
        layer = str(path.relative_to(SAFETY_FILE.parent))
        assert lean_limit_problem(layer, values, sim_layers) is None


@pytest.mark.parametrize(
    ("name", "angle", "ok"),
    [
        ("sd_fpv7-sim.parm", 30.0, True),
        ("sd_fpv7-sim.parm", 35.0, True),
        ("sd_fpv7-sim.parm", 25.0, False),
        ("sd_fpv7-sim.parm", 36.0, False),
        ("other-sim.parm", 35.0, False),
        ("searchdrone-safety.parm", 35.0, False),
        ("searchdrone-safety.field.parm", 25.0, False),
    ],
)
def test_lean_limit_rule_refuses_lower_higher_and_undeclared(
    name: str, angle: float, ok: bool
) -> None:
    sim_layers = {"sd_quad14-sim.parm", "sd_fpv7-sim.parm"}
    problem = lean_limit_problem(name, {"ATC_ANGLE_MAX": angle}, sim_layers)
    assert (problem is None) is ok


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


def test_a_test_layer_may_only_set_allowed_overrides() -> None:
    rules = {n: (op, v) for n, op, v in params.layer_rules({"FS_THR_ENABLE": 4.0})}
    assert rules["FS_THR_ENABLE"] == ("==", 4.0)
    assert rules["FENCE_ENABLE"] == ("==", 1.0)  # every other rule still applies
    assert params.layer_rules({"SIM_WIND_SPD": 3.0}) == list(params.SAFETY_RULES)
    with pytest.raises(params.ParamFileError, match="FENCE_ALT_MAX"):
        params.layer_rules({"FENCE_ALT_MAX": 500.0})
    with pytest.raises(params.ParamFileError, match="FS_THR_ENABLE"):
        params.layer_rules({"FS_THR_ENABLE": 2.0})


def test_a_test_layer_may_only_turn_off_the_fence_brake() -> None:
    rules = {n: (op, v) for n, op, v in params.layer_rules({"AVOID_ENABLE": 2.0})}
    assert rules["AVOID_ENABLE"] == ("==", 2.0)
    assert rules["FENCE_ALT_MAX"] == ("==", 125.0)
    for value in (0.0, 1.0, 7.0):
        with pytest.raises(params.ParamFileError, match="AVOID_ENABLE"):
            params.layer_rules({"AVOID_ENABLE": value})


def test_only_sc16b_flies_without_the_fence_brake() -> None:
    layers = SAFETY_FILE.parent / "tests"
    no_brake = [
        p.name
        for p in sorted(layers.glob("*.parm"))
        if "AVOID_ENABLE" in params.parse_parm(p.read_text(encoding="utf-8"), p.name)
    ]
    assert no_brake == ["fence_no_brake.parm"]
    text = (layers / "fence_no_brake.parm").read_text(encoding="utf-8")
    assert "# test: test_sc16b_fc_fence_height" in text.splitlines()
    users = [
        p.name
        for p in sorted((paths.REPO / "simulation" / "tests").rglob("*.py"))
        if "fence_no_brake.parm" in p.read_text(encoding="utf-8")
    ]
    assert users == ["test_sc_failsafe.py"]


def test_a_test_layer_cannot_change_what_the_safety_file_sets() -> None:
    # Not ruled, but set by the safety file (the RC mode switch, the GUIDED timeout).
    for name, value in (("FLTMODE1", 6.0), ("GUID_TIMEOUT", 10.0)):
        with pytest.raises(params.ParamFileError):
            params.layer_rules({name: value})
    with pytest.raises(params.ParamFileError):
        params.layer_rules({"ANY_NAME": 1.0}, safety_names=["ANY_NAME"])
    assert params.layer_rules({"ANY_NAME": 1.0}, safety_names=[]) == list(params.SAFETY_RULES)
