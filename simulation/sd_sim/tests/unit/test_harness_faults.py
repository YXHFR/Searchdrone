"""The test rig's writes and checks, with fakes (no SITL): what a test may change in the
flight controller, the strict safety-parameter check and the per-test parameter layers."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import pytest

from sd_sim import harness, params
from sd_sim.errors import InfraError
from sd_sim.sitl import SitlLayout


class FakeMav:
    def __init__(self) -> None:
        self.sets: list[tuple[str, float]] = []

    def param_set_send(self, _s: int, _c: int, name: bytes, value: float, _t: int) -> None:
        self.sets.append((name.decode(), value))


class FakeConn:
    def __init__(self) -> None:
        self.mav = FakeMav()


def make(values: dict[str, float], *, sticky: bool = True) -> harness.Harness:
    """A harness whose flight controller holds ``values``; a ``PARAM_SET`` takes effect
    unless ``sticky`` is False (the controller ignores it)."""
    h = harness.Harness.__new__(harness.Harness)
    h.state = harness.VehicleState()
    conn = FakeConn()
    h._conn = conn  # pyright: ignore[reportPrivateUsage]
    fc = dict(values)

    def read(name: str, timeout: float = 5.0) -> float:
        for set_name, value in conn.mav.sets:
            if sticky and set_name == name:
                fc[name] = value
        if name not in fc:
            raise InfraError(f"parametern {name} lästes inte")
        return fc[name]

    h.read_param = read  # type: ignore[method-assign]
    return h


def safe_values() -> dict[str, float]:
    return {name: value for name, _, value in params.SAFETY_RULES}


def test_only_sim_parameters_may_be_written() -> None:
    h = make({"SIM_RC_FAIL": 0.0, "FENCE_ENABLE": 1.0})
    with pytest.raises(InfraError, match="SIM_"):
        h.set_sim_param("FENCE_ENABLE", 0)
    assert h._conn.mav.sets == []  # pyright: ignore[reportPrivateUsage, reportAttributeAccessIssue]
    h.set_sim_param("SIM_RC_FAIL", 1)
    assert h.read_param("SIM_RC_FAIL") == 1.0


def test_a_write_the_controller_never_confirms_is_infra_error() -> None:
    h = make({"SIM_RC_FAIL": 0.0}, sticky=False)
    with pytest.raises(InfraError, match="SIM_RC_FAIL"):
        h.set_sim_param("SIM_RC_FAIL", 1, timeout=0.3)


def test_safety_violations_strict_and_with_a_test_layer() -> None:
    assert make(safe_values()).safety_violations() == []
    weak = {**safe_values(), "FS_GCS_ENABLE": 0.0}
    assert make(weak).safety_violations() == ["FS_GCS_ENABLE: 0 uppfyller inte == 3"]
    rc = {**safe_values(), "FS_THR_ENABLE": 4.0}
    assert make(rc).safety_violations()  # the simulator's rule is 0 ...
    assert make(rc).safety_violations({"FS_THR_ENABLE": 4.0}) == []  # ... the layer's 4
    # The layer changes one rule; every other one still applies.
    both = {**rc, "FENCE_ENABLE": 0.0}
    assert make(both).safety_violations({"FS_THR_ENABLE": 4.0}) == [
        "FENCE_ENABLE: 0 uppfyller inte == 1"
    ]
    with pytest.raises(InfraError, match="FENCE_RADIUS"):
        make(safe_values()).safety_violations({"FENCE_RADIUS": 5000.0})


def layout(tmp_path: Path, layers: tuple[Path, ...]) -> SitlLayout:
    repo = tmp_path / "repo"
    p = repo / "simulation" / "sitl" / "params"
    (p / "tests").mkdir(parents=True)
    (p / "searchdrone-safety.parm").write_text("")
    (p / "sd_quad14-sim.parm").write_text("")
    ap = tmp_path / "ardupilot"
    return SitlLayout(arducopter=ap / "x", ardupilot_home=ap, repo=repo, test_layers=layers)


def test_a_test_layer_is_loaded_last(tmp_path: Path) -> None:
    own = tmp_path / "repo/simulation/sitl/params/tests/x.parm"
    lay = layout(tmp_path, (own,))
    own.write_text("FS_THR_ENABLE 4\n")
    layers = lay.defaults("builtin")
    assert layers[-1] == own.resolve()
    assert layers[-2].name == "sd_quad14-sim.parm"


@pytest.mark.parametrize("where", ["outside", "missing"])
def test_a_test_layer_must_be_in_the_tests_folder(tmp_path: Path, where: str) -> None:
    other: Any = tmp_path / "elsewhere.parm"
    if where == "outside":
        other.write_text("FS_THR_ENABLE 4\n")
    else:
        other = tmp_path / "repo/simulation/sitl/params/tests/gone.parm"
    with pytest.raises(InfraError, match="tests"):
        layout(tmp_path, (other,)).defaults("builtin")
