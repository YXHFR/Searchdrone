"""ArduPilot's own default parameter layers never decide a safety-relevant value.

Needs the ArduPilot tree that SITL is built from (``copter.parm`` and
``gazebo-iris.parm``), so it runs with the SITL tests, where it can never be skipped.
"""

from __future__ import annotations

import importlib.util
from types import ModuleType

import pytest

from sd_sim import paths


def params_check() -> ModuleType:
    path = paths.REPO / "tools" / "params_check.py"
    spec = importlib.util.spec_from_file_location("tools_params_check", path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


@pytest.mark.sitl
@pytest.mark.budget_s(5)
@pytest.mark.skipif(not paths.arducopter_bin().is_file(), reason="ArduCopter SITL is not built")
def test_default_layers_do_not_shadow() -> None:
    check = params_check()
    layers = check.default_layers(paths.ardupilot_home())
    assert all(p.is_file() for p in layers), layers
    safety = set(check.names(check.SAFETY.read_text(encoding="utf-8")))
    assert check.shadow_problems(layers, safety) == []
