"""The fpv7 model is built once per test session; the modules sit one directory up."""

from __future__ import annotations

import sys
from pathlib import Path
from typing import Any

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import assembly
import checks


@pytest.fixture(scope="session")
def params() -> dict[str, Any]:
    return assembly.load_params()


@pytest.fixture(scope="session")
def drone(params: dict[str, Any]) -> assembly.Drone:
    return assembly.build(params)


@pytest.fixture(scope="session")
def report(drone: assembly.Drone, params: dict[str, Any]) -> dict[str, Any]:
    return checks.report(drone, params)
