"""Fixtures for the scenario tests: a fresh simulator and a fresh migrated database."""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "ground"))
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "onboard"))

from test_mission_flow_sitl import (  # pyright: ignore[reportMissingImports]
    fresh_db,
    migrated_db,
    passwords,
)
from test_onboard_sitl import sim  # pyright: ignore[reportMissingImports]

__all__ = ["fresh_db", "migrated_db", "passwords", "sim"]
