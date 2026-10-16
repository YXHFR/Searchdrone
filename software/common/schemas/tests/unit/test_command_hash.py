"""The approval hash is fixed: ground and onboard must agree byte for byte."""

from __future__ import annotations

from pathlib import Path

from sd_schemas.approval import command_hash
from sd_schemas.skills import GotoParams, HoldParams

GOLDEN = Path(__file__).parent.parent / "golden" / "command_hash.txt"


def test_command_hash_golden() -> None:
    params = GotoParams(lat=-35.36146, lon=149.16523, alt_m=40, speed_ms=8)
    assert command_hash("goto", params, 2) == GOLDEN.read_text().strip()


def test_hash_covers_skill_params_defaults_and_version() -> None:
    goto = GotoParams(lat=-35.36146, lon=149.16523, alt_m=40, speed_ms=8)
    base = command_hash("goto", goto, 2)
    assert command_hash("goto", goto, 3) != base
    assert command_hash("goto", goto.model_copy(update={"alt_m": 41}), 2) != base
    assert command_hash("hold", HoldParams(), 2) == command_hash("hold", HoldParams(reason=""), 2)
