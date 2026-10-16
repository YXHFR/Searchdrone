"""Rules about the code itself, checked by reading it."""

from __future__ import annotations

import re
from pathlib import Path

from sd_schemas.reason_codes import ONBOARD_REASON_CODES, is_known

REPO = Path(__file__).resolve().parents[4]
ONBOARD = REPO / "software/onboard/src/sd_onboard"
SAFETY = REPO / "software/common/safety/src/sd_safety"


def _sources(*roots: Path) -> list[Path]:
    return [p for root in roots for p in root.rglob("*.py")]


def test_no_wallclock_in_logic() -> None:
    wall = re.compile(
        r"\btime\.time\(|\btime\.monotonic\(|datetime\.now\(|\.now\(\s*\)|loop\.time\("
    )
    for path in _sources(ONBOARD / "executor", ONBOARD / "skills", SAFETY):
        assert not wall.search(path.read_text()), path


def test_reason_codes_have_swedish_text() -> None:
    code = re.compile(r'"(ONB_[A-Z_]+)"')
    used = {c for path in _sources(ONBOARD) for c in code.findall(path.read_text())}
    assert used, "no onboard codes found"
    assert used <= set(ONBOARD_REASON_CODES), used - set(ONBOARD_REASON_CODES)
    for text in ONBOARD_REASON_CODES.values():
        assert text.strip()


def test_every_returned_code_is_known() -> None:
    returned = re.compile(r'return "([A-Z][A-Z0-9_]+)"')
    for path in _sources(ONBOARD):
        for c in returned.findall(path.read_text()):
            assert is_known(c), (path.name, c)


def test_no_forbidden_mavlink_in_onboard() -> None:
    for path in _sources(ONBOARD):
        text = path.read_text()
        for word in ("PARAM_SET", "MAV_CMD_DO_FENCE_ENABLE", "21196", "MANUAL_CONTROL("):
            assert word not in text, (path.name, word)


def test_fallback_origin_only_in_the_fallback() -> None:
    for path in _sources(ONBOARD):
        if path.name not in ("override.py", "override_fallback.py"):
            assert "SendOrigin.OVERRIDE_FALLBACK" not in path.read_text(), path


def test_no_test_package_in_production() -> None:
    for path in _sources(ONBOARD):
        text = path.read_text()
        assert "sd_link_testing" not in text and "sd_testing" not in text, path
