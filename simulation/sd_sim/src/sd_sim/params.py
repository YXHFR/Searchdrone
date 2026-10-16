"""ArduPilot parameter files and the comparison with a flight controller's parameters."""

from __future__ import annotations

import re
from collections.abc import Iterable
from pathlib import Path

TOLERANCE = 1e-4

# Safety parameters the smoke test checks independently of the files, once the safety
# layer exists: (name, rule, value). The flight controller must satisfy each rule.
SAFETY_RULES: tuple[tuple[str, str, float], ...] = (
    ("MAV_GCS_SYSID", "==", 250.0),
    ("FENCE_ENABLE", "==", 1.0),
    ("BATT_FS_LOW_ACT", "!=", 0.0),
    ("BATT_LOW_MAH", ">", 0.0),
    ("FS_GCS_ENABLE", "!=", 0.0),
)


class ParamFileError(ValueError):
    """A line in a parameter file is not ``NAME value``."""


def parse_parm(text: str, source: str = "<text>") -> dict[str, float]:
    """``NAME value`` or ``NAME,value`` per line; ``#`` starts a comment."""
    out: dict[str, float] = {}
    for no, raw in enumerate(text.splitlines(), 1):
        line = raw.split("#", 1)[0].strip()
        if not line:
            continue
        parts = [p for p in re.split(r"[\s,]+", line) if p]
        if len(parts) != 2:
            raise ParamFileError(f"{source}:{no}: väntade 'NAMN värde', fick {raw!r}")
        try:
            out[parts[0]] = float(parts[1])
        except ValueError as exc:
            raise ParamFileError(f"{source}:{no}: {parts[1]!r} är inget tal") from exc
    return out


def merge_layers(files: Iterable[Path]) -> dict[str, float]:
    """Parameters from the files in load order; a later file wins."""
    merged: dict[str, float] = {}
    for path in files:
        merged.update(parse_parm(path.read_text(encoding="utf-8"), str(path.name)))
    return merged


def mismatches(
    expected: dict[str, float], actual: dict[str, float], tolerance: float = TOLERANCE
) -> list[str]:
    out = []
    for name, value in sorted(expected.items()):
        if name not in actual:
            out.append(f"{name}: saknas i flygkontrollen (väntat {value:g})")
        elif abs(actual[name] - value) > tolerance:
            out.append(f"{name}: {actual[name]:g} (väntat {value:g})")
    return out


def rule_violations(
    actual: dict[str, float], rules: Iterable[tuple[str, str, float]] = SAFETY_RULES
) -> list[str]:
    checks = {
        "==": lambda a, b: abs(a - b) <= TOLERANCE,
        "!=": lambda a, b: abs(a - b) > TOLERANCE,
        ">": lambda a, b: a > b,
    }
    out = []
    for name, op, value in rules:
        if name not in actual:
            out.append(f"{name}: saknas i flygkontrollen")
        elif not checks[op](actual[name], value):
            out.append(f"{name}: {actual[name]:g} uppfyller inte {op} {value:g}")
    return out
