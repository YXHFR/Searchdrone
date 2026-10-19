"""ArduPilot parameter files and the comparison with a flight controller's parameters."""

from __future__ import annotations

import re
from collections.abc import Iterable
from pathlib import Path

from sd_safety.constants import FIELD_RC_FAILSAFE
from sd_sim import paths

TOLERANCE = 1e-4

# Safety parameters the flight controller must have, checked independently of the
# parameter files: (name, rule, value). Exact values, so a changed safety file fails here
# too. Every FS_*, FENCE_* and BATT_* line in the safety file has a rule, and every value
# the ground station expects (sd_safety.constants.FC_SAFETY_EXPECTED) has an equal rule;
# both are unit-tested.
SAFETY_RULES: tuple[tuple[str, str, float], ...] = (
    ("MAV_GCS_SYSID", "==", 250.0),
    ("FS_GCS_ENABLE", "==", 3.0),
    ("FS_GCS_TIMEOUT", "==", 20.0),
    ("FS_OPTIONS", "==", 8.0),
    ("FS_EKF_ACTION", "==", 1.0),
    ("FS_THR_ENABLE", "==", 0.0),  # simulator profile: no RC link in SITL
    ("FENCE_ENABLE", "==", 1.0),
    ("FENCE_TYPE", "==", 7.0),
    ("FENCE_ACTION", "==", 3.0),
    ("FENCE_ALT_MAX", "==", 125.0),
    ("FENCE_RADIUS", "==", 530.0),
    ("FENCE_MARGIN", "==", 5.0),
    ("AVOID_ENABLE", "==", 3.0),
    ("BATT_MONITOR", "==", 4.0),
    ("BATT_CAPACITY", "==", 10000.0),
    ("BATT_LOW_MAH", "==", 1500.0),
    ("BATT_CRT_MAH", "==", 700.0),
    ("BATT_LOW_VOLT", "==", 0.0),
    ("BATT_CRT_VOLT", "==", 0.0),
    ("BATT_FS_LOW_ACT", "==", 3.0),
    ("BATT_FS_CRT_ACT", "==", 1.0),
    ("LOIT_SPEED_MS", "==", 15.0),
    ("ARMING_SKIPCHK", "==", 0.0),  # every arming check on, the RC check included
    ("ARMING_RUDDER", "==", 0.0),
    ("DISARM_DELAY", "==", 10.0),
    ("RC_OVERRIDE_TIME", "==", 1.0),
    ("RTL_ALT_M", "==", 30.0),
    ("RTL_SPEED_MS", "==", 8.0),
    ("RTL_LOIT_TIME", "==", 0.0),
    ("SRTL_POINTS", "==", 300.0),
    ("SCR_ENABLE", "==", 0.0),
)
# Prefixes whose every line in the safety file must have a rule above.
RULED_PREFIXES = (
    "FS_",
    "FENCE_",
    "AVOID_",
    "BATT_",
    "ARMING_",
    "DISARM_",
    "RC_",
    "RTL_",
    "SRTL_",
    "SCR_",
)

# Real flight (searchdrone-safety.parm + searchdrone-safety.field.parm): the same rules,
# except that the RC transmitter is required, so its failsafe must act.
FIELD_RULES: tuple[tuple[str, str, float], ...] = tuple(
    ("FS_THR_ENABLE", "==", FIELD_RC_FAILSAFE) if name == "FS_THR_ENABLE" else (name, op, value)
    for name, op, value in SAFETY_RULES
)
# What a test's own parameter layer may change in the strict rules, and to which value:
# the real-flight values (stricter than the simulator's), and AVOID_ENABLE 2 (no braking
# before the fence) so that SC16b breaches the height fence every time instead of the
# brake holding the drone just below it. Anything else is refused.
TEST_LAYER_OVERRIDES: dict[str, frozenset[float]] = {
    "FS_THR_ENABLE": frozenset({FIELD_RC_FAILSAFE}),
    "AVOID_ENABLE": frozenset({2.0}),
}


def layer_rules(
    layer: dict[str, float], safety_names: Iterable[str] | None = None
) -> list[tuple[str, str, float]]:
    """``SAFETY_RULES`` with a test layer's values as exact rules. A layer may not touch a
    ruled name or any name the safety file sets (``safety_names``, by default read from
    ``simulation/sitl/params/searchdrone-safety.parm``) unless ``TEST_LAYER_OVERRIDES``
    allows that value, so it can never loosen them (except SC16b's AVOID_ENABLE 2) or
    change the flight modes."""
    if safety_names is None:
        path = paths.REPO / "simulation" / "sitl" / "params" / "searchdrone-safety.parm"
        safety_names = parse_parm(path.read_text(encoding="utf-8"), path.name)
    protected = {name for name, _, _ in SAFETY_RULES} | set(safety_names)
    for name, value in layer.items():
        if name in protected and value not in TEST_LAYER_OVERRIDES.get(name, frozenset()):
            raise ParamFileError(f"testlagret får inte ändra {name} till {value:g}")
    return [(n, "==", layer[n]) if n in layer else (n, op, v) for n, op, v in SAFETY_RULES]


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
