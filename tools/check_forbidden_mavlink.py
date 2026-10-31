"""No code outside the MAVLink gate may build or send forbidden MAVLink.

``sd_mav`` is the gate: ``Vehicle.send`` checks every message against the deny list
(:mod:`sd_mav.denylist`) and the allowlist before anything reaches the flight controller.
This check reads the rest of the repository's Python (``ast``, nothing is imported or
run) and fails when code outside the allowed modules names something on the deny list,
so a forbidden message is caught at review time and not only when it is sent:

* ``forbidden-message``: a denied message type (``PARAM_SET``, ``SET_MODE``,
  ``RC_CHANNELS_OVERRIDE`` ...) as a pymavlink constructor
  (``MAVLink_param_set_message``), a send or encode method (``param_set_send``) or as
  the exact string ``"PARAM_SET"``;
* ``forbidden-command``: a denied command (``MAV_CMD_DO_FENCE_ENABLE``,
  ``MAV_CMD_DO_FLIGHTTERMINATION`` ...) by name, or by number in the command position of
  a ``COMMAND_LONG``, ``COMMAND_INT`` or ``MISSION_ITEM_INT``;
* ``forced-arming``: the number 21196 (``COMPONENT_ARM_DISARM`` param2 that skips every
  arming check);
* ``raw-send``: production code under ``software/`` that writes on a pymavlink
  connection itself (``<conn>.mav.<message>_send(...)``) instead of ``Vehicle.send``.

Test code (``tests/`` folders, ``conftest.py``) is not read: tests build forbidden
messages on purpose to see them refused. The allowed modules are listed in ``ALLOWED``
with the rules they are exempt from; adding one needs review.

``uv run tools/check_forbidden_mavlink.py [paths ...]`` checks the files git would take
(tracked or new), or only ``paths``. Exit code 1 prints every finding as
``path:line: rule: message``.
"""

from __future__ import annotations

import argparse
import ast
import subprocess
import sys
from collections.abc import Iterable, Iterator
from dataclasses import dataclass
from pathlib import Path

from pymavlink.dialects.v20 import ardupilotmega as mavlink

from sd_mav.denylist import ARM_FORCE_MAGIC, DENIED_COMMANDS, DENIED_MESSAGES

REPO = Path(__file__).resolve().parent.parent
ROOTS = ("software", "simulation", "tools")
ALL_RULES = frozenset({"forbidden-message", "forbidden-command", "forced-arming", "raw-send"})

# Path prefix (repository-relative, POSIX) -> the rules it is exempt from.
ALLOWED: dict[str, frozenset[str]] = {
    # The gate itself: it names the deny list, builds the one allowed PARAM_SET (the
    # fence limits) and owns the connection.
    "software/common/mav/src/sd_mav/": ALL_RULES,
    # The simulator harness injects faults with PARAM_SET of SIM_* parameters only (it
    # raises for any other name) and talks to SITL on its own connection.
    "simulation/sd_sim/src/sd_sim/harness.py": frozenset({"forbidden-message", "raw-send"}),
}
# Only production code is held to the raw-send rule; the simulator and tools may talk
# to SITL on their own connections (still without forbidden traffic).
RAW_SEND_ROOT = "software/"

# Where the command number sits in each constructor's or send method's arguments.
COMMAND_ARG = {"command_long": 2, "command_int": 3, "mission_item_int": 4}


@dataclass(frozen=True)
class Finding:
    path: str
    line: int
    rule: str
    message: str

    def __str__(self) -> str:
        return f"{self.path}:{self.line}: {self.rule}: {self.message}"


def _command_names() -> dict[str, int]:
    enum = mavlink.enums["MAV_CMD"]
    return {enum[number].name: number for number in DENIED_COMMANDS}


DENIED_COMMAND_NAMES = _command_names()
DENIED_MESSAGE_CALLS: dict[str, str] = {}
for _name in DENIED_MESSAGES:
    _low = _name.lower()
    for _call in (f"MAVLink_{_low}_message", f"{_low}_send", f"{_low}_encode"):
        DENIED_MESSAGE_CALLS[_call] = _name


def _call_name(func: ast.expr) -> str | None:
    if isinstance(func, ast.Attribute):
        return func.attr
    if isinstance(func, ast.Name):
        return func.id
    return None


def _message_kind(call_name: str) -> str | None:
    """``command_long`` for ``MAVLink_command_long_message`` / ``command_long_send``."""
    for kind in COMMAND_ARG:
        if call_name in (f"MAVLink_{kind}_message", f"{kind}_send", f"{kind}_encode"):
            return kind
    return None


def _int_value(node: ast.expr) -> int | None:
    if isinstance(node, ast.Constant) and type(node.value) in (int, float):
        value = float(node.value)  # pyright: ignore[reportArgumentType]  # int or float here
        return int(value) if value.is_integer() else None
    return None


def _docstrings(tree: ast.AST) -> set[int]:
    """ids of the string constants that are docstrings (prose, not traffic)."""
    out: set[int] = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Module | ast.ClassDef | ast.FunctionDef | ast.AsyncFunctionDef):
            body = node.body
            if body and isinstance(body[0], ast.Expr) and isinstance(body[0].value, ast.Constant):
                out.add(id(body[0].value))
    return out


def _is_raw_send(func: ast.expr) -> bool:
    """``<x>.mav.<name>_send`` or ``<x>.mav.send``: a write on a pymavlink connection."""
    return (
        isinstance(func, ast.Attribute)
        and (func.attr.endswith("_send") or func.attr == "send")
        and isinstance(func.value, ast.Attribute)
        and func.value.attr == "mav"
    )


def check_source(text: str, path: str) -> list[Finding]:
    """Every finding in one Python file (``path`` is repository-relative, POSIX)."""
    exempt: frozenset[str] = frozenset()
    for prefix, rules in ALLOWED.items():
        if path == prefix or (prefix.endswith("/") and path.startswith(prefix)):
            exempt = exempt | rules
    if exempt >= ALL_RULES:
        return []
    try:
        tree = ast.parse(text, filename=path)
    except SyntaxError as exc:
        return [Finding(path, exc.lineno or 0, "syntax", "filen går inte att läsa som Python")]
    docs = _docstrings(tree)
    out: list[Finding] = []

    def add(node: ast.AST, rule: str, message: str) -> None:
        if rule not in exempt:
            out.append(Finding(path, getattr(node, "lineno", 0), rule, message))

    for node in ast.walk(tree):
        if isinstance(node, ast.Attribute | ast.Name):
            name = node.attr if isinstance(node, ast.Attribute) else node.id
            if name in DENIED_COMMAND_NAMES:
                add(node, "forbidden-command", f"{name} är ett förbjudet kommando (denylist)")
            elif name in DENIED_MESSAGE_CALLS:
                msg = DENIED_MESSAGE_CALLS[name]
                add(node, "forbidden-message", f"{name}: {msg} är förbjudet (denylist)")
        elif isinstance(node, ast.Constant) and id(node) not in docs:
            value = node.value
            if isinstance(value, bytes):
                value = value.decode("ascii", errors="replace")
            if isinstance(value, str) and value in DENIED_MESSAGES:
                add(node, "forbidden-message", f'"{value}" är ett förbjudet meddelande (denylist)')
            elif _int_value(node) == ARM_FORCE_MAGIC and not isinstance(value, bool):
                add(node, "forced-arming", f"{ARM_FORCE_MAGIC} tvingar armering förbi kontrollerna")
        if isinstance(node, ast.Call):
            call = _call_name(node.func)
            kind = _message_kind(call) if call else None
            if kind is not None:
                pos = COMMAND_ARG[kind]
                arg = node.args[pos] if len(node.args) > pos else None
                for kw in node.keywords:
                    if kw.arg == "command":
                        arg = kw.value
                number = _int_value(arg) if arg is not None else None
                if number is not None and number in DENIED_COMMANDS:
                    add(node, "forbidden-command", f"kommando {number} i {call} är förbjudet")
            if path.startswith(RAW_SEND_ROOT) and _is_raw_send(node.func):
                add(
                    node,
                    "raw-send",
                    f"{ast.unparse(node.func)}: skickar förbi Vehicle.send (sd_mav-grinden)",
                )
    return sorted(out, key=lambda f: (f.line, f.rule))


def is_test_file(path: str) -> bool:
    parts = path.split("/")
    return "tests" in parts or "e2e" in parts or parts[-1] == "conftest.py"


def candidate_files(root: Path) -> list[str]:
    """Python files git would take (tracked or new) under ``ROOTS``, tests left out."""
    args = ["ls-files", "--cached", "--others", "--exclude-standard", "--"]
    args += [f"{r}/*.py" for r in ROOTS]
    out = subprocess.run(  # noqa: S603 - fixed git arguments, no shell
        ["git", "-C", str(root), *args],  # noqa: S607 - git from PATH
        capture_output=True,
        text=True,
        check=True,
    ).stdout
    return sorted({p for p in out.splitlines() if p and not is_test_file(p)})


def check_files(root: Path, paths: Iterable[str]) -> Iterator[Finding]:
    for rel in paths:
        file = root / rel
        if file.is_file():
            yield from check_source(file.read_text(encoding="utf-8"), rel)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=(__doc__ or "").splitlines()[0])
    parser.add_argument("paths", nargs="*", help="repository-relative files (default: all)")
    args = parser.parse_args(argv)
    paths = args.paths or candidate_files(REPO)
    findings = list(check_files(REPO, paths))
    for finding in findings:
        print(finding)
    if findings:
        print(
            f"{len(findings)} fynd: förbjuden MAVLink utanför sd_mav. Skicka via Vehicle.send, "
            "eller (efter granskning) lägg modulen i ALLOWED i tools/check_forbidden_mavlink.py."
        )
        return 1
    print(f"check_forbidden_mavlink: {len(paths)} filer, inga fynd")
    return 0


if __name__ == "__main__":
    sys.exit(main())
