"""No code outside the MAVLink gate may build or send forbidden MAVLink.

``sd_mav`` is the gate: ``Vehicle.send`` checks every message against the deny list
(:mod:`sd_mav.denylist`) and the allowlist before anything reaches the flight controller.
This check reads the rest of the repository's Python (``ast``, nothing is imported or
run) and fails when code outside the allowed modules names something on the deny list,
so a forbidden message is caught at review time and not only when it is sent:

* ``forbidden-message``: a denied message type (``PARAM_SET``, ``SET_MODE``,
  ``RC_CHANNELS_OVERRIDE`` ...) as a pymavlink constructor
  (``MAVLink_param_set_message``), a send or encode method (``param_set_send``), its
  message id (``MAVLINK_MSG_ID_PARAM_SET``, or the number looked up in ``mavlink_map``)
  or as the exact string ``"PARAM_SET"`` or ``"param_set_send"`` (``getattr``), also
  when imported under another name;
* ``forbidden-command``: a denied command (``MAV_CMD_DO_FENCE_ENABLE``,
  ``MAV_CMD_DO_FLIGHTTERMINATION`` ...) by name, or by number in the command position of
  a ``COMMAND_LONG``, ``COMMAND_INT`` or ``MISSION_ITEM_INT``.
  The mavfile helpers that send denied traffic themselves (``reboot_autopilot``,
  ``set_servo``, ``calibrate_level``, ``set_mode_*`` ...) are refused everywhere
  outside ``sd_mav`` as ``forbidden-command`` when named;
* ``forced-arming``: the number 21196 (``COMPONENT_ARM_DISARM`` param2 that skips every
  arming check);
* ``raw-send``: production code under ``software/`` that could write on a pymavlink
  connection itself instead of ``Vehicle.send``: ``<conn>.mav.<message>_send(...)``,
  any use of ``pymavlink.mavutil`` (``mavlink_connection``, ``mavfile``, also through
  ``importlib.import_module`` or ``__import__``), a bare ``MAVLink`` encoder,
  ``mavlink_map``, the gate's own connection (``_conn`` and ``write_message`` on
  another object or by bare name; a class's own ``self._conn`` or
  ``self.write_message`` is its database or websocket), ``pymavlink.mavutil`` as a
  string (``sys.modules[...]``), a mavfile helper that sends on its own
  (``reboot_autopilot``, ``arducopter_arm``, ``set_mode*`` ...), called, only
  referenced or spelled as a ``getattr`` string, and any message's ``*_send`` /
  ``*_encode`` method and ``send`` on a ``mav`` (also one taken into another name,
  ``m = conn.mav``). Plain ``set_mode``
  is also the skills' and ``Vehicle``'s method, which takes a ``CopterMode``, so it is
  refused only with mavfile's arguments (a literal mode, ``custom_mode=``) or when
  only referenced.
  Building messages from the dialect for ``Vehicle.send`` stays allowed.

The check is a static net on top of ``Vehicle.send``, not a proof: a command number
that only exists at run time (``cmd = 207`` passed on) is not seen, which is why
production code may not own a connection at all. Names built at run time (an f-string,
``21195 + 1``) are not seen either, nor a mavfile helper spelled as a ``getattr`` string
outside production. The harness's ``set_sim_param`` is trusted to send the name it
checked: the guard proves the first parameter starts with ``SIM_``, not that this is
the name the ``PARAM_SET`` carries.

Test code (``tests/`` and ``e2e/`` folders and ``conftest.py`` outside a ``src/``
package) is not read: tests build forbidden messages on purpose to see them refused.
The allowed modules are listed in ``ALLOWED`` with the rules they are exempt from, and
in ``ALLOWED_MESSAGES`` with the denied messages they may name (this file names
``PARAM_SET``, which it looks for). ``SIM_PARAM_WRITERS`` lets the simulator harness
send PARAM_SET from ``set_sim_param``, which refuses any name but ``SIM_*`` at run
time, and in it may carry only that guarded name or a literal ``SIM_*`` one; anywhere
else in the harness a PARAM_SET needs a literal ``SIM_*`` name, and outside these files
none is allowed. Adding
an entry to any of these needs review.

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
    # The simulator harness talks to SITL on its own connection.
    "simulation/sd_sim/src/sd_sim/harness.py": frozenset({"raw-send"}),
}
# Path -> the denied messages it may name (any spelling).
ALLOWED_MESSAGES: dict[str, frozenset[str]] = {
    # This check names the messages it looks for; it sends nothing.
    "tools/check_forbidden_mavlink.py": frozenset({"PARAM_SET"}),
}
# Path -> the functions in it that may send PARAM_SET because they refuse any name but
# SIM_* at run time: their first statement must be ``if not <name>.startswith("SIM_"):
# raise ...``, or they get no exemption. Elsewhere in the file a PARAM_SET needs a
# literal SIM_* name.
SIM_PARAM_WRITERS: dict[str, frozenset[str]] = {
    # The harness injects faults (SIM_RC_FAIL, the wind ...) through set_sim_param only.
    "simulation/sd_sim/src/sd_sim/harness.py": frozenset({"set_sim_param"}),
}
SIM_PARAM_PREFIX = "SIM_"
# Only production code is held to the raw-send rule; the simulator and tools may talk
# to SITL on their own connections (still without forbidden traffic).
RAW_SEND_ROOT = "software/"

# Calls that write on a pymavlink connection without Vehicle.send (mavutil.mavfile).
MAVFILE_HELPERS = frozenset(
    {
        "reboot_autopilot",
        "arducopter_arm",
        "arducopter_disarm",
        "set_mode_apm",
        "set_mode_px4",
        "set_mode_auto",
        "set_mode_rtl",
        "set_mode_loiter",
        "set_mode_manual",
        "set_mode_fbwa",
        "set_servo",
        "set_relay",
        "calibrate_level",
        "calibrate_pressure",
        "write_message",
    }
)
# The helpers that send a denied command or message themselves (reboot, servo, mode,
# calibration ...): refused everywhere outside the gate, not only in production.
DENIED_HELPERS = MAVFILE_HELPERS - {"write_message"}
# Names that open or are a connection, a raw encoder or the gate's own connection.
CONNECTION_NAMES = frozenset(
    {"mavutil", "mavlink_connection", "mavfile", "MAVLink", "mavlink_map", "_conn"}
)
IMPORTERS = frozenset({"import_module", "__import__"})
# Names other code also uses for itself: a class's own ``_conn`` (a database) and
# ``write_message`` (a websocket) are not the gate's. Reached on another object
# (``vehicle._conn``, ``conn.write_message``) or by bare name they still count.
OWN_NAMES = frozenset({"_conn", "write_message"})

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
# Every spelling of a denied message -> the message: constructor, send and encode
# methods and the message id constant.
DENIED_MESSAGE_SPELLINGS: dict[str, str] = {}
# Message id -> denied message.
DENIED_MESSAGE_IDS: dict[int, str] = {}
for _name in DENIED_MESSAGES:
    _low = _name.lower()
    for _call in (f"MAVLink_{_low}_message", f"{_low}_send", f"{_low}_encode"):
        DENIED_MESSAGE_SPELLINGS[_call] = _name
    DENIED_MESSAGE_SPELLINGS[f"MAVLINK_MSG_ID_{_name}"] = _name
    DENIED_MESSAGE_IDS[getattr(mavlink, f"MAVLINK_MSG_ID_{_name}")] = _name
PARAM_SET_CALLS = frozenset({"MAVLink_param_set_message", "param_set_send", "param_set_encode"})
# Every message's send and encode method: writing on a connection's ``mav``, also when
# the ``mav`` was taken into another name first.
DIALECT_SENDS = frozenset(
    f"{cls.msgname.lower()}_{verb}"
    for cls in mavlink.mavlink_map.values()
    for verb in ("send", "encode")
)


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


def _is_raw_send(func: ast.expr, mav_names: frozenset[str] = frozenset()) -> bool:
    """``<x>.mav.<name>_send`` or ``<x>.mav.send``, also on a name the ``mav`` was taken
    into (``m = conn.mav; m.send(msg)``): a write on a pymavlink connection."""
    if not isinstance(func, ast.Attribute):
        return False
    if not (func.attr.endswith("_send") or func.attr == "send"):
        return False
    owner = func.value
    return (isinstance(owner, ast.Attribute) and owner.attr == "mav") or (
        isinstance(owner, ast.Name) and owner.id in mav_names
    )


def _mav_names(tree: ast.AST) -> frozenset[str]:
    """Names bound to a connection's ``mav`` (``m = conn.mav``)."""
    out = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Assign | ast.AnnAssign) and isinstance(node.value, ast.Attribute):
            targets = node.targets if isinstance(node, ast.Assign) else [node.target]
            if node.value.attr == "mav":
                out |= {t.id for t in targets if isinstance(t, ast.Name)}
    return frozenset(out)


def _own_member(node: ast.AST) -> bool:
    """``self.<name>`` / ``cls.<name>``: an attribute of the class's own instance."""
    return (
        isinstance(node, ast.Attribute)
        and isinstance(node.value, ast.Name)
        and node.value.id in ("self", "cls")
    )


def _matches(path: str, prefix: str) -> bool:
    return path == prefix or (prefix.endswith("/") and path.startswith(prefix))


def _is_mavfile_set_mode(node: ast.Call) -> bool:
    """mavfile's ``set_mode("GUIDED")`` / ``set_mode(4, custom_mode=...)``.

    The skills' and ``Vehicle``'s ``set_mode`` take a ``CopterMode`` (a name, never a
    literal); mavfile's takes the mode as a string or a number."""
    first = node.args[0] if node.args else None
    for kw in node.keywords:
        if kw.arg in ("custom_mode", "custom_sub_mode"):
            return True
        if kw.arg == "mode":
            first = kw.value
    return first is None or isinstance(first, ast.Constant)


def _param_arg(node: ast.Call) -> ast.expr | None:
    """The parameter name a PARAM_SET call carries, ``.encode()`` taken off.

    The dialect's ``MAVLink_param_set_message`` / ``param_set_encode`` and
    ``<conn>.mav.param_set_send`` take it third (``param_id``); mavfile's own
    ``param_set_send(parm_name, parm_value)`` takes it first."""
    func = node.func
    mavfile = (
        _call_name(func) == "param_set_send"
        and isinstance(func, ast.Attribute)
        and not (isinstance(func.value, ast.Attribute) and func.value.attr == "mav")
    )
    pos, key = (0, "parm_name") if mavfile else (2, "param_id")
    arg = node.args[pos] if len(node.args) > pos else None
    for kw in node.keywords:
        if kw.arg == key:
            arg = kw.value
    if (
        isinstance(arg, ast.Call)
        and isinstance(arg.func, ast.Attribute)
        and arg.func.attr == "encode"
        and not arg.args
    ):
        arg = arg.func.value
    return arg


def _param_name(node: ast.Call) -> str | None:
    """The literal parameter name of a PARAM_SET (``b"X"``, ``"X"``, ``"X".encode()``)."""
    arg = _param_arg(node)
    value = arg.value if isinstance(arg, ast.Constant) else None
    if isinstance(value, bytes):
        value = value.decode("ascii", errors="replace")
    return value if isinstance(value, str) else None


def _sim_name(node: ast.Call) -> bool:
    name = _param_name(node)
    return name is not None and name.startswith(SIM_PARAM_PREFIX)


def _params(fn: ast.FunctionDef | ast.AsyncFunctionDef) -> list[str]:
    return [a.arg for a in fn.args.args if a.arg not in ("self", "cls")]


def _guards_sim_name(fn: ast.FunctionDef | ast.AsyncFunctionDef) -> bool:
    """The function opens with ``if not <first parameter>.startswith("SIM_"): raise``."""
    params = _params(fn)
    body = fn.body
    if body and isinstance(body[0], ast.Expr) and isinstance(body[0].value, ast.Constant):
        body = body[1:]
    if not params or not body or not isinstance(body[0], ast.If):
        return False
    guard = body[0]
    test = guard.test
    return (
        isinstance(test, ast.UnaryOp)
        and isinstance(test.op, ast.Not)
        and isinstance(test.operand, ast.Call)
        and isinstance(test.operand.func, ast.Attribute)
        and test.operand.func.attr == "startswith"
        and isinstance(test.operand.func.value, ast.Name)
        and test.operand.func.value.id == params[0]
        and len(test.operand.args) == 1
        and isinstance(test.operand.args[0], ast.Constant)
        and test.operand.args[0].value == SIM_PARAM_PREFIX
        and isinstance(guard.body[0], ast.Raise)
    )


def _own_calls(fn: ast.AST) -> Iterator[ast.Call]:
    """The calls in a function's own body, not in a function or class defined inside it."""
    for child in ast.iter_child_nodes(fn):
        if isinstance(child, ast.FunctionDef | ast.AsyncFunctionDef | ast.ClassDef | ast.Lambda):
            continue
        if isinstance(child, ast.Call):
            yield child
        yield from _own_calls(child)


def _sim_param_sets(tree: ast.Module, writers: frozenset[str]) -> set[int]:
    """ids of the PARAM_SET call functions allowed in a SIM_PARAM_WRITERS file.

    A writer is a module-level function or a method, defined once under its name, that
    opens with the SIM_ guard; in it, a PARAM_SET may carry the guarded parameter or a
    literal ``SIM_*`` name, nothing else. Calls in functions nested inside it are not
    covered. Elsewhere in the file only a literal ``SIM_*`` name is allowed; outside
    these files, no PARAM_SET at all."""
    if not writers:
        return set()
    found: dict[str, list[ast.FunctionDef | ast.AsyncFunctionDef]] = {}
    scopes: list[ast.Module | ast.ClassDef] = [tree]
    scopes += [n for n in tree.body if isinstance(n, ast.ClassDef)]
    for scope in scopes:
        for fn in scope.body:
            if isinstance(fn, ast.FunctionDef | ast.AsyncFunctionDef) and fn.name in writers:
                found.setdefault(fn.name, []).append(fn)
    out: set[int] = set()
    for defs in found.values():
        if len(defs) == 1 and _guards_sim_name(defs[0]):
            guarded = _params(defs[0])[0]
            for n in _own_calls(defs[0]):
                arg = _param_arg(n)
                if _call_name(n.func) in PARAM_SET_CALLS and (
                    (isinstance(arg, ast.Name) and arg.id == guarded) or _sim_name(n)
                ):
                    out.add(id(n.func))
    for n in ast.walk(tree):
        if isinstance(n, ast.Call) and _call_name(n.func) in PARAM_SET_CALLS and _sim_name(n):
            out.add(id(n.func))
    return out


def _strings(node: ast.AST) -> Iterator[str]:
    for sub in ast.walk(node):
        if isinstance(sub, ast.Constant) and isinstance(sub.value, str):
            yield sub.value


def check_source(text: str, path: str) -> list[Finding]:
    """Every finding in one Python file (``path`` is repository-relative, POSIX)."""
    exempt: frozenset[str] = frozenset()
    for prefix, rules in ALLOWED.items():
        if _matches(path, prefix):
            exempt = exempt | rules
    may_name: frozenset[str] = frozenset()
    for prefix, messages in ALLOWED_MESSAGES.items():
        if _matches(path, prefix):
            may_name = may_name | messages
    if exempt >= ALL_RULES:
        return []
    try:
        tree = ast.parse(text, filename=path)
    except SyntaxError as exc:
        return [Finding(path, exc.lineno or 0, "syntax", "filen går inte att läsa som Python")]
    docs = _docstrings(tree)
    out: list[Finding] = []
    sim_param_sets = _sim_param_sets(tree, SIM_PARAM_WRITERS.get(path, frozenset()))

    def add(node: ast.AST, rule: str, message: str, denied: str | None = None) -> None:
        if rule in exempt or denied in may_name:
            return
        if denied == "PARAM_SET" and id(node) in sim_param_sets:
            return
        out.append(Finding(path, getattr(node, "lineno", 0), rule, message))

    def denied_message(node: ast.AST, spelled: str) -> None:
        msg = DENIED_MESSAGE_SPELLINGS[spelled]
        add(node, "forbidden-message", f"{spelled}: {msg} är förbjudet (denylist)", msg)

    production = path.startswith(RAW_SEND_ROOT)
    mav_names = _mav_names(tree) if production else frozenset()
    # The function of every call, so a helper only referenced is told from one called.
    called = {id(n.func) for n in ast.walk(tree) if isinstance(n, ast.Call)}

    def raw(node: ast.AST, what: str) -> None:
        add(node, "raw-send", f"{what}: skickar förbi Vehicle.send (sd_mav-grinden)")

    for node in ast.walk(tree):
        if isinstance(node, ast.Import | ast.ImportFrom):
            module = (node.module or "") if isinstance(node, ast.ImportFrom) else ""
            for alias in node.names:
                full = f"{module}.{alias.name}" if module else alias.name
                if production and any(n in CONNECTION_NAMES for n in full.split(".")):
                    raw(node, f"import {full}")
                if alias.name in DENIED_COMMAND_NAMES:
                    add(node, "forbidden-command", f"{alias.name} är ett förbjudet kommando")
                elif alias.name in DENIED_MESSAGE_SPELLINGS:
                    denied_message(node, alias.name)
        if production and isinstance(node, ast.Attribute | ast.Name | ast.Constant):
            name = (
                node.attr
                if isinstance(node, ast.Attribute)
                else node.id
                if isinstance(node, ast.Name)
                else node.value
            )
            if not isinstance(name, str) or id(node) in docs:
                pass
            elif name in OWN_NAMES and _own_member(node):
                pass  # a class's own _conn (a database) or write_message (a websocket)
            elif name in OWN_NAMES and not isinstance(node, ast.Constant):
                raw(node, f"{name} (sd_mav:s anslutning; en klass egna {name} nås via self)")
            elif name in CONNECTION_NAMES or name in DIALECT_SENDS:
                raw(node, name)
            elif isinstance(node, ast.Constant):
                if name in MAVFILE_HELPERS:
                    raw(node, f'"{name}"')
                elif "." in name and CONNECTION_NAMES & set(name.split(".")):
                    raw(node, f'"{name}"')  # sys.modules["pymavlink.mavutil"]
            elif (name in MAVFILE_HELPERS or name == "set_mode") and id(node) not in called:
                raw(node, f"{name} (utan anrop)")
        if not production and isinstance(node, ast.Attribute | ast.Name):
            name = node.attr if isinstance(node, ast.Attribute) else node.id
            if name in DENIED_HELPERS:
                text = f"{name}: mavfile-hjälparen skickar förbjuden MAVLink förbi grinden"
                add(node, "forbidden-command", text)
        if isinstance(node, ast.Attribute | ast.Name):
            name = node.attr if isinstance(node, ast.Attribute) else node.id
            if name in DENIED_COMMAND_NAMES:
                add(node, "forbidden-command", f"{name} är ett förbjudet kommando (denylist)")
            elif name in DENIED_MESSAGE_SPELLINGS:
                denied_message(node, name)
        elif isinstance(node, ast.Constant) and id(node) not in docs:
            value = node.value
            if isinstance(value, bytes):
                value = value.decode("ascii", errors="replace")
            if isinstance(value, str) and value in DENIED_MESSAGES:
                text = f'"{value}" är ett förbjudet meddelande (denylist)'
                add(node, "forbidden-message", text, value)
            elif isinstance(value, str) and value in DENIED_MESSAGE_SPELLINGS:
                denied_message(node, value)
            elif isinstance(value, str) and value in DENIED_COMMAND_NAMES:
                add(node, "forbidden-command", f'"{value}" är ett förbjudet kommando (denylist)')
            elif _int_value(node) == ARM_FORCE_MAGIC and not isinstance(value, bool):
                add(node, "forced-arming", f"{ARM_FORCE_MAGIC} tvingar armering förbi kontrollerna")
        looked_up = None
        if isinstance(node, ast.Subscript) and _call_name(node.value) == "mavlink_map":
            looked_up = node.slice
        elif (
            isinstance(node, ast.Call)
            and isinstance(node.func, ast.Attribute)
            and node.func.attr == "get"
            and _call_name(node.func.value) == "mavlink_map"
            and node.args
        ):
            looked_up = node.args[0]
        if looked_up is not None:
            number = _int_value(looked_up)
            if number in DENIED_MESSAGE_IDS:
                msg = DENIED_MESSAGE_IDS[number]
                add(node, "forbidden-message", f"mavlink_map[{number}]: {msg} är förbjudet", msg)
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
            if production and (
                _is_raw_send(node.func, mav_names)
                or (
                    isinstance(node.func, ast.Attribute)
                    and node.func.attr in MAVFILE_HELPERS
                    and not (node.func.attr in OWN_NAMES and _own_member(node.func))
                )
                or (
                    isinstance(node.func, ast.Attribute)
                    and node.func.attr == "set_mode"
                    and _is_mavfile_set_mode(node)
                )
            ):
                raw(node, ast.unparse(node.func))
            if production and call in IMPORTERS and any("mavutil" in s for s in _strings(node)):
                raw(node, f"{call}(... mavutil ...)")
    return sorted(out, key=lambda f: (f.line, f.rule))


def is_test_file(path: str) -> bool:
    """A test, outside any ``src/`` package: code inside a package is always read."""
    parts = path.split("/")
    for part in parts[:-1]:
        if part == "src":
            return False
        if part in ("tests", "e2e"):
            return True
    return parts[-1] == "conftest.py"


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
