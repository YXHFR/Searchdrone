"""Simulation launcher: ``python -m sd_sim.launch {up|down|status} [--demo] [key=value ...]``.

Only the command-line interface exists so far; starting the stack is not built yet.
"""

from __future__ import annotations

import json
import sys

from sd_sim import _args

ALLOWED_KEYS = (
    "physics",
    "gui",
    "video",
    "brain",
    "script",
    "instance",
    "link_profile",
    "qgc",
    "record",
    "lan",
    "private_db",
    "dry",
)
COMMANDS = ("up", "down", "status")
FLAGS = ("--demo", "--with-db")


def main(argv: list[str] | None = None) -> int:
    argv = list(sys.argv[1:] if argv is None else argv)
    if not argv or argv[0] not in COMMANDS:
        _args.fail(f"sd_sim.launch: ange ett av {', '.join(COMMANDS)}.")
    command, rest = argv[0], argv[1:]
    flags = [a for a in rest if a.startswith("--")]
    for flag in flags:
        if flag not in FLAGS:
            _args.fail(f"sd_sim.launch: okänd flagga '{flag}'. Inställningar skrivs nyckel=värde.")
    values = _args.parse([a for a in rest if not a.startswith("--")], ALLOWED_KEYS, "sd_sim.launch")
    if "record" in values and "--demo" not in flags:
        _args.fail("sd_sim.launch: record= fungerar bara med just demo.")
    if values.get("dry") == "1":
        print(json.dumps({"command": command, "flags": flags, "args": rest}))
        return 0
    return _args.not_built("B0-05")


if __name__ == "__main__":
    sys.exit(main())
