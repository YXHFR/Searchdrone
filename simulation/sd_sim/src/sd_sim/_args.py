"""Shared ``key=value`` argument parsing for the simulation entry points.

just passes recipe arguments through unchanged (``just test-sitl sc=SC04``), so the
parsing and the error messages live here. Unknown keys exit with code 2.
"""

from __future__ import annotations

import sys
from collections.abc import Iterable

NOT_BUILT_EXIT = 2


def parse(argv: Iterable[str], allowed: Iterable[str], prog: str) -> dict[str, str]:
    allowed_set = set(allowed)
    out: dict[str, str] = {}
    for raw in argv:
        key, sep, value = raw.partition("=")
        if not sep or not key:
            fail(f"{prog}: '{raw}' är inte på formen nyckel=värde (t.ex. physics=builtin).")
        if key not in allowed_set:
            known = ", ".join(sorted(allowed_set))
            fail(f"{prog}: okänd nyckel '{key}'. Tillåtna nycklar: {known}.")
        out[key] = value
    return out


def fail(message: str) -> None:
    print(message, file=sys.stderr)
    raise SystemExit(2)


def not_built(ticket: str) -> int:
    print(f"inte byggt än ({ticket})")
    return NOT_BUILT_EXIT
