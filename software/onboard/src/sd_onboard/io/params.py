"""The flight controller's parameter files (``NAME value`` per line, ``#`` comments).

PF6 compares the flight controller with the effective set: the files in load order, a
later file wins.
"""

from __future__ import annotations

import re
from collections.abc import Iterable
from pathlib import Path

__all__ = ["effective_params", "parse_parm"]


def parse_parm(text: str, source: str = "<text>") -> dict[str, float]:
    out: dict[str, float] = {}
    for no, raw in enumerate(text.splitlines(), 1):
        line = raw.split("#", 1)[0].strip()
        if not line:
            continue
        parts = [p for p in re.split(r"[\s,]+", line) if p]
        if len(parts) != 2:
            raise ValueError(f"{source}:{no}: expected 'NAME value'")
        out[parts[0]] = float(parts[1])
    return out


def effective_params(files: Iterable[Path]) -> dict[str, float]:
    merged: dict[str, float] = {}
    for path in files:
        merged.update(parse_parm(path.read_text(encoding="utf-8"), path.name))
    return merged
