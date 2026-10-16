"""Print one value from simulation/versions.lock: ``python3 tools/versions.py <key>``.

Standard library only, so shell scripts can use it before any tool is installed.
"""

from __future__ import annotations

import sys
import tomllib
from pathlib import Path

LOCK = Path(__file__).resolve().parents[1] / "simulation" / "versions.lock"


def main(argv: list[str] | None = None) -> int:
    args = sys.argv[1:] if argv is None else argv
    if len(args) != 1:
        print("usage: versions.py <key>", file=sys.stderr)
        return 2
    data = tomllib.loads(LOCK.read_text(encoding="utf-8"))
    if args[0] not in data:
        print(f"versions.py: '{args[0]}' saknas i simulation/versions.lock", file=sys.stderr)
        return 2
    print(data[args[0]])
    return 0


if __name__ == "__main__":
    sys.exit(main())
