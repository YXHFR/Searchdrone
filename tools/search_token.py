"""Write a new random search token to ``secrets/search_token`` (``just search-token``).

The token for the search service (``sd-search``): it may read the search API and nothing
that steers, approves or decides. Same format as the commander token (32 random bytes,
base64url, mode 0600); the value is never printed. Restart ``sd-ground`` afterwards.
"""

from __future__ import annotations

import sys
from pathlib import Path

from brain_token import write_token

ROOT = Path(__file__).resolve().parents[1]


def main(argv: list[str] | None = None) -> int:
    target = Path(argv[0]) if argv else ROOT / "secrets" / "search_token"
    write_token(target)
    print(f"Nytt söktoken i {target}. Starta om sd-ground.")
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
