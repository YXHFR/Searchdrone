"""Write a new random commander token to ``secrets/brain_token`` (``just brain-token``).

32 random bytes, base64url, mode 0600. The value is never printed. Restart both
``sd-ground`` and ``sd-brain`` afterwards.
"""

from __future__ import annotations

import base64
import os
import secrets
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def write_token(path: Path) -> None:
    path.parent.mkdir(mode=0o700, parents=True, exist_ok=True)
    if path.is_symlink():
        raise SystemExit(f"{path} är en symbolisk länk. Ta bort den och kör igen.")
    token = base64.urlsafe_b64encode(secrets.token_bytes(32)).decode().rstrip("=")
    tmp = path.with_name(f".{path.name}.{secrets.token_hex(4)}.tmp")
    fd = os.open(tmp, os.O_CREAT | os.O_EXCL | os.O_WRONLY, 0o600)
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as fh:
            fh.write(token + "\n")
        tmp.replace(path)
    except BaseException:
        tmp.unlink(missing_ok=True)
        raise


def main(argv: list[str] | None = None) -> int:
    target = Path(argv[0]) if argv else ROOT / "secrets" / "brain_token"
    write_token(target)
    print(f"Nytt befälhavartoken i {target}. Starta om sd-ground och sd-brain.")
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
