"""Create local config and secrets (``just setup-config``).

- Copies every ``config/<part>.example.toml`` to ``config/<part>.toml`` if missing.
  Existing files are never overwritten.
- Creates ``.env`` from ``.env.example`` if missing and fills every empty value that
  ``.env.example`` marks as ``generated`` with a random 32-character value. Other
  empty values with a known default (user id, data directory) are filled too.
  Optional values (marked ``optional``) stay empty.
- Creates ``secrets/`` with mode 0700.

Values are never printed.
"""

from __future__ import annotations

import os
import re
import secrets
import shutil
import string
import sys
from pathlib import Path

ALPHABET = string.ascii_letters + string.digits
PLACEHOLDERS = {"", "change_me", "change-me", "changeme"}
LINE = re.compile(r"^([A-Z][A-Z0-9_]*)=(.*)$")


def random_secret() -> str:
    return "".join(secrets.choice(ALPHABET) for _ in range(32))


def example_keys(example: Path) -> dict[str, str]:
    """Map each key in .env.example to its kind: generated, optional or default."""
    kinds: dict[str, str] = {}
    kind = "default"
    for raw in example.read_text(encoding="utf-8").splitlines():
        line = raw.strip()
        if line.startswith("#"):
            word = line.lstrip("# ").split(":", 1)[0]
            if word in ("generated", "optional", "filled by `just setup-config`"):
                kind = {"generated": "generated", "optional": "optional"}.get(word, "default")
            continue
        m = LINE.match(line)
        if m:
            kinds[m.group(1)] = kind
            kind = "default"
    return kinds


def defaults() -> dict[str, str]:
    data_home = os.environ.get("XDG_DATA_HOME") or str(Path.home() / ".local" / "share")
    return {
        "SD_UID": str(os.getuid()),
        "SD_GID": str(os.getgid()),
        "SD_DATA_DIR": str(Path(data_home) / "searchdrone"),
    }


def fill_env(root: Path) -> list[str]:
    example, env = root / ".env.example", root / ".env"
    kinds = example_keys(example)
    current: dict[str, str] = {}
    lines: list[str] = []
    if env.is_file():
        lines = env.read_text(encoding="utf-8").splitlines()
        for line in lines:
            m = LINE.match(line.strip())
            if m:
                current[m.group(1)] = m.group(2)
    changed: list[str] = []
    known = defaults()
    new_lines: list[str] = []
    seen: set[str] = set()
    for line in lines:
        m = LINE.match(line.strip())
        if not m:
            new_lines.append(line)
            continue
        key, value = m.group(1), m.group(2)
        seen.add(key)
        new_lines.append(f"{key}={_value(key, value, kinds, known, changed)}")
    for key in kinds:
        if key not in seen:
            new_lines.append(f"{key}={_value(key, '', kinds, known, changed)}")
    old_umask = os.umask(0o077)
    try:
        env.write_text("\n".join(new_lines) + "\n", encoding="utf-8")
    finally:
        os.umask(old_umask)
    env.chmod(0o600)
    return changed


def _value(
    key: str, value: str, kinds: dict[str, str], known: dict[str, str], changed: list[str]
) -> str:
    if value.strip().strip('"').lower() not in PLACEHOLDERS:
        return value
    kind = kinds.get(key, "default")
    if kind == "generated":
        changed.append(key)
        return random_secret()
    if key in known:
        changed.append(key)
        return known[key]
    return ""


def copy_configs(root: Path) -> list[str]:
    copied = []
    for example in sorted((root / "config").glob("*.example.toml")):
        target = example.with_name(example.name.replace(".example.toml", ".toml"))
        if not target.exists():
            shutil.copyfile(example, target)
            copied.append(target.name)
    return copied


def main(argv: list[str] | None = None) -> int:
    args = sys.argv[1:] if argv is None else argv
    root = Path(args[0]) if args else Path()
    for name in copy_configs(root):
        print(f"skapade config/{name}")
    for key in fill_env(root):
        print(f".env: fyllde i {key}")
    secrets_dir = root / "secrets"
    secrets_dir.mkdir(mode=0o700, exist_ok=True)
    secrets_dir.chmod(0o700)
    print("setup-config klar")
    return 0


if __name__ == "__main__":
    sys.exit(main())
