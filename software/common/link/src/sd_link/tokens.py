"""Link tokens: 32 random bytes (base64url). The ground keeps only the SHA-256.

``secrets/link_tokens.toml`` holds one line per vehicle: ``"sim-01" = "<sha256-hex>"``.
Comparison is constant-time (``hmac.compare_digest``) against every entry.
"""

from __future__ import annotations

import base64
import hashlib
import hmac
import os
import re
import secrets
import stat
import tomllib
from collections.abc import Mapping
from pathlib import Path
from typing import Final

from sd_link.constants import TOKEN_BYTES
from sd_schemas.base import VEHICLE_ID_PATTERN

__all__ = [
    "TOKEN_MIN_CHARS",
    "TokenFileError",
    "TokenStore",
    "generate_token",
    "hash_token",
    "read_token_file",
    "write_token_hash",
]

TOKEN_MIN_CHARS: Final = 43  # 32 bytes in base64url without padding
_HEX64 = re.compile(r"^[0-9a-f]{64}$")
_VEHICLE = re.compile(VEHICLE_ID_PATTERN)


class TokenFileError(Exception):
    pass


def generate_token() -> str:
    return base64.urlsafe_b64encode(secrets.token_bytes(TOKEN_BYTES)).rstrip(b"=").decode()


def hash_token(token: str) -> str:
    return hashlib.sha256(token.encode()).hexdigest()


class TokenStore:
    """Vehicle id -> token hash. ``match`` returns the vehicle a token belongs to."""

    def __init__(self, hashes: Mapping[str, str]) -> None:
        for vehicle_id, digest in hashes.items():
            if not _VEHICLE.match(vehicle_id) or not _HEX64.match(digest):
                raise TokenFileError(f"bad entry for {vehicle_id[:40]!r}")
        self._hashes = dict(hashes)

    @classmethod
    def load(cls, path: Path) -> TokenStore:
        if not path.exists():
            return cls({})
        try:
            data = tomllib.loads(path.read_text(encoding="utf-8"))
        except tomllib.TOMLDecodeError as exc:
            raise TokenFileError(f"{path.name}: {exc}") from None
        if not all(isinstance(v, str) for v in data.values()):
            raise TokenFileError(f"{path.name}: every value is a sha256 hex string")
        return cls({k: str(v) for k, v in data.items()})

    def hashes(self) -> dict[str, str]:
        return dict(self._hashes)

    def __contains__(self, vehicle_id: str) -> bool:
        return vehicle_id in self._hashes

    def match(self, token: str | None) -> str | None:
        if not token:
            return None
        digest = hash_token(token)
        found: str | None = None
        for vehicle_id, expected in self._hashes.items():  # no early exit
            if hmac.compare_digest(digest, expected):
                found = vehicle_id
        return found


def write_token_hash(path: Path, vehicle_id: str, digest: str) -> None:
    """Add or replace one vehicle's hash. The file is written with mode 600."""
    hashes = TokenStore.load(path).hashes()
    hashes[vehicle_id] = digest
    TokenStore(hashes)  # validates
    lines = [f'"{v}" = "{d}"' for v, d in sorted(hashes.items())]
    path.parent.mkdir(parents=True, exist_ok=True)
    fd = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_TRUNC, 0o600)
    with os.fdopen(fd, "w", encoding="utf-8") as fh:
        fh.write("\n".join(lines) + "\n")
    path.chmod(0o600)


def read_token_file(path: Path) -> str:
    """Read the onboard token. Refuses a file others can read or a too short token."""
    try:
        mode = path.stat().st_mode
    except FileNotFoundError:
        raise TokenFileError(f"token file missing: {path.name}") from None
    if mode & (stat.S_IRWXG | stat.S_IRWXO):
        raise TokenFileError(f"token file {path.name} must have mode 600")
    token = path.read_text(encoding="utf-8").strip()
    if len(token) < TOKEN_MIN_CHARS:
        raise TokenFileError(f"token in {path.name} is too short")
    return token
