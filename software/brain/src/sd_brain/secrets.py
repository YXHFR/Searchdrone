"""Secrets of ``sd-brain`` and masking them in logs.

* the commander token: read from ``token_file`` (``secrets/brain_token``), never ``.env``;
* ``SD_LLM_API_KEY``: the key to the local model server (llama-server);
* ``SD_CLOUD_API_KEY``: optional key to the cloud model (OpenAI-compatible);
* ``SD_JEV_API_KEY`` and ``SD_OPENAI_API_KEY``: optional keys to the decision engines.

The required ones are refused when empty, a placeholder or shorter than 32 characters.
The keys are read from the environment, or else from the repository's ``.env``; only
the names above are read from that file, so the process never gets the ground
station's database or video secrets.
"""

from __future__ import annotations

import logging
import os
import re
import stat
from collections.abc import Iterable, Mapping
from dataclasses import dataclass, field
from pathlib import Path

__all__ = [
    "BRAIN_KEYS",
    "MIN_SECRET_LEN",
    "SecretError",
    "Secrets",
    "brain_env",
    "check_secret",
    "install_log_masking",
    "load_secrets",
    "mask",
]

MIN_SECRET_LEN = 32
BRAIN_KEYS = (
    "SD_LLM_API_KEY",
    "SD_CLOUD_API_KEY",
    "SD_JEV_API_KEY",
    "SD_OPENAI_API_KEY",
)
_ENV_LINE = re.compile(r"^\s*([A-Z][A-Z0-9_]*)\s*=\s*(.*?)\s*$")


def brain_env(env: Mapping[str, str] | None = None, dotenv: Path | None = None) -> dict[str, str]:
    """The AI's own keys: from ``env`` first, else from ``.env``. Nothing else is read."""
    environ = os.environ if env is None else env
    out = {k: environ[k] for k in BRAIN_KEYS if environ.get(k)}
    path = dotenv if dotenv is not None else Path(".env")
    if path.is_file() and not path.is_symlink():
        for line in path.read_text(encoding="utf-8").splitlines():
            m = _ENV_LINE.match(line)
            if m and m.group(1) in BRAIN_KEYS and m.group(1) not in out:
                out[m.group(1)] = m.group(2).strip().strip('"').strip("'")
    return out


_PLACEHOLDERS = {"change_me", "change-me", "changeme"}


class SecretError(ValueError):
    """A secret is missing or too weak. The message is Swedish and never holds the value."""


def check_secret(name: str, value: str | None) -> str:
    """Return ``value`` stripped, or raise :class:`SecretError`."""
    v = (value or "").strip()
    if not v:
        raise SecretError(f"{name} saknas. Kör: just setup-config")
    if v.lower() in _PLACEHOLDERS:
        raise SecretError(f"{name} är bara en platshållare. Kör: just setup-config")
    if len(v) < MIN_SECRET_LEN:
        raise SecretError(f"{name} är för kort (minst {MIN_SECRET_LEN} tecken)")
    return v


def _optional(name: str, value: str | None) -> str | None:
    v = (value or "").strip()
    if not v:
        return None
    if v.lower() in _PLACEHOLDERS:
        raise SecretError(f"{name} är bara en platshållare. Lämna den tom eller fyll i nyckeln.")
    return v


@dataclass(frozen=True)
class Secrets:
    brain_token: str = field(repr=False)
    llm_api_key: str = field(repr=False)
    cloud_api_key: str | None = field(default=None, repr=False)
    jev_api_key: str | None = field(default=None, repr=False)
    openai_api_key: str | None = field(default=None, repr=False)

    def values(self) -> list[str]:
        return [
            v
            for v in (
                self.brain_token,
                self.llm_api_key,
                self.cloud_api_key,
                self.jev_api_key,
                self.openai_api_key,
            )
            if v
        ]


def _read_token_file(path: Path) -> str:
    if not path.is_file():
        raise SecretError(f"Befälhavartokenet {path} saknas. Kör: just brain-token")
    if path.is_symlink():
        raise SecretError(f"Befälhavartokenet {path} får inte vara en symbolisk länk")
    mode = stat.S_IMODE(path.stat().st_mode)
    if os.name == "posix" and mode & 0o077:
        raise SecretError(f"Befälhavartokenet {path} får bara kunna läsas av dig (chmod 600)")
    return path.read_text(encoding="utf-8")


def load_secrets(
    token_file: str | Path, env: Mapping[str, str] | None = None, dotenv: Path | None = None
) -> Secrets:
    environ = brain_env(env, dotenv)
    token = check_secret("befälhavartokenet", _read_token_file(Path(token_file)))
    return Secrets(
        brain_token=token,
        llm_api_key=check_secret("SD_LLM_API_KEY", environ.get("SD_LLM_API_KEY")),
        cloud_api_key=_optional("SD_CLOUD_API_KEY", environ.get("SD_CLOUD_API_KEY")),
        jev_api_key=_optional("SD_JEV_API_KEY", environ.get("SD_JEV_API_KEY")),
        openai_api_key=_optional("SD_OPENAI_API_KEY", environ.get("SD_OPENAI_API_KEY")),
    )


def mask(text: str, secrets: Iterable[str]) -> str:
    """Replace every secret (and its ``Bearer`` form) in ``text`` with ``***``."""
    out = text
    for value in sorted({s for s in secrets if s}, key=len, reverse=True):
        out = out.replace(value, "***")
    return re.sub(r"(?i)(bearer\s+)[A-Za-z0-9._~+/=-]{16,}", r"\1***", out)


class _MaskingFilter(logging.Filter):
    def __init__(self, secrets: Iterable[str]) -> None:
        super().__init__()
        self._secrets = [s for s in secrets if s]

    def filter(self, record: logging.LogRecord) -> bool:
        message = record.getMessage()
        record.msg = mask(message, self._secrets)
        record.args = None
        return True


def install_log_masking(secrets: Secrets, logger: logging.Logger | None = None) -> None:
    """Mask the secrets in every record that reaches ``logger``'s handlers."""
    target = logger or logging.getLogger()
    flt = _MaskingFilter(secrets.values())
    target.addFilter(flt)
    for handler in target.handlers:
        handler.addFilter(flt)
