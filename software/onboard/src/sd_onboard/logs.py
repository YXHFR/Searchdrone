"""Logging for ``sd-onboard``: JSON lines to ``state_dir/onboard.jsonl`` and stderr.

A filter replaces every secret (the link token) and every value of a key called
``token`` or ``authorization`` with ``***`` before anything is written.
"""

from __future__ import annotations

import json
import logging
import re
from collections.abc import Iterable
from pathlib import Path
from typing import Final

__all__ = ["MASK", "SecretFilter", "setup_logging"]

MASK: Final = "***"
_KEYED: Final = re.compile(
    r"(?i)\b(token|authorization)(['\"]?\s*[:=]\s*['\"]?)(bearer\s+)?([^\s,'\"}]+)"
)


class SecretFilter(logging.Filter):
    def __init__(self, secrets: Iterable[str]) -> None:
        super().__init__()
        self._secrets = [s for s in secrets if s]

    def mask(self, text: str) -> str:
        for secret in self._secrets:
            text = text.replace(secret, MASK)
        return _KEYED.sub(lambda m: f"{m.group(1)}{m.group(2)}{m.group(3) or ''}{MASK}", text)

    def filter(self, record: logging.LogRecord) -> bool:
        record.msg = self.mask(record.getMessage())
        record.args = None
        return True


class _JsonFormatter(logging.Formatter):
    def format(self, record: logging.LogRecord) -> str:
        return json.dumps(
            {
                "t": self.formatTime(record),
                "level": record.levelname.lower(),
                "logger": record.name,
                "msg": record.getMessage(),
            },
            ensure_ascii=False,
        )


def setup_logging(state_dir: Path, *, secrets: Iterable[str]) -> SecretFilter:
    masker = SecretFilter(secrets)
    state_dir.mkdir(parents=True, exist_ok=True, mode=0o700)
    root = logging.getLogger()
    root.setLevel(logging.INFO)
    for handler in (logging.FileHandler(state_dir / "onboard.jsonl"), logging.StreamHandler()):
        handler.addFilter(masker)
        handler.setFormatter(_JsonFormatter())
        root.addHandler(handler)
    return masker
