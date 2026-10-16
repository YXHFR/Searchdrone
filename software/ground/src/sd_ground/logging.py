"""structlog setup: one JSON line per event, and secrets never reach a log.

Two guards: keys that name a secret (``token``, ``code``, ``pin``, ``password``, ``cookie``,
``authorization`` ...) are masked, and every secret value the server hands out (login codes,
PINs, session tokens) is registered with ``Redactor.add`` and masked wherever it appears.
"""

from __future__ import annotations

import logging
import re
import sys
import threading
from collections.abc import MutableMapping
from logging.handlers import RotatingFileHandler
from pathlib import Path
from typing import Any

import structlog

__all__ = ["REDACTOR", "Redactor", "get_logger", "setup_logging"]

_SECRET_KEY = re.compile(r"(token|code|pin|password|passwd|secret|cookie|authorization|jwt)", re.I)
_MASK = "***"


class Redactor:
    def __init__(self) -> None:
        self._lock = threading.Lock()
        self._values: set[str] = set()

    def add(self, value: str) -> None:
        if len(value) >= 6:
            with self._lock:
                self._values.add(value)

    def scrub(self, text: str) -> str:
        with self._lock:
            values = list(self._values)
        for value in values:
            if value in text:
                text = text.replace(value, _MASK)
        return text

    def _clean(self, key: str, value: Any) -> Any:
        if _SECRET_KEY.search(key) and not key.endswith(("_codes", "reason_code")):
            return _MASK
        if isinstance(value, str):
            return self.scrub(value)
        if isinstance(value, dict):
            return {k: self._clean(str(k), v) for k, v in value.items()}  # pyright: ignore[reportUnknownVariableType]
        return value

    def __call__(
        self, _: Any, __: str, event: MutableMapping[str, Any]
    ) -> MutableMapping[str, Any]:
        for key in list(event):
            if key in ("event", "level", "logger", "ts"):
                if isinstance(event[key], str):
                    event[key] = self.scrub(event[key])
                continue
            event[key] = self._clean(key, event[key])
        return event


REDACTOR = Redactor()


def setup_logging(log_file: Path | None = None, *, level: int = logging.INFO) -> None:
    handlers: list[logging.Handler] = [logging.StreamHandler(sys.stderr)]
    if log_file is not None:
        log_file.parent.mkdir(parents=True, exist_ok=True, mode=0o700)
        if not log_file.exists():
            log_file.touch(mode=0o600)
        handlers.append(RotatingFileHandler(log_file, maxBytes=50 * 1024 * 1024, backupCount=10))
    root = logging.getLogger()
    for handler in list(root.handlers):
        root.removeHandler(handler)
    for handler in handlers:
        handler.setFormatter(logging.Formatter("%(message)s"))
        root.addHandler(handler)
    root.setLevel(level)
    structlog.configure(
        processors=[
            structlog.contextvars.merge_contextvars,
            structlog.stdlib.add_logger_name,
            structlog.processors.add_log_level,
            structlog.processors.TimeStamper(fmt="iso", utc=True, key="ts"),
            REDACTOR,
            structlog.processors.JSONRenderer(),
        ],
        logger_factory=structlog.stdlib.LoggerFactory(),
        wrapper_class=structlog.make_filtering_bound_logger(level),
        cache_logger_on_first_use=False,
    )


def get_logger(name: str) -> structlog.typing.FilteringBoundLogger:
    return structlog.get_logger(name)  # pyright: ignore[reportReturnType]
