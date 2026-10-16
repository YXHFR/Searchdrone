"""The logger handed to ``websockets``. It never writes the link token.

``websockets`` logs every handshake header at DEBUG, ``Authorization`` included.
This logger is pinned to INFO and also redacts any Bearer value that reaches it.
"""

from __future__ import annotations

import logging
import re

__all__ = ["WS_LOGGER", "redact"]

_BEARER = re.compile(r"(?i)\bbearer\s+\S+")


def redact(text: str) -> str:
    return _BEARER.sub("Bearer [REDACTED]", text)


class _RedactBearer(logging.Filter):
    def filter(self, record: logging.LogRecord) -> bool:
        if isinstance(record.msg, str):
            record.msg = redact(record.msg)
        if isinstance(record.args, tuple):
            record.args = tuple(redact(a) if isinstance(a, str) else a for a in record.args)
        return True


WS_LOGGER = logging.getLogger("sd_link.ws")
WS_LOGGER.setLevel(logging.INFO)
WS_LOGGER.addFilter(_RedactBearer())
