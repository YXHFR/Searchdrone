"""Wall-clock time for timestamps people read (decisions use monotonic time)."""

from __future__ import annotations

import datetime as dt

__all__ = ["utc_now"]


def utc_now() -> dt.datetime:
    return dt.datetime.now(dt.UTC)
