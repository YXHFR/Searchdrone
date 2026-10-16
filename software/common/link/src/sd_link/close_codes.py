"""WebSocket close codes on SD-link. Locked: the onboard side reacts to them."""

from __future__ import annotations

from enum import IntEnum

__all__ = ["FATAL_CLOSE_CODES", "CloseCode"]


class CloseCode(IntEnum):
    GOING_AWAY = 1001  # link lost (no traffic for LINK_LOST_S) or server shutting down
    POLICY_VIOLATION = 1008  # too many invalid frames, or events not consumed (backlog)
    AUTH_FAILED = 4401  # wrong or missing token
    FORBIDDEN = 4403  # unknown vehicle, vehicle mismatch, or an Origin header
    BAD_HELLO = 4400  # first message not Hello, no Hello in time, or wrong schema_major
    REPLACED = 4409  # replaced by a newer connection for the same vehicle


# After these the client waits AUTH_RETRY_S and logs critical; others use normal backoff.
FATAL_CLOSE_CODES = frozenset({CloseCode.AUTH_FAILED, CloseCode.FORBIDDEN, CloseCode.BAD_HELLO})
