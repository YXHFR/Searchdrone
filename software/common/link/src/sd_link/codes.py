"""Reasons the link itself gives (not gate codes)."""

from __future__ import annotations

from typing import Final

__all__ = ["LINK_NO_ACK", "LINK_RECONNECT", "MSG_TOO_LARGE", "SCHEMA_INVALID"]

LINK_NO_ACK: Final = "LINK_NO_ACK"  # no Ack after the last resend (or the ttl ran out)
LINK_RECONNECT: Final = "LINK_RECONNECT"  # the connection ended before an Ack
MSG_TOO_LARGE: Final = "MSG_TOO_LARGE"  # Ack reason for a message over MAX_MESSAGE_BYTES
SCHEMA_INVALID: Final = "SCHEMA_INVALID"  # Ack reason for a message that does not validate
