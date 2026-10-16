"""Reasons the link itself gives (not gate codes)."""

from __future__ import annotations

from typing import Final

__all__ = ["LINK_NO_ACK", "LINK_RECONNECT", "LINK_STALE", "MSG_TOO_LARGE", "SCHEMA_INVALID"]

# LINK_NO_ACK and LINK_RECONNECT mean "outcome unknown": the peer may or may not have
# acted (only the Acks may have been lost). Resynchronise from StateSnapshot/OnboardEvent.
LINK_NO_ACK: Final = "LINK_NO_ACK"  # no Ack after the last resend (or the ttl ran out)
LINK_RECONNECT: Final = "LINK_RECONNECT"  # the connection ended before an Ack
# Ack(expired) reason: arrived right after a dropout, possibly late, so not acted on.
# The sender may re-issue after STALE_WINDOW_S, with a new command_id/takeover_id.
LINK_STALE: Final = "LINK_STALE"
MSG_TOO_LARGE: Final = "MSG_TOO_LARGE"  # Ack reason for a message over MAX_MESSAGE_BYTES
SCHEMA_INVALID: Final = "SCHEMA_INVALID"  # Ack reason for a message that does not validate
