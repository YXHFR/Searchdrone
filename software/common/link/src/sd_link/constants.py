"""SD-link timing and size constants.

``LINK_LOSS_DETECT_S`` and ``LINK_LOST_S`` also live in ``sd_safety.constants`` once
the safety package has them; ``tests/unit/test_constants_and_tokens.py`` checks that
the values agree. None of these come from config or the environment.
"""

from __future__ import annotations

from typing import Final

__all__ = [
    "APP_DRAIN_S",
    "AUTH_RETRY_S",
    "CLOSE_TIMEOUT_S",
    "DEDUP_WINDOW",
    "HEARTBEAT_INTERVAL_S",
    "HELLO_ACK_TIMEOUT_S",
    "HELLO_TIMEOUT_S",
    "INVALID_WINDOW_S",
    "LINK_LOSS_DETECT_S",
    "LINK_LOST_S",
    "MANUAL_QUARANTINE_S",
    "MANUAL_STALL_GAP_S",
    "MAX_EVENT_BACKLOG",
    "MAX_FRAME_BYTES",
    "MAX_INVALID_FRAMES",
    "MAX_MESSAGE_BYTES",
    "MAX_PENDING_RELIABLE",
    "OPEN_TIMEOUT_S",
    "RECONNECT_INITIAL_S",
    "RECONNECT_JITTER",
    "RECONNECT_MAX_S",
    "SDLINK_MAX_RESENDS",
    "SDLINK_PATH",
    "SDLINK_RESEND_S",
    "STALE_BURST_GAP_S",
    "STALE_GAP_S",
    "STALE_WINDOW_MAX_S",
    "STALE_WINDOW_S",
    "TICK_S",
    "TOKEN_BYTES",
    "WRITE_TIMEOUT_S",
]

SDLINK_PATH: Final = "/sdlink/v1/control"

# Reliability
SDLINK_RESEND_S: Final = 1.0  # resend a reliable message without an Ack after this long
SDLINK_MAX_RESENDS: Final = 3  # then it expires (LINK_NO_ACK)
DEDUP_WINDOW: Final = 10_000  # msg_ids remembered per sender
MAX_PENDING_RELIABLE: Final = 1_000  # unacknowledged reliable messages per connection

# Liveness
HEARTBEAT_INTERVAL_S: Final = 1.0  # both sides, not reliable
LINK_LOSS_DETECT_S: Final = 3.0  # no inbound traffic -> degraded
LINK_LOST_S: Final = 10.0  # no inbound traffic -> lost, connection closed with 1001
TICK_S: Final = 0.05  # how often a session checks resends and liveness
WRITE_TIMEOUT_S: Final = 2.0  # a write blocked longer than this aborts the connection
CLOSE_TIMEOUT_S: Final = 1.0  # wait for the peer's close frame at most this long
OPEN_TIMEOUT_S: Final = 5.0  # WebSocket opening handshake
APP_DRAIN_S: Final = 2.0  # on_session must return this soon after Closed, or is cancelled

# Late frames after a dropout. A sender expires a reliable message 4 s after sending it,
# so a frame it may have given up on can only arrive after an inbound gap of at least
# LINK_LOSS_DETECT_S. Orders, clearances, releases, charters and takeovers arriving
# within STALE_WINDOW_S after such a gap are answered Ack(expired, LINK_STALE).
STALE_GAP_S: Final = LINK_LOSS_DETECT_S
STALE_WINDOW_S: Final = 1.0  # extended while the burst keeps arriving back-to-back
STALE_BURST_GAP_S: Final = 0.1  # closer frames are one burst (sticks and heartbeats never count)
STALE_WINDOW_MAX_S: Final = 3.0  # the window never stays open longer than this after a gap

# Manual control (stick values, 10-20 Hz): never replay a stalled burst
MANUAL_STALL_GAP_S: Final = 0.5  # no stick value for this long = the stream stalled
MANUAL_QUARANTINE_S: Final = 1.0  # stick values arriving this soon after a stall are dropped

# Abuse limits (per connection)
MAX_EVENT_BACKLOG: Final = 4096  # undelivered events; more closes the connection (1008)
MAX_INVALID_FRAMES: Final = 20  # invalid frames per window; more closes it (1008)
INVALID_WINDOW_S: Final = 10.0

# Handshake
HELLO_TIMEOUT_S: Final = 5.0  # server: first message must be Hello within this
HELLO_ACK_TIMEOUT_S: Final = 5.0  # client: HelloAck must come within this

# Reconnect (client)
RECONNECT_INITIAL_S: Final = 0.5
RECONNECT_MAX_S: Final = 5.0
RECONNECT_JITTER: Final = 0.2  # +-20 %
AUTH_RETRY_S: Final = 30.0  # after 4401/4403/4400

# Sizes
MAX_MESSAGE_BYTES: Final = 64 * 1024  # larger -> Ack(rejected, MSG_TOO_LARGE)
MAX_FRAME_BYTES: Final = 1024 * 1024  # larger frames close the connection (1009)
TOKEN_BYTES: Final = 32
