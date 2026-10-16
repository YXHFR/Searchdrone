"""SD-link timing and size constants.

The same names live in ``sd_safety.constants`` once the safety package has them;
``tests/unit/test_constants.py`` checks that the values agree. None of these come
from config or the environment.
"""

from __future__ import annotations

from typing import Final

__all__ = [
    "AUTH_RETRY_S",
    "DEDUP_WINDOW",
    "HEARTBEAT_INTERVAL_S",
    "HELLO_ACK_TIMEOUT_S",
    "HELLO_TIMEOUT_S",
    "LINK_LOSS_DETECT_S",
    "LINK_LOST_S",
    "MAX_FRAME_BYTES",
    "MAX_MESSAGE_BYTES",
    "MAX_PENDING_RELIABLE",
    "RECONNECT_INITIAL_S",
    "RECONNECT_JITTER",
    "RECONNECT_MAX_S",
    "SDLINK_MAX_RESENDS",
    "SDLINK_PATH",
    "SDLINK_RESEND_S",
    "TICK_S",
    "TOKEN_BYTES",
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
