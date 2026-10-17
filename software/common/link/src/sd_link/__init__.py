"""SD-link: control channel between the ground station and the onboard computer.

JSON envelopes over one WebSocket per vehicle, with acknowledgements, resends,
idempotency and link liveness. The message catalogue is ``sd_schemas.link``.
"""

from importlib.metadata import version

from sd_link.client import LinkClient
from sd_link.clock import Clock
from sd_link.close_codes import CloseCode
from sd_link.server import LinkServer
from sd_link.session import (
    Closed,
    Expired,
    Inbound,
    LinkBusyError,
    LinkClosedError,
    LinkEvent,
    LinkSession,
    LinkState,
    StateChanged,
)
from sd_link.tokens import TokenStore

__version__ = version("sd-link")

__all__ = [
    "Clock",
    "CloseCode",
    "Closed",
    "Expired",
    "Inbound",
    "LinkBusyError",
    "LinkClient",
    "LinkClosedError",
    "LinkEvent",
    "LinkServer",
    "LinkSession",
    "LinkState",
    "StateChanged",
    "TokenStore",
    "__version__",
]
