"""Test-only helpers for SD-link (fake clock, in-memory and lossy transports, a lossy
WebSocket proxy, fake peers). Never imported by production packages."""

from importlib.metadata import version

from sd_link_testing.clock import FakeClock
from sd_link_testing.fakes import FakeGround, FakeLink, next_inbound
from sd_link_testing.memory import LossyTransport, MemoryTransport, transport_pair
from sd_link_testing.proxy import LossyProxy

__version__ = version("sd-link-testing")

__all__ = [
    "FakeClock",
    "FakeGround",
    "FakeLink",
    "LossyProxy",
    "LossyTransport",
    "MemoryTransport",
    "__version__",
    "next_inbound",
    "transport_pair",
]
