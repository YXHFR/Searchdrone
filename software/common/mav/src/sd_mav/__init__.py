"""The vehicle link: MAVLink to the flight controller behind a default-deny allowlist.

``Vehicle.send`` checks every message against :mod:`sd_mav.denylist` and
:mod:`sd_mav.allowlist` before it leaves; nothing in the public API can write a
flight-controller parameter. :mod:`sd_mav.fake` is an in-memory flight controller for
tests.
"""

from importlib.metadata import version

from sd_mav.allowlist import Profile, SendContext, SendEnvelope
from sd_mav.connection import MavConnection, Role, connect
from sd_mav.modes import CopterMode
from sd_mav.vehicle import (
    ControlNotAllowedError,
    FcTelemetry,
    ForbiddenMavlinkError,
    MavResult,
    SendOrigin,
    Vehicle,
)

__version__ = version("sd-mav")

__all__ = [
    "ControlNotAllowedError",
    "CopterMode",
    "FcTelemetry",
    "ForbiddenMavlinkError",
    "MavConnection",
    "MavResult",
    "Profile",
    "Role",
    "SendContext",
    "SendEnvelope",
    "SendOrigin",
    "Vehicle",
    "__version__",
    "connect",
]
