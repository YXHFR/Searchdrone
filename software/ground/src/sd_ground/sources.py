"""What the ground station knows about each vehicle, behind small interfaces.

The MAVLink thread (``sd_ground.mav``) and the SD-link session (``sd_ground.link``) fill
these in. Until they exist, ``NoVehicleLink`` and ``NoSdLink`` report "down", so
``/health`` says ``ready = false`` and nothing pretends a vehicle is connected. Tests pass
fakes with the same shape.

``EmergencyPath`` is the STOP / RETURN HOME handler that runs in the ingress thread
(``sd_ground.ingress``). It must not touch the database or the main event loop.
"""

from __future__ import annotations

import datetime as dt
from typing import Literal, Protocol

from sd_ground.auth import Principal
from sd_schemas.api import ActionDispatched, LinkStatus, Telemetry
from sd_schemas.link import StateSnapshot

__all__ = [
    "EmergencyAction",
    "EmergencyPath",
    "EmergencyUnavailableError",
    "NoEmergencyPath",
    "NoSdLink",
    "NoVehicleLink",
    "PartState",
    "SdLink",
    "VehicleLink",
]

PartState = Literal["ok", "degraded", "down"]
EmergencyAction = Literal["stop", "return_home"]


class VehicleLink(Protocol):
    """The ground station's own MAVLink channel to the flight controllers."""

    def state(self) -> PartState: ...

    def connected(self, vehicle_id: str) -> bool: ...

    def telemetry(self, vehicle_id: str) -> Telemetry | None: ...

    def in_flight(self, vehicle_id: str) -> bool: ...

    def airborne_since(self) -> dt.datetime | None:
        """When the earliest vehicle now in the air took off (wall clock), else ``None``."""
        ...

    def onboard_state(self) -> PartState:
        """The companion computer's heartbeat (MAVLink compid 191)."""
        ...


class SdLink(Protocol):
    def state(self) -> PartState: ...

    def link_status(self, vehicle_id: str) -> LinkStatus: ...

    def snapshot(self, vehicle_id: str) -> StateSnapshot | None: ...


class EmergencyUnavailableError(Exception):
    """The emergency path cannot act; ``code`` is the API code (e.g. ``NO_PATH``)."""

    def __init__(self, status: int, code: str, detail_sv: str) -> None:
        super().__init__(code)
        self.status, self.code, self.detail_sv = status, code, detail_sv


class EmergencyPath(Protocol):
    def dispatch(
        self,
        action: EmergencyAction,
        vehicle_id: str,
        *,
        client_action_id: str,
        via: Literal["button", "chat", "cli"],
        by: Principal,
    ) -> ActionDispatched: ...

    def vehicles(self) -> list[str]:
        """Vehicles with a live MAVLink or SD-link connection (for stop_all)."""
        ...


class NoVehicleLink:
    def state(self) -> PartState:
        return "down"

    def connected(self, vehicle_id: str) -> bool:
        return False

    def telemetry(self, vehicle_id: str) -> Telemetry | None:
        return None

    def in_flight(self, vehicle_id: str) -> bool:
        return False

    def airborne_since(self) -> dt.datetime | None:
        return None

    def onboard_state(self) -> PartState:
        return "down"


class NoSdLink:
    def state(self) -> PartState:
        return "down"

    def link_status(self, vehicle_id: str) -> LinkStatus:
        return LinkStatus(sdlink="lost", mavlink="lost", onboard_heartbeat_age_ms=None, rtt_ms=None)

    def snapshot(self, vehicle_id: str) -> StateSnapshot | None:
        return None


class NoEmergencyPath:
    """Placeholder until the STOP / RETURN HOME path is built (B1-08).

    It refuses with 503 so a caller never believes a stop was sent.
    """

    def dispatch(
        self,
        action: EmergencyAction,
        vehicle_id: str,
        *,
        client_action_id: str,
        via: Literal["button", "chat", "cli"],
        by: Principal,
    ) -> ActionDispatched:
        raise EmergencyUnavailableError(
            503, "NO_PATH", "Nödvägen är inte byggd än. Ingen flygning är möjlig utan den."
        )

    def vehicles(self) -> list[str]:
        return []
