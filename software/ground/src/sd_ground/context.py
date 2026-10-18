"""Everything one running ground station shares: config, clock, bus, sessions and links."""

from __future__ import annotations

import datetime as dt
import socket
from collections import deque
from dataclasses import dataclass, field
from importlib.metadata import version
from typing import TYPE_CHECKING, Any

from sd_ground.auth import BrainToken, PendingLogins, SessionRecord, SessionStore
from sd_ground.bus import EventBus
from sd_ground.config import Loaded, parse_bind
from sd_ground.db.repo import EventRow, Repo, new_event
from sd_ground.logging import REDACTOR, get_logger
from sd_ground.services.vehicle_actions import VehicleActions
from sd_ground.sources import (
    EmergencyPath,
    NoEmergencyPath,
    NoSdLink,
    NoVehicleLink,
    PartState,
    SdLink,
    VehicleLink,
)
from sd_obs.clock import Clock

if TYPE_CHECKING:
    from sd_ground.services.missions import MissionFlow

__all__ = ["EVENT_QUEUE_MAX", "Ground", "lan_addresses"]

EVENT_QUEUE_MAX = 10_000
log = get_logger("sd_ground")


def lan_addresses() -> list[str]:
    """Private IPv4 addresses of this machine, plus ``<hostname>.local``."""
    found: set[str] = set()
    try:
        for info in socket.getaddrinfo(socket.gethostname(), None, socket.AF_INET):
            addr = str(info[4][0])
            if addr.startswith(("10.", "192.168.")) or (
                addr.startswith("172.") and 16 <= int(addr.split(".")[1]) <= 31
            ):
                found.add(addr)
    except OSError:
        pass
    return [*sorted(found), f"{socket.gethostname()}.local"]


@dataclass
class Ground:
    loaded: Loaded
    clock: Clock
    brain_token: BrainToken
    search_token: BrainToken = field(default_factory=lambda: BrainToken(None))
    repo: Repo | None = None
    vehicle_link: VehicleLink = field(default_factory=NoVehicleLink)
    sdlink: SdLink = field(default_factory=NoSdLink)
    emergency: EmergencyPath = field(default_factory=NoEmergencyPath)
    vehicle_actions: VehicleActions | None = None  # the emergency path once wired
    flow: MissionFlow | None = None  # the mission flow (needs the database)
    bus: EventBus = field(default_factory=EventBus)
    lan_hosts: list[str] = field(default_factory=list[str])
    version: str = field(default_factory=lambda: version("sd-ground"))
    db_ok: bool = False
    brain_state: str = "down"
    video_state: PartState = "down"  # set by services.video.VideoMonitor
    operator_seen_mono: float | None = None
    pilot_ws: int = 0
    _events: deque[EventRow] = field(
        default_factory=lambda: deque[EventRow](maxlen=EVENT_QUEUE_MAX)
    )
    logins: PendingLogins = field(init=False)
    sessions: SessionStore = field(init=False)

    def __post_init__(self) -> None:
        self.logins = PendingLogins(
            self.clock,
            code_ttl_s=self.loaded.config.auth.login_code_ttl_s,
            on_pin_locked=self._pin_locked,
        )
        self.sessions = SessionStore(
            self.clock,
            airborne_since=lambda: self.vehicle_link.airborne_since(),  # follows a swapped link
            on_extended=self._session_extended,
        )

    @property
    def config(self):
        return self.loaded.config

    @property
    def test_rig(self) -> bool:
        return self.loaded.config.auth.cli_may_confirm

    def allowed_hosts(self) -> list[str]:
        _, port = parse_bind(self.config.http.panel_bind)
        hosts = [f"localhost:{port}", f"127.0.0.1:{port}"]
        if self.config.lan.enabled:
            _, lan_port = parse_bind(self.config.lan.bind)
            hosts += [f"{h}:{lan_port}" for h in self.lan_hosts]
        return hosts

    # --- login links -----------------------------------------------------------------

    def login_link(self) -> tuple[str, str]:
        """A one-time code and the link that uses it (``#login=`` never reaches a server)."""
        code = self.logins.create_login_code()
        REDACTOR.add(code)
        _, port = parse_bind(self.config.http.panel_bind)
        return code, f"http://localhost:{port}/#login={code}"

    def pair(self) -> tuple[str, str, str]:
        code, pin = self.logins.create_pair()
        REDACTOR.add(code)
        REDACTOR.add(pin)
        host = self.lan_hosts[0] if self.lan_hosts else "localhost"
        _, port = parse_bind(self.config.lan.bind)
        return code, pin, f"https://{host}:{port}/#login={code}"

    def _pin_locked(self) -> None:
        log.warning("auth.pin_locked")
        self.add_event("auth.pin_locked", severity="warning")

    def _session_extended(self, rec: SessionRecord) -> None:
        log.info("session.extended_in_flight", session_id=rec.id)
        self.add_event("session.extended_in_flight", payload={"session_id": rec.id})

    # --- events ----------------------------------------------------------------------

    def add_event(
        self,
        type_: str,
        *,
        severity: str = "info",
        mission_id: str | None = None,
        vehicle_id: str | None = None,
        payload: dict[str, Any] | None = None,
    ) -> None:
        """Queue an audit event for the database. Thread-safe (deque append)."""
        row = new_event(
            type_,
            t=self.clock.utc_now(),
            severity=severity,
            mission_id=mission_id,
            vehicle_id=vehicle_id,
            payload=payload,
        )
        self._events.append(row)

    async def flush_events(self) -> None:
        if self.repo is None or not self._events:
            return
        batch = list(self._events)
        try:
            await self.repo.add_events(batch)
        except Exception as exc:
            log.warning("events.flush_failed", error=type(exc).__name__)
            return
        for _ in batch:
            self._events.popleft()

    def queued_events(self) -> list[EventRow]:
        return list(self._events)

    def now(self) -> dt.datetime:
        return self.clock.utc_now()

    async def sleep(self, seconds: float) -> None:
        await self.clock.sleep(seconds)
