"""Fakes and fixtures for the ground station's unit and component tests (no database).

``FakeVehicleLink`` and ``FakeSdLink`` stand in for the MAVLink thread and the SD-link
session until those parts exist; ``FakeEmergency`` records STOP / RETURN HOME calls.
"""

from __future__ import annotations

import datetime as dt
import sys
from collections.abc import Callable, Iterator
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Literal

import pytest
from fastapi.testclient import TestClient

from sd_ground.app import commander_asgi, create_commander_app, create_panel_app, panel_asgi
from sd_ground.auth import BrainToken, Principal
from sd_ground.config import GroundConfig, Loaded
from sd_ground.context import Ground
from sd_ground.sources import PartState
from sd_schemas.api import ActionDispatched, LinkStatus, Telemetry
from sd_schemas.base import new_id
from sd_schemas.link import StateSnapshot
from sd_testing import FakeClock

sys.path.insert(0, str(Path(__file__).parent))  # stop_helpers

BRAIN_TOKEN = "brain-token-for-tests-0123456789abcdef"
PANEL = "http://127.0.0.1:8080"
COMMANDER = "http://127.0.0.1:7700"
CMAC = (-35.363261, 149.165230)


def telemetry(t: dt.datetime, *, alt: float = 0.0, landed: str = "on_ground") -> Telemetry:
    return Telemetry(
        t=t,
        time_boot_ms=1000,
        lat=CMAC[0],
        lon=CMAC[1],
        alt_amsl_m=584.0 + alt,
        alt_rel_home_m=alt,
        heading_deg=0.0,
        groundspeed_ms=0.0,
        climb_ms=0.0,
        battery_pct=90.0,
        battery_v=16.2,
        flight_mode="LOITER",
        armed=alt > 0,
        landed_state=landed,
        gps_fix=3,
        gps_sats=12,
        home=None,
        dist_to_pilot_m=0.0,
    )


@dataclass
class FakeVehicleLink:
    mav: PartState = "ok"
    onboard: PartState = "ok"
    values: dict[str, Telemetry] = field(default_factory=dict[str, Telemetry])
    flying_since: dt.datetime | None = None

    def state(self) -> PartState:
        return self.mav

    def connected(self, vehicle_id: str) -> bool:
        return vehicle_id in self.values

    def telemetry(self, vehicle_id: str) -> Telemetry | None:
        return self.values.get(vehicle_id)

    def in_flight(self, vehicle_id: str) -> bool:
        return self.flying_since is not None

    def airborne_since(self) -> dt.datetime | None:
        return self.flying_since

    def onboard_state(self) -> PartState:
        return self.onboard


@dataclass
class FakeSdLink:
    link: PartState = "ok"

    def state(self) -> PartState:
        return self.link

    def link_status(self, vehicle_id: str) -> LinkStatus:
        return LinkStatus(
            sdlink="up" if self.link == "ok" else "lost",
            mavlink="up",
            onboard_heartbeat_age_ms=100,
            rtt_ms=5.0,
        )

    def snapshot(self, vehicle_id: str) -> StateSnapshot | None:
        return None


@dataclass
class FakeEmergency:
    calls: list[tuple[str, str, str]] = field(default_factory=list[tuple[str, str, str]])

    def dispatch(
        self,
        action: Literal["stop", "return_home"],
        vehicle_id: str,
        *,
        client_action_id: str,
        via: Literal["button", "chat", "cli"],
        by: Principal,
    ) -> ActionDispatched:
        self.calls.append((action, vehicle_id, by.issuer_ref))
        return ActionDispatched(
            action_id=new_id(),
            action_seq=len(self.calls),
            paths={"mavlink": "queued", "sdlink": "queued"},
        )

    def vehicles(self) -> list[str]:
        return ["sim-01"]


def make_loaded(tmp_path: Path, **sections: dict[str, Any]) -> Loaded:
    data: dict[str, Any] = {"data_dir": str(tmp_path / "data")}
    data.update(sections)
    return Loaded(config=GroundConfig.model_validate(data), root=tmp_path, env={})


@pytest.fixture
def brain_token() -> str:
    return BRAIN_TOKEN


@pytest.fixture
def clock() -> FakeClock:
    return FakeClock()


@pytest.fixture
def vehicle_link() -> FakeVehicleLink:
    return FakeVehicleLink()


@pytest.fixture
def emergency() -> FakeEmergency:
    return FakeEmergency()


@pytest.fixture
def make_ground(
    tmp_path: Path, clock: FakeClock, vehicle_link: FakeVehicleLink, emergency: FakeEmergency
) -> Callable[..., Ground]:
    def build(**sections: dict[str, Any]) -> Ground:
        return Ground(
            loaded=make_loaded(tmp_path, **sections),
            clock=clock,
            brain_token=BrainToken(BRAIN_TOKEN),
            vehicle_link=vehicle_link,
            sdlink=FakeSdLink(),
            emergency=emergency,
            version="0.0.0",
        )

    return build


@pytest.fixture
def ground(make_ground: Callable[..., Ground]) -> Ground:
    return make_ground()


@pytest.fixture
def panel(ground: Ground) -> Iterator[TestClient]:
    app = panel_asgi(ground, create_panel_app(ground))
    with TestClient(app, base_url=PANEL) as client:  # pyright: ignore[reportArgumentType]
        yield client


@pytest.fixture
def commander(ground: Ground) -> Iterator[TestClient]:
    app = commander_asgi(ground, create_commander_app(ground))
    with TestClient(app, base_url=COMMANDER) as client:  # pyright: ignore[reportArgumentType]
        yield client


PANEL_HEADERS = {"X-SD-Client": "panel", "Content-Type": "application/json"}


def _login(panel: TestClient, ground: Ground) -> str:
    code, _ = ground.login_link()
    res = panel.post("/api/v1/session", json={"login_code": code}, headers=PANEL_HEADERS)
    assert res.status_code == 200, res.text
    return res.cookies["sd_session"]


@pytest.fixture
def login(panel: TestClient, ground: Ground) -> Callable[[], str]:
    """Log in the way the panel does; returns the session cookie value."""
    return lambda: _login(panel, ground)


@pytest.fixture
def logged_in(panel: TestClient, login: Callable[[], str]) -> TestClient:
    panel.cookies.set("sd_session", login())
    return panel


@pytest.fixture
def pilot_token(ground: Ground, clock: FakeClock) -> str:
    """A CLI token with role pilot, as ``sd-ground token create`` would make."""
    from sd_ground.auth import SessionRecord, hash_token, new_token

    token = new_token()
    now = clock.utc_now()
    ground.sessions.load(
        [
            SessionRecord(
                id="t1",
                kind="token",
                role="pilot",
                issuer_ref="operator:ci",
                auth_mode="cli",
                token_sha256=hash_token(token),
                created_at=now,
                last_used_at=now,
                expires_at=now + dt.timedelta(hours=24),
                name="ci",
            )
        ]
    )
    return token
