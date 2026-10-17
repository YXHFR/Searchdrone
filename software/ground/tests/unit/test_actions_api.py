"""The panel routes for Resume, takeover, hand-back and action outcomes."""

from __future__ import annotations

from fastapi.testclient import TestClient
from stop_helpers import VID

from sd_ground.context import Ground
from sd_ground.services.latches import Latches
from sd_ground.services.vehicle_actions import VehicleActions
from sd_testing import FakeClock

H = {"X-SD-Client": "panel", "Content-Type": "application/json"}
BODY = {"client_action_id": "0192f4a1-7c2e-7d10-9a3b-5e1f0c2d8a21"}


def _wire(ground: Ground, clock: FakeClock) -> VehicleActions:
    actions = VehicleActions(clock, Latches(clock), vehicle_ids=[VID])
    ground.vehicle_actions = actions
    return actions


def test_resume_without_the_service_is_not_built(logged_in: TestClient) -> None:
    res = logged_in.post(f"/api/v1/vehicles/{VID}/actions/resume", json=BODY, headers=H)
    assert res.status_code == 501


def test_resume_and_outcome(logged_in: TestClient, ground: Ground, clock: FakeClock) -> None:
    actions = _wire(ground, clock)
    res = logged_in.post(f"/api/v1/vehicles/{VID}/actions/resume", json=BODY, headers=H)
    assert res.status_code == 202, res.text
    action_id = res.json()["action_id"]
    assert res.json()["paths"] == {"sdlink": "unavailable"}
    got = logged_in.get(f"/api/v1/vehicles/{VID}/actions/{action_id}")
    assert got.status_code == 200 and got.json()["action"] == "resume"
    other = "0192f4a1-7c2e-7d10-9a3b-5e1f0c2d8a99"
    assert logged_in.get(f"/api/v1/vehicles/{VID}/actions/{other}").status_code == 404
    # Under FLY HOME, Resume is refused with 409.
    seq = actions.latches.next_seq(VID)
    actions.latches.set(VID, "return_home", seq, "operator:demo")
    again = {"client_action_id": "0192f4a1-7c2e-7d10-9a3b-5e1f0c2d8a22"}
    res = logged_in.post(f"/api/v1/vehicles/{VID}/actions/resume", json=again, headers=H)
    assert res.status_code == 409 and res.json()["code"] == "RETURN_HOME_ACTIVE"


def test_takeover_and_handback_routes(
    logged_in: TestClient, ground: Ground, clock: FakeClock
) -> None:
    _wire(ground, clock)
    res = logged_in.post(f"/api/v1/vehicles/{VID}/actions/takeover", json=BODY, headers=H)
    assert res.status_code == 202, res.text
    back = {"client_action_id": "0192f4a1-7c2e-7d10-9a3b-5e1f0c2d8a23"}
    res = logged_in.post(f"/api/v1/vehicles/{VID}/actions/handback", json=back, headers=H)
    assert res.status_code == 202, res.text


def test_commander_token_cannot_resume(
    panel: TestClient, ground: Ground, clock: FakeClock, brain_token: str
) -> None:
    _wire(ground, clock)
    res = panel.post(
        f"/api/v1/vehicles/{VID}/actions/resume",
        json=BODY,
        headers={**H, "Authorization": f"Bearer {brain_token}"},
    )
    assert res.status_code in (401, 403)
