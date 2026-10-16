"""STOP and RETURN HOME: who may press which, and the one place that dispatches them.

Used by the ingress thread (which answers these requests in production) and by the
documented FastAPI routes, so the rule can never differ between the two. Every logged-in
user may press STOP (pilot and observer); RETURN HOME is for the pilot only. Nothing here
touches the database or the main event loop.
"""

from __future__ import annotations

from typing import Literal

from sd_ground.auth import Principal
from sd_ground.sources import EmergencyPath, EmergencyUnavailableError
from sd_schemas.api import ActionDispatched, ActionDispatchedAll, ActionDispatchedVehicle

__all__ = ["ALLOWED_ROLES", "EmergencyRefusedError", "check_role", "dispatch", "dispatch_all"]

Action = Literal["stop", "return_home"]
ALLOWED_ROLES: dict[Action, frozenset[str]] = {
    "stop": frozenset({"pilot", "observer"}),
    "return_home": frozenset({"pilot"}),
}


class EmergencyRefusedError(Exception):
    def __init__(self, status: int, code: str, detail_sv: str) -> None:
        super().__init__(code)
        self.status, self.code, self.detail_sv = status, code, detail_sv


def check_role(action: Action, who: Principal) -> None:
    if who.role not in ALLOWED_ROLES[action]:
        raise EmergencyRefusedError(403, "FORBIDDEN_ROLE", "Bara piloten får flyga hem.")


def dispatch(
    path: EmergencyPath, action: Action, vehicle_id: str, client_action_id: str, who: Principal
) -> ActionDispatched:
    """Raises ``EmergencyRefusedError`` or ``sources.EmergencyUnavailableError``."""
    check_role(action, who)
    return path.dispatch(
        action, vehicle_id, client_action_id=client_action_id, via="button", by=who
    )


def dispatch_all(
    path: EmergencyPath, action: Action, client_action_id: str, who: Principal
) -> ActionDispatchedAll:
    """Send to every connected vehicle, even when one of them fails.

    Fails (503) when no vehicle is connected or when any vehicle could not be reached, so
    an answer of 202 always means every vehicle got the order.
    """
    check_role(action, who)
    vehicles = list(path.vehicles())  # one snapshot for the check and the loop
    if not vehicles:  # an empty list would look like success with nothing sent
        raise EmergencyUnavailableError(
            503,
            "NO_PATH",
            "Ingen drönare är ansluten. Använd RC-sändaren eller flygkontrollens egen väg.",
        )
    sent: list[ActionDispatchedVehicle] = []
    failed: list[str] = []
    for vid in vehicles:
        try:
            result = path.dispatch(
                action, vid, client_action_id=client_action_id, via="button", by=who
            )
        except EmergencyUnavailableError:
            failed.append(vid)
            continue
        sent.append(ActionDispatchedVehicle(**result.model_dump(), vehicle_id=vid))
    if failed:
        raise EmergencyUnavailableError(
            503,
            "NO_PATH",
            f"Kunde inte nå {', '.join(failed)}. Skickat till: "
            f"{', '.join(a.vehicle_id for a in sent) or 'ingen'}.",
        )
    return ActionDispatchedAll(actions=sent)
