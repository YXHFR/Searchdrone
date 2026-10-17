""" "Fritt runt drönaren": the pilot's confirmation that nobody stands near the drone.

One confirmation per take-off: a pilot with an interactive session clicks it for the
active approved charter while the vehicle is on the ground. It is bound to the mission,
vehicle, charter and charter version, valid ``CLEARANCE_VALID_S`` (120 s) on the ground's
monotonic clock from the click, and consumed by the take-off that gets ``ALLOW``. It
becomes invalid at landing or disarming, with a new charter version, a new autonomy mode
and at every restart of the ground station. The onboard computer gets its own copy
(``ClearanceConfirmation``) and counts its 120 s from receipt. The AI can never create one.
"""

from __future__ import annotations

import datetime as dt
from dataclasses import dataclass
from typing import TYPE_CHECKING, Literal
from uuid import UUID

from sd_safety.constants import CLEARANCE_VALID_S
from sd_schemas.api import Clearance, Mission
from sd_schemas.base import new_id
from sd_schemas.gate import ClearanceState
from sd_schemas.link import ClearanceConfirmation

if TYPE_CHECKING:
    from sd_ground.services.missions import MissionFlow

__all__ = ["ActiveClearance", "Clearances", "confirm"]


@dataclass(frozen=True, slots=True)
class ActiveClearance:
    clearance_id: str
    state: ClearanceState


class Clearances:
    """The active confirmation per vehicle (memory; the database keeps the record)."""

    def __init__(self) -> None:
        self._active: dict[str, ActiveClearance] = {}

    def confirm(
        self,
        *,
        clearance_id: str,
        origin: Literal["local", "lan", "remote"],
        mission_id: str,
        vehicle_id: str,
        charter_id: str,
        charter_version: int,
        now_ms: int,
    ) -> ActiveClearance:
        state = ClearanceState.model_validate(
            {
                "clearance_id": clearance_id,
                "origin": origin,
                "mission_id": mission_id,
                "vehicle_id": vehicle_id,
                "charter_id": charter_id,
                "charter_version": charter_version,
                "received_mono_ms": now_ms,
                "valid_for_ms": CLEARANCE_VALID_S * 1000,
                "consumed": False,
            }
        )
        active = ActiveClearance(clearance_id, state)
        self._active[vehicle_id] = active
        return active

    def get(self, vehicle_id: str) -> ActiveClearance | None:
        return self._active.get(vehicle_id)

    def consume(self, vehicle_id: str) -> ActiveClearance | None:
        active = self._active.pop(vehicle_id, None)
        return active

    def drop(self, vehicle_id: str) -> ActiveClearance | None:
        return self._active.pop(vehicle_id, None)

    def expired(self, now_ms: int) -> list[tuple[str, ActiveClearance]]:
        out = []
        for vid, active in list(self._active.items()):
            s = active.state
            if now_ms - s.received_mono_ms > s.valid_for_ms:
                out.append((vid, active))
                del self._active[vid]
        return out


async def confirm(flow: MissionFlow, mission: Mission, *, by: str, origin: str) -> Clearance:
    """``POST /missions/{id}/clearance`` (and the tick box on a take-off card): needs an
    approved charter and a vehicle that is not in the air. ``origin`` is the listener
    that took the click (``local`` or ``lan``), never a value from the request."""
    from sd_ground.services.missions import FlowError

    if origin not in ("local", "lan"):
        raise FlowError(403, "CLEARANCE_REMOTE", "Bekräftelsen måste ges på plats.")
    row = await flow.active_charter_row(mission)
    if row is None:
        raise FlowError(409, "CHARTER_NOT_APPROVED", "Det finns ingen godkänd uppdragsram.")
    vid, mid = mission.vehicle_id, str(mission.id)
    if flow.may_be_airborne(mission):
        raise FlowError(
            409,
            "AIRBORNE",
            "Drönaren måste stå avstängd på marken, med kontakt med flygkontrollern.",
        )
    clearance_id = str(new_id())
    now = flow.now()
    until = now + dt.timedelta(seconds=CLEARANCE_VALID_S)
    await flow.repo.add_clearance(
        clearance_id=clearance_id,
        mission_id=mid,
        vehicle_id=vid,
        charter_id=str(row.id),
        charter_version=row.version,
        by=by,
        now=now,
        valid_until=until,
    )
    active = flow.clearances.confirm(
        clearance_id=clearance_id,
        origin="lan" if origin == "lan" else "local",
        mission_id=mid,
        vehicle_id=vid,
        charter_id=str(row.id),
        charter_version=row.version,
        now_ms=flow.now_ms(),
    )
    if flow.link is not None:
        await flow.link.send_message(
            vid,
            ClearanceConfirmation(
                clearance_id=UUID(clearance_id),
                mission_id=mission.id,
                vehicle_id=vid,
                charter_id=row.id,
                charter_version=row.version,
                valid_for_ms=active.state.valid_for_ms,
                confirmed_by=by,
                origin=active.state.origin,
            ),
            mission_id=mid,
            ttl_ms=5000,
        )
    flow.publish(
        "mission",
        "clearance.updated",
        {"status": "active", "confirmed_by": by, "valid_until": until.isoformat()},
        vehicle_id=vid,
        mission_id=mid,
    )
    stored = await flow.repo.clearance(clearance_id)
    if stored is None:
        raise RuntimeError("row vanished after write")
    return stored
