"""An ``OnboardEvent`` from the drone is stored, and the ones the pilot must see reach the
panel: ``executor.perception_off`` tells them person detection is off on board, and
``executor.perception_on`` that it works again. An event is stored under the mission it
names (it may come after that mission closed), else the vehicle's open mission."""

from __future__ import annotations

import uuid
from types import SimpleNamespace
from typing import Any

from sd_ground.services.missions import MissionFlow
from sd_schemas.link import OnboardEvent

VID = "sd-test-1"


class _Flow:
    def __init__(self, missions: dict[str, Any] | None = None, open_id: str | None = None) -> None:
        self.stored: list[str] = []
        self.stored_in: list[str | None] = []
        self.published: list[tuple[str, dict[str, Any]]] = []
        known = missions or {}

        async def open_mission(vehicle_id: str) -> Any:
            return None if open_id is None else known[open_id]

        async def mission(mission_id: str) -> Any:
            return known.get(mission_id)

        def add_event(kind: str, *, mission_id: str | None, **_: Any) -> None:
            self.stored.append(kind)
            self.stored_in.append(mission_id)

        self.repo = SimpleNamespace(open_mission=open_mission, mission=mission)
        self.g = SimpleNamespace(add_event=add_event)

    def publish(self, topic: str, kind: str, data: dict[str, Any], **_: Any) -> None:
        self.published.append((kind, data))


async def _receive(event: OnboardEvent, mission_id: uuid.UUID | None = None, **kw: Any) -> _Flow:
    flow = _Flow(**kw)
    inbound = SimpleNamespace(body=event, envelope=SimpleNamespace(mission_id=mission_id))
    await MissionFlow.on_inbound(flow, VID, inbound)  # type: ignore[arg-type]
    return flow


async def test_perception_off_reaches_the_pilot() -> None:
    data = {"reason": "sources_failed", "error": "ModelError"}
    flow = await _receive(
        OnboardEvent(type="executor.perception_off", severity="warning", data=data)
    )
    assert flow.stored == ["executor.perception_off"]
    assert flow.published == [("executor.perception_off", data)]


async def test_perception_back_on_reaches_the_pilot() -> None:
    flow = await _receive(OnboardEvent(type="executor.perception_on", severity="info"))
    assert flow.published == [("executor.perception_on", {})]


async def test_other_onboard_events_are_only_stored() -> None:
    flow = await _receive(OnboardEvent(type="executor.something_else", severity="info"))
    assert flow.stored == ["executor.something_else"] and flow.published == []


def _mission(vehicle_id: str = VID) -> Any:
    return SimpleNamespace(id=uuid.uuid4(), vehicle_id=vehicle_id)


async def test_an_event_is_stored_in_the_mission_it_names() -> None:
    """A find hold held back by a lost link comes after its mission closed: it is stored
    in that mission, not in none (or in a newer one)."""
    closed, newer = _mission(), _mission()
    missions = {str(closed.id): closed, str(newer.id): newer}
    event = OnboardEvent(type="executor.find_hold_started", severity="warning")
    flow = await _receive(event, closed.id, missions=missions)
    assert flow.stored_in == [str(closed.id)]
    flow = await _receive(event, closed.id, missions=missions, open_id=str(newer.id))
    assert flow.stored_in == [str(closed.id)]


async def test_an_event_naming_no_mission_of_its_vehicle_goes_to_the_open_one() -> None:
    other, unknown, open_one = _mission("sd-test-2"), uuid.uuid4(), _mission()
    missions = {str(other.id): other, str(open_one.id): open_one}
    event = OnboardEvent(type="executor.find_hold_started", severity="warning")
    for named in (None, other.id, unknown):
        flow = await _receive(event, named, missions=missions, open_id=str(open_one.id))
        assert flow.stored_in == [str(open_one.id)]
    flow = await _receive(event, other.id, missions=missions)
    assert flow.stored_in == [None]
