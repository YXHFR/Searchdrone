"""An ``OnboardEvent`` from the drone is stored, and the ones the pilot must see reach the
panel: ``executor.perception_off`` tells them person detection is off on board, and
``executor.perception_on`` that it works again."""

from __future__ import annotations

from types import SimpleNamespace
from typing import Any

from sd_ground.services.missions import MissionFlow
from sd_schemas.link import OnboardEvent

VID = "sd-test-1"


class _Flow:
    def __init__(self) -> None:
        self.stored: list[str] = []
        self.published: list[tuple[str, dict[str, Any]]] = []

        async def open_mission(vehicle_id: str) -> None:
            return None

        self.repo = SimpleNamespace(open_mission=open_mission)
        self.g = SimpleNamespace(add_event=lambda kind, **_: self.stored.append(kind))

    def publish(self, topic: str, kind: str, data: dict[str, Any], **_: Any) -> None:
        self.published.append((kind, data))


async def _receive(event: OnboardEvent) -> _Flow:
    flow = _Flow()
    await MissionFlow.on_inbound(flow, VID, SimpleNamespace(body=event))  # type: ignore[arg-type]
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
