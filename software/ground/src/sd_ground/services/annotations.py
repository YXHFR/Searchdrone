"""Map annotations: the pilot's pins and drawn areas, and the AI's marks.

The pilot adds pins (``P1``, ...) and areas; a drawn area can become a charter's search
area (``CharterProposal.area_annotation_id``). The AI's ``mark_location`` only ever adds
a pin with the next free label ``M<n>`` (inside the charter, checked by the gate); its
own label travels as a note. Nothing here moves the vehicle.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

from sd_ground.services.missions import FlowError
from sd_schemas.api import Annotation, AnnotationCreate, Mission

if TYPE_CHECKING:
    from sd_ground.services.missions import MissionFlow

__all__ = ["add", "delete", "mark"]


def _model(row: dict[str, Any]) -> Annotation:
    return Annotation.model_validate(
        {k: v for k, v in row.items() if k not in ("vehicle_id", "deleted_at")}
    )


def _publish(
    flow: MissionFlow,
    mission: Mission,
    ann: Annotation,
    type_: str,
    extra: dict[str, Any] | None = None,
) -> None:
    flow.publish(
        "mission",
        type_,
        {**ann.model_dump(mode="json"), **(extra or {})},
        vehicle_id=mission.vehicle_id,
        mission_id=str(mission.id),
    )


async def add(flow: MissionFlow, mission_id: str, body: AnnotationCreate, by: str) -> Annotation:
    mission = await flow.open_mission_of(mission_id)
    new = await flow.repo.add_annotation(
        mission_id=mission_id,
        vehicle_id=mission.vehicle_id,
        kind=body.kind,
        label=body.label,
        geometry=dict(body.geometry),
        created_by=by,
        now=flow.now(),
    )
    if new is None:
        raise FlowError(409, "LABEL_TAKEN", "Namnet används redan på kartan.")
    row = await flow.repo.annotation(new)
    if row is None:
        raise RuntimeError("row vanished after write")
    ann = _model(row)
    _publish(flow, mission, ann, "annotation.created")
    return ann


async def mark(
    flow: MissionFlow, mission: Mission, *, lat: float, lon: float, note: str, by: str
) -> Annotation | None:
    new = await flow.repo.add_annotation(
        mission_id=str(mission.id),
        vehicle_id=mission.vehicle_id,
        kind="pin",
        label=None,
        geometry={"type": "Point", "coordinates": [lon, lat]},
        created_by=by,
        now=flow.now(),
    )
    if new is None:
        return None
    row = await flow.repo.annotation(new)
    if row is None:
        return None
    ann = _model(row)
    _publish(flow, mission, ann, "annotation.created", {"note": note[:40]})
    return ann


async def delete(flow: MissionFlow, annotation_id: str) -> None:
    row = await flow.repo.annotation(annotation_id)
    if row is None or row["deleted_at"] is not None:
        raise FlowError(404, "ANNOTATION_NOT_FOUND", "Markeringen finns inte.")
    mission = await flow.open_mission_of(str(row["mission_id"]))
    await flow.repo.delete_annotation(annotation_id, flow.now())
    flow.publish(
        "mission",
        "annotation.deleted",
        {"id": annotation_id},
        vehicle_id=mission.vehicle_id,
        mission_id=str(mission.id),
    )
