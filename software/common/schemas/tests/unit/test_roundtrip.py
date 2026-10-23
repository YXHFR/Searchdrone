"""Every exported model survives model -> JSON -> model, and ids and vehicles are typed."""

from __future__ import annotations

import typing
import uuid

import pytest
from pydantic import BaseModel

import sd_schemas
from sd_schemas_examples import examples

EXPORTED_MODELS = sorted(
    (
        obj
        for name in sd_schemas.__all__
        if isinstance(obj := getattr(sd_schemas, name), type) and issubclass(obj, BaseModel)
    ),
    key=lambda m: m.__name__,
)
EXAMPLES = examples()


def test_every_exported_model_has_an_example() -> None:
    missing = [m.__name__ for m in EXPORTED_MODELS if m not in EXAMPLES]
    assert missing == []


@pytest.mark.parametrize("model", EXPORTED_MODELS, ids=lambda m: m.__name__)
def test_roundtrip(model: type[BaseModel]) -> None:
    item = EXAMPLES[model]
    again = model.model_validate_json(item.model_dump_json())
    assert again == item
    assert again.model_dump_json() == item.model_dump_json()


@pytest.mark.parametrize("model", EXPORTED_MODELS, ids=lambda m: m.__name__)
def test_unknown_fields_are_rejected(model: type[BaseModel]) -> None:
    data = EXAMPLES[model].model_dump(mode="json")
    data["no_such_field"] = 1
    with pytest.raises(ValueError, match="no_such_field"):
        model.model_validate(data)


def _contains(annotation: object, target: type) -> bool:
    if annotation is target:
        return True
    return any(_contains(a, target) for a in typing.get_args(annotation))


# Fields that end in "id" but are labels, not UUIDs.
LABEL_IDS = {
    ("MapPin", "id"),  # the pin label "P1"
    ("Vehicle", "id"),  # the vehicle id "sim-01"
    ("Sector", "sector_id"),
    ("SurveySectorParams", "sector_id"),
    ("SectorScore", "sector_id"),
    ("PanelConfig", "default_vehicle_id"),
    ("DetectorInfo", "profile_id"),  # the detector profile "det_<model>_vN"
}
# Field names that are references of another kind: "operator:<alias>", a model name,
# or the onboard boot counter.
NON_UUID_ID_NAMES = {"operator_id", "model_id", "boot_id"}


@pytest.mark.parametrize("model", EXPORTED_MODELS, ids=lambda m: m.__name__)
def test_id_fields_are_uuids(model: type[BaseModel]) -> None:
    for name, field in model.model_fields.items():
        if not (name == "id" or name.endswith("_id")) or name == "vehicle_id":
            continue
        if (model.__name__, name) in LABEL_IDS or name in NON_UUID_ID_NAMES:
            continue
        assert _contains(field.annotation, uuid.UUID), f"{model.__name__}.{name}"


# Every model that is about one vehicle carries vehicle_id (or is the vehicle itself).
VEHICLE_MODELS = [
    "Envelope",
    "MissionCharter",
    "GateRequest",
    "ClearanceState",
    "ClearanceConfirmation",
    "OperatorLatch",
    "VehicleState",
    "Mission",
    "MissionSummary",
    "Approval",
    "Command",
    "Clearance",
    "Event",
    "VehicleLive",
    "ActionOutcome",
    "ActionDispatchedVehicle",
    "Situation",
]


@pytest.mark.parametrize("name", VEHICLE_MODELS)
def test_vehicle_models_carry_vehicle_id(name: str) -> None:
    model = getattr(sd_schemas, name)
    assert "vehicle_id" in model.model_fields


def test_no_forbidden_vehicle_names() -> None:
    for model in EXPORTED_MODELS:
        for name in model.model_fields:
            assert name not in {"drone_id", "uav_id"}, f"{model.__name__}.{name}"


def test_models_are_frozen() -> None:
    item = EXAMPLES[sd_schemas.SafetyLimits]
    with pytest.raises(ValueError, match="frozen"):
        item.max_alt_agl_m = 200  # type: ignore[misc]


def test_new_id_is_uuid7_and_ordered() -> None:
    a, b = sd_schemas.new_id(), sd_schemas.new_id()
    assert isinstance(a, uuid.UUID)
    assert a.version == 7
    assert a < b
