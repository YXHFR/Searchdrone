"""Golden example files validate against the Pydantic models and the generated JSON Schema."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import jsonschema
import pytest
from pydantic import ValidationError

import sd_schemas
from sd_schemas.link import KIND_MODELS, Envelope

HERE = Path(__file__).resolve().parent
GOLDEN = HERE.parent / "golden"
SCHEMA_FILE = HERE.parents[1] / "generated" / "sd_schemas.schema.json"
SDLINK = sorted((GOLDEN / "sdlink").glob("*.json"))
API = sorted((GOLDEN / "api").glob("*.json"))


def _load(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


@pytest.fixture(scope="module")
def schema() -> dict[str, Any]:
    return _load(SCHEMA_FILE)


def _validator(schema: dict[str, Any], name: str) -> jsonschema.Draft202012Validator:
    return jsonschema.Draft202012Validator({"$ref": f"#/$defs/{name}", "$defs": schema["$defs"]})


def test_golden_files_exist() -> None:
    assert len(SDLINK) >= 3
    assert {p.stem for p in API} >= {
        "Mission",
        "Charter",
        "Approval",
        "Command",
        "ChatMessage",
        "ActionOutcome",
        "Situation",
        "GateDecision",
    }


@pytest.mark.parametrize("path", SDLINK, ids=lambda p: p.stem)
def test_sdlink_golden(path: Path, schema: dict[str, Any]) -> None:
    data = _load(path)
    env = Envelope.model_validate(data)
    body = env.parse_body()
    assert type(body).__name__ == env.kind
    assert env.vehicle_id == "sim-01"
    _validator(schema, "Envelope").validate(data)
    _validator(schema, env.kind).validate(data["body"])
    assert env.kind in KIND_MODELS


@pytest.mark.parametrize("path", API, ids=lambda p: p.stem)
def test_api_golden(path: Path, schema: dict[str, Any]) -> None:
    data = _load(path)
    model = getattr(sd_schemas, path.stem)
    item = model.model_validate(data)
    assert item.model_dump(mode="json") == data
    _validator(schema, path.stem).validate(data)


def test_schema_rejects_unknown_kind_body() -> None:
    data = _load(SDLINK[0])
    data["kind"] = "SkillOrder"  # an old name that no longer exists
    with pytest.raises(ValidationError, match="unknown kind"):
        Envelope.model_validate(data)
