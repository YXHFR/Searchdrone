"""OpenAPI of both APIs equals the committed snapshots (04 T24). Update with ``just gen``."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from sd_ground.app import openapi_documents

pytestmark = pytest.mark.contract
SNAPSHOTS = Path(__file__).resolve().parent / "snapshots"


@pytest.mark.parametrize("name", ["panel", "commander"])
def test_openapi_matches_snapshot(name: str) -> None:
    current = json.loads(json.dumps(openapi_documents()[name]))
    stored = json.loads((SNAPSHOTS / f"{name}.openapi.json").read_text(encoding="utf-8"))
    assert current == stored, f"OpenAPI för {name} har ändrats. Kör: just gen"


def test_commander_has_no_operator_actions() -> None:
    paths = " ".join(openapi_documents()["commander"]["paths"])
    for word in ("approval", "clearance", "autonomy", "limits", "actions", "annotations", "/ai"):
        assert word not in paths, word
