"""tools/gen_schemas.py: the export list, determinism and the TypeScript output."""

from __future__ import annotations

import shutil
import subprocess

import gen_schemas
import pytest
from pydantic import BaseModel

import sd_schemas

PNPM = shutil.which("pnpm")
needs_pnpm = pytest.mark.skipif(PNPM is None, reason="pnpm is not installed")


def test_export_list_is_every_model_and_enum_in_all() -> None:
    names = {t.__name__ for t in gen_schemas.exported_types()}
    assert "SafetyLimits" in names
    assert "MissionState" in names  # an enum
    assert "SkillSpec" not in names  # internal, carries Python types
    assert "new_id" not in names


def test_schema_is_deterministic() -> None:
    assert gen_schemas.dump(gen_schemas.build_schema()) == gen_schemas.dump(
        gen_schemas.build_schema()
    )


def test_new_model_in_all_shows_up(monkeypatch: pytest.MonkeyPatch) -> None:
    class BrandNewThing(BaseModel):
        x: int

    monkeypatch.setattr(sd_schemas, "BrandNewThing", BrandNewThing, raising=False)
    monkeypatch.setattr(sd_schemas, "__all__", [*sd_schemas.__all__, "BrandNewThing"])
    assert "BrandNewThing" in gen_schemas.build_schema()["$defs"]


def test_no_field_titles_and_no_ref_siblings() -> None:
    def walk(node: object, top: bool) -> None:
        if isinstance(node, dict):
            if not top:
                assert not isinstance(node.get("title"), str), node
            if "$ref" in node:
                assert "default" not in node, node
            for v in node.values():
                walk(v, False)
        elif isinstance(node, list):
            for v in node:
                walk(v, False)

    for body in gen_schemas.ts_input(gen_schemas.build_schema())["$defs"].values():
        walk(body, True)


def test_committed_schema_keeps_defaults_next_to_refs() -> None:
    props = gen_schemas.build_schema()["$defs"]["SurveyResumePoint"]["properties"]
    assert props["kind"]["default"] == "survey"


def test_committed_schema_matches_the_models() -> None:
    expected = gen_schemas.dump(gen_schemas.build_schema())
    assert gen_schemas.SCHEMA_FILE.read_text(encoding="utf-8") == expected, "kör 'just gen'"


@needs_pnpm
def test_typescript_has_every_type_once_and_compiles() -> None:
    text = gen_schemas.TS_FILE.read_text(encoding="utf-8")
    assert text.startswith(gen_schemas.BANNER)
    for name in gen_schemas.build_schema()["$defs"]:
        declared = text.count(f"export interface {name} ") + text.count(f"export type {name} ")
        assert declared == 1, name
    assert PNPM is not None
    res = subprocess.run(
        [PNPM, "--filter", "@searchdrone/schemas", "exec", "tsc", "--noEmit", "-p", "."],
        cwd=gen_schemas.ROOT,
        capture_output=True,
        text=True,
        check=False,
    )
    assert res.returncode == 0, res.stdout + res.stderr
