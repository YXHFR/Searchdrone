"""E5 (B2-01): no face recognition and no identity fields in the detector or the schemas.

Covers ``sd_detect`` and the stage 2 schemas ``sd_schemas.search`` and ``sd_schemas.link``:
no identifier (class, function, argument, name or attribute) and no model field may
match the banned words. Comments and strings are not checked.
"""

from __future__ import annotations

import ast
import inspect
import re
from pathlib import Path

from pydantic import BaseModel

import sd_detect
from sd_detect import api
from sd_schemas import link, search

# E5: no face recognition and nothing that describes who someone is.
BANNED = re.compile(r"face|photo|ethnic|religion|identity|person_id|reid", re.IGNORECASE)


def _names(path: Path) -> list[str]:
    """Every identifier a source file defines or uses (not its comments or strings)."""
    out: list[str] = []
    for node in ast.walk(ast.parse(path.read_text(encoding="utf-8"))):
        if isinstance(node, ast.ClassDef | ast.FunctionDef | ast.AsyncFunctionDef):
            out.append(node.name)
        elif isinstance(node, ast.Name):
            out.append(node.id)
        elif isinstance(node, ast.Attribute):
            out.append(node.attr)
        elif isinstance(node, ast.arg):
            out.append(node.arg)
        elif isinstance(node, ast.alias):
            out.append(node.asname or node.name)
    return out


def _models(module: object) -> list[type[BaseModel]]:
    return [
        obj
        for _, obj in inspect.getmembers(module, inspect.isclass)
        if issubclass(obj, BaseModel) and obj.__module__ == getattr(module, "__name__", "")
    ]


def test_no_identity_fields() -> None:
    detect_files = sorted(Path(sd_detect.__file__).parent.rglob("*.py"))
    files = [*detect_files, Path(search.__file__), Path(link.__file__)]
    assert detect_files
    for path in files:
        assert not [n for n in _names(path) if BANNED.search(n)], path.name
    for module in (api, search, link):
        models = _models(module)
        assert models
        for model in models:
            assert not [f for f in model.model_fields if BANNED.search(f)], model.__name__
