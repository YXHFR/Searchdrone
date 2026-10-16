"""The gate is a pure function: no clock, no I/O, no randomness, no changed inputs."""

from __future__ import annotations

import ast
import builtins
import socket
import time
from pathlib import Path

import pytest
from gate_fixtures import ctx, goto, ground_ctx, req

from sd_safety import evaluate

SRC = Path(__file__).resolve().parents[2] / "src" / "sd_safety"
ALLOWED_IMPORTS = {
    "__future__",
    "collections.abc",
    "dataclasses",
    "importlib.metadata",
    "itertools",
    "math",
    "types",
    "typing",
    "pydantic",
}


def test_evaluate_pure(monkeypatch: pytest.MonkeyPatch) -> None:
    cases = [
        (req(), ctx()),
        (req(params=goto(0, 310, alt_m=200)), ctx()),
        (req("takeoff", {"alt_m": 40}), ground_ctx()),
    ]
    first = [evaluate(r, c) for r, c in cases]
    snapshots = [(r.model_dump(), c.model_dump()) for r, c in cases]

    def forbidden(*_a: object, **_k: object) -> None:
        raise AssertionError("the gate must not read clocks or do I/O")

    for name in ("time", "monotonic", "monotonic_ns", "perf_counter", "time_ns"):
        monkeypatch.setattr(time, name, forbidden)
    monkeypatch.setattr(builtins, "open", forbidden)
    monkeypatch.setattr(socket, "socket", forbidden)
    for _ in range(3):
        assert [evaluate(r, c) for r, c in cases] == first
    assert [(r.model_dump(), c.model_dump()) for r, c in cases] == snapshots


def test_sources_import_nothing_impure() -> None:
    for path in SRC.glob("*.py"):
        tree = ast.parse(path.read_text())
        for node in ast.walk(tree):
            if isinstance(node, ast.Import):
                names = [a.name for a in node.names]
            elif isinstance(node, ast.ImportFrom):
                names = [node.module or ""]
            else:
                continue
            for name in names:
                assert name in ALLOWED_IMPORTS or name.split(".")[0] in {
                    "sd_schemas",
                    "sd_safety",
                }, f"{path.name} imports {name}"
