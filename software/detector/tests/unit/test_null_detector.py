"""B1-14: the detector interface and the null detector."""

from __future__ import annotations

import ast
import re
from pathlib import Path

import pytest
from pydantic import ValidationError

import sd_detect
from sd_detect import BBox, Detection, Detector, Frame, NullDetector

SRC = Path(sd_detect.__file__).parent


def _frame(w: int = 4, h: int = 3) -> Frame:
    return Frame(width=w, height=h, rgb=bytes(w * h * 3), ts_mono_ms=0)


def test_null_detector() -> None:
    det: Detector = NullDetector()  # pyright checks that it satisfies the protocol
    assert isinstance(det, Detector)
    assert det.detect(_frame()) == []
    assert NullDetector.name == "null"
    assert NullDetector.license == "Apache-2.0"


def test_null_detector_says_it_is_not_built() -> None:
    assert NullDetector.built is False
    assert "inte byggd" in NullDetector.reason
    assert "etapp 2" in NullDetector.reason


def test_detection_schema() -> None:
    assert set(Detection.model_fields) == {"label", "score", "bbox", "embedding", "source"}
    assert Detection.model_fields["source"].default is None
    assert Detection.model_fields["embedding"].is_required() is False
    assert Detection.model_fields["embedding"].default is None
    assert set(BBox.model_fields) == {"x", "y", "w", "h"}
    d = Detection(label="person", score=0.5, bbox=BBox(x=10, y=20, w=30, h=40))
    assert d.embedding is None
    assert Detection(label="person", score=1, bbox=d.bbox, embedding=[0.1, 0.2]).embedding


def test_no_face_or_identity_names() -> None:
    banned = re.compile(r"face|identity|person_id", re.IGNORECASE)
    for path in SRC.rglob("*.py"):
        tree = ast.parse(path.read_text(encoding="utf-8"))
        for node in ast.walk(tree):
            names: list[str] = []
            if isinstance(node, ast.ClassDef | ast.FunctionDef | ast.AsyncFunctionDef):
                names.append(node.name)
            elif isinstance(node, ast.Name):
                names.append(node.id)
            elif isinstance(node, ast.Attribute):
                names.append(node.attr)
            elif isinstance(node, ast.arg):
                names.append(node.arg)
            elif isinstance(node, ast.alias):
                names.append(node.asname or node.name)
            assert not [n for n in names if banned.search(n)], (path.name, names)
    for model in (Frame, BBox, Detection):
        assert not [f for f in model.model_fields if banned.search(f)], model


@pytest.mark.parametrize("label", ["Person", "", "x", "red block", "a" * 40])
def test_label_is_a_class_id(label: str) -> None:
    with pytest.raises(ValidationError, match="label"):
        Detection(label=label, score=0.5, bbox=BBox(x=0, y=0, w=1, h=1))


@pytest.mark.parametrize("label", ["face", "person_face", "licence_plate", "identity"])
def test_never_identities(label: str) -> None:
    with pytest.raises(ValidationError, match="identities"):
        Detection(label=label, score=0.5, bbox=BBox(x=0, y=0, w=1, h=1))


@pytest.mark.parametrize(
    "kwargs",
    [
        {"score": 1.1},
        {"score": -0.1},
        {"score": float("nan")},
        {"bbox": {"x": -1, "y": 0, "w": 1, "h": 1}},
        {"bbox": {"x": 0, "y": 0, "w": 0, "h": 1}},
        {"embedding": [float("inf")]},
        {"embedding": [0.0] * 4097},
        {"extra": 1},
    ],
)
def test_detection_rejects_bad_values(kwargs: dict[str, object]) -> None:
    base: dict[str, object] = {
        "label": "person",
        "score": 0.5,
        "bbox": {"x": 0, "y": 0, "w": 1, "h": 1},
    }
    with pytest.raises(ValidationError):
        Detection.model_validate(base | kwargs)


def test_frame_size_must_match() -> None:
    with pytest.raises(ValidationError, match="width"):
        Frame(width=2, height=2, rgb=bytes(11), ts_mono_ms=0)
    with pytest.raises(ValidationError):
        Frame(width=0, height=2, rgb=b"", ts_mono_ms=0)
