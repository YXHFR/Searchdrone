"""``OnnxDetector`` on tiny hand-built ONNX graphs of both families: the decoding, the
letterbox and the checksum, with no trained weights."""

from __future__ import annotations

import json
import math
from collections.abc import Callable
from pathlib import Path

import numpy as np
import onnx
import pytest
from onnx import TensorProto, helper, numpy_helper

from sd_detect.api import Frame
from sd_detect.boxes import from_bbox, iou_matrix
from sd_detect.classes import ClassRegistry
from sd_detect.onnx import (
    MANIFEST_SCHEMA,
    ModelError,
    OnnxDetector,
    check_model,
    load_model,
    sha256_of,
)
from sd_detect.sources import MODEL_ENV, build_sources
from sd_detect.still import Still

INPUT_PX = 64
CELLS = sum((INPUT_PX // s) ** 2 for s in (8, 16, 32))  # 84
# one person 16 x 16 input px centred at (40, 24); the 128 x 96 picture is shrunk by 2
TRUTH = np.array([[64.0, 32.0, 96.0, 64.0]])


def _graph(outputs: dict[str, np.ndarray]) -> onnx.ModelProto:
    """A graph that ignores its picture (times zero) and gives fixed tensors, so the
    session still needs a real ``[1, 3, s, s]`` input."""
    x = helper.make_tensor_value_info("images", TensorProto.FLOAT, [1, 3, INPUT_PX, INPUT_PX])
    nodes = [
        helper.make_node("ReduceMean", ["images"], ["mean"], keepdims=0),
        helper.make_node("Mul", ["mean", "zero"], ["nothing"]),
    ]
    inits = [numpy_helper.from_array(np.zeros((), np.float32), "zero")]
    outs = []
    for name, value in outputs.items():
        inits.append(numpy_helper.from_array(value.astype(np.float32), f"{name}_c"))
        nodes.append(helper.make_node("Add", [f"{name}_c", "nothing"], [name]))
        outs.append(helper.make_tensor_value_info(name, TensorProto.FLOAT, list(value.shape)))
    graph = helper.make_graph(nodes, "tiny", [x], outs, inits)
    return helper.make_model(graph, opset_imports=[helper.make_opsetid("", 13)], ir_version=8)


def _yolox_raw(score: float = 0.9) -> np.ndarray:
    raw = np.zeros((1, CELLS, 6), np.float32)
    raw[0, :, 2:4] = -3.0  # everything else tiny and unsure
    cell = 3 * (INPUT_PX // 8) + 5  # stride 8, column 5, row 3 -> centre (40, 24)
    raw[0, cell] = [0.0, 0.0, math.log(2.0), math.log(2.0), score, score]
    return raw


def _damo_outputs(score: float = 0.81) -> dict[str, np.ndarray]:
    scores = np.zeros((1, CELLS, 2), np.float32)
    boxes = np.zeros((1, CELLS, 4), np.float32)
    boxes[0, :] = [0, 0, 2, 2]
    scores[0, 7] = [score, 0.0]
    boxes[0, 7] = [32, 16, 48, 32]
    return {"scores": scores, "boxes": boxes}


def _save(tmp_path: Path, model: onnx.ModelProto, name: str = "m.onnx") -> Path:
    path = tmp_path / name
    onnx.save(model, path)
    return path


def _frame(width: int = 128, height: int = 96) -> Frame:
    rgb = np.zeros((height, width, 3), np.uint8)
    return Frame(width=width, height=height, rgb=rgb.tobytes(), ts_mono_ms=0)


@pytest.mark.parametrize("family", ["yolox", "damoyolo"])
def test_both_families_find_the_person_where_it_is(tmp_path: Path, family: str) -> None:
    model = _graph({"output": _yolox_raw()} if family == "yolox" else _damo_outputs())
    path = _save(tmp_path, model)
    det = OnnxDetector(path, family, sha256_of(path), input_px=INPUT_PX)  # type: ignore[arg-type]
    found = det.detect(_frame())
    assert len(found) == 1
    assert found[0].label == "person" and found[0].source == "model"
    assert found[0].score == pytest.approx(0.81, abs=1e-4)
    assert iou_matrix(from_bbox([found[0].bbox]), TRUTH)[0, 0] >= 0.7


def test_low_scores_are_dropped(tmp_path: Path) -> None:
    path = _save(tmp_path, _graph({"output": _yolox_raw(score=0.1)}))  # 0.01 < 0.05
    det = OnnxDetector(path, "yolox", sha256_of(path), input_px=INPUT_PX)
    assert det.detect(_frame()) == []


def test_a_model_that_does_not_match_its_checksum_is_never_run(tmp_path: Path) -> None:
    path = _save(tmp_path, _graph({"output": _yolox_raw()}))
    with pytest.raises(ModelError, match="kontrollsumma"):
        OnnxDetector(path, "yolox", "0" * 64, input_px=INPUT_PX)


def test_letterbox_is_bgr_top_left_with_the_family_pad(tmp_path: Path) -> None:
    path = _save(tmp_path, _graph({"output": _yolox_raw()}))
    rgb = np.zeros((32, 64, 3), np.uint8)
    rgb[..., 0] = 200  # red
    frame = Frame(width=64, height=32, rgb=rgb.tobytes(), ts_mono_ms=0)
    tensor, r = OnnxDetector(path, "yolox", sha256_of(path), input_px=INPUT_PX).preprocess(frame)
    assert r == 1.0 and tensor.shape == (1, 3, INPUT_PX, INPUT_PX)
    assert tensor[0, 2, 0, 0] == 200 and tensor[0, 0, 0, 0] == 0  # R last: BGR
    assert tensor[0, 0, 40, 0] == 114  # below the picture: grey pad
    damo = _save(tmp_path, _graph(_damo_outputs()), "d.onnx")
    tensor, _ = OnnxDetector(damo, "damoyolo", sha256_of(damo), input_px=INPUT_PX).preprocess(frame)
    assert tensor[0, 0, 40, 0] == 0


def _manifest(tmp_path: Path, **over: object) -> Path:
    path = _save(tmp_path, _graph({"output": _yolox_raw()}), "person.onnx")
    body = {
        "schema": MANIFEST_SCHEMA,
        "onnx": "person.onnx",
        "family": "yolox",
        "sha256": sha256_of(path),
        "input_px": INPUT_PX,
        "classes": ["person"],
        "licence": "Apache-2.0",
    } | over
    manifest = tmp_path / "person.json"
    manifest.write_text(json.dumps(body), encoding="utf-8")
    return manifest


def test_the_model_source_tiles_a_still(tmp_path: Path) -> None:
    source = load_model(_manifest(tmp_path))
    assert source.tiled.tile_px == INPUT_PX
    still = Still(frame=_frame())
    found = source.detect_still(still)
    assert found and all(d.label == "person" for d in found)
    assert source.tiled.last_tiles > 1


@pytest.mark.parametrize(
    ("over", "why"),
    [
        ({"onnx": "../person.onnx"}, "filnamn"),
        ({"schema": "other"}, "manifest"),
        ({"classes": ["car"]}, "person"),
        ({"sha256": "1" * 64}, "kontrollsumma"),
        ({"family": "yolo9"}, "modellfamilj"),
        ({"licence": "CC-BY-NC-4.0"}, "licens"),
        ({"onnx": "gone.onnx"}, "finns inte"),
        ({"sha256": None}, "64 hextecken"),
    ],
)
def test_a_wrong_manifest_is_refused(tmp_path: Path, over: dict[str, object], why: str) -> None:
    with pytest.raises(ModelError, match=why):
        load_model(_manifest(tmp_path, **over))


@pytest.mark.parametrize("check", [check_model, load_model])
@pytest.mark.parametrize(
    ("over", "why"),
    [
        ({"sha256": "1" * 64}, "kontrollsumma"),
        ({"onnx": "gone.onnx"}, "finns inte"),
        ({"schema": "other"}, "manifest"),
        ({"input_px": 100}, "delbart med 32"),
        ({"input_px": "640"}, "heltal"),
        ({"tile_px": 16}, "minst 32"),
    ],
)
def test_check_model_refuses_what_load_model_refuses(
    tmp_path: Path, over: dict[str, object], why: str, check: Callable[[Path], object]
) -> None:
    with pytest.raises(ModelError, match=why):
        check(_manifest(tmp_path, **over))


def test_check_model_gives_the_checked_file(tmp_path: Path) -> None:
    assert check_model(_manifest(tmp_path)) == tmp_path / "person.onnx"
    with pytest.raises(ModelError, match="Kan inte läsa"):
        check_model(tmp_path / "missing.json")


def test_a_model_over_the_size_cap_is_refused(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    import sd_detect.onnx as onnx_mod

    manifest = _manifest(tmp_path)
    monkeypatch.setattr(onnx_mod, "MAX_MODEL_BYTES", 100)
    with pytest.raises(ModelError, match="orimligt stor"):
        load_model(manifest)


def test_bytes_that_match_but_are_not_onnx_are_refused(tmp_path: Path) -> None:
    import hashlib

    manifest = _manifest(tmp_path)
    junk = b"not an onnx graph"
    (tmp_path / "person.onnx").write_bytes(junk)
    body = json.loads(manifest.read_text(encoding="utf-8"))
    body["sha256"] = hashlib.sha256(junk).hexdigest()
    manifest.write_text(json.dumps(body), encoding="utf-8")
    with pytest.raises(Exception):  # noqa: B017 - ONNX Runtime's own error type
        load_model(manifest)


def test_onnx_runtime_gets_few_threads_and_no_busy_wait(tmp_path: Path) -> None:
    path = _save(tmp_path, _graph({"output": _yolox_raw()}))
    det = OnnxDetector(path, "yolox", sha256_of(path), input_px=INPUT_PX)
    options = det.session.get_session_options()
    assert options.intra_op_num_threads == 2 and options.inter_op_num_threads == 1
    assert options.get_session_config_entry("session.intra_op.allow_spinning") == "0"


def test_the_register_builds_the_model_only_when_one_is_chosen(tmp_path: Path) -> None:
    registry = ClassRegistry.load()
    assert "model" in build_sources(registry, env={}).skipped
    built = build_sources(registry, env={MODEL_ENV: str(_manifest(tmp_path))})
    assert built.sources["model"].source == "model"
    with pytest.raises(ModelError):
        build_sources(registry, env={MODEL_ENV: str(tmp_path / "missing.json")})
