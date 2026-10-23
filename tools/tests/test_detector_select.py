"""Choosing and loading the trained detector: ``pick_detector``, ``model_detector`` in
``eval_detector``, ``tools/train/fetch_weights.py`` and ``datadir`` (no real model)."""

from __future__ import annotations

import hashlib
import importlib.util
import json
import re
from pathlib import Path
from types import ModuleType
from typing import Any

import numpy as np
import pytest
from bench import datadir
from bench import eval_detector as ev
from bench import pick_detector as pd
from bench.common import BenchError
from bench.datasets import Image
from PIL import Image as PILImage

from sd_detect.onnx import MANIFEST_SCHEMA, MAX_MODEL_BYTES, load_model

TRAIN = Path(__file__).resolve().parents[1] / "train"


def _fetch_weights() -> ModuleType:
    spec = importlib.util.spec_from_file_location("fetch_weights", TRAIN / "fetch_weights.py")
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


fw = _fetch_weights()


# --- pick_detector ------------------------------------------------------------------


def _body(
    model: str, gt32: float, mid: float, p50: float, source: str = "splits/v1 val (heridal)"
) -> dict[str, Any]:
    return {
        "schema": "bench.detector/v1",
        "model": model,
        "source": source,
        "n_pictures": 221,
        "recall": {"gt32": {"recall": gt32}, "16to32": {"recall": mid}},
        "latency_ms": {"p50": p50},
    }


def test_the_highest_recall_over_32_px_wins() -> None:
    chosen = pd.pick([_body("a", 0.80, 0.9, 10), _body("b", 0.90, 0.1, 900)])
    assert chosen["model"] == "b"


def test_within_two_points_the_mid_size_recall_decides() -> None:
    chosen = pd.pick([_body("a", 0.90, 0.50, 10), _body("b", 0.885, 0.60, 900)])
    assert chosen["model"] == "b"


def test_still_level_the_faster_one_wins() -> None:
    chosen = pd.pick([_body("a", 0.90, 0.60, 900), _body("b", 0.89, 0.59, 400)])
    assert chosen["model"] == "b"


def test_nothing_to_pick_from_is_an_error() -> None:
    with pytest.raises(BenchError):
        pd.pick([])


def _write(folder: Path, body: dict[str, Any]) -> Path:
    path = folder / f"det_{body['model']}_val.json"
    path.write_text(json.dumps(body), encoding="utf-8")
    return path


def test_main_skips_built_ins_and_other_parts(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    files = [
        _write(tmp_path, _body("person_a", 0.80, 0.5, 10)),
        _write(tmp_path, _body("facit", 1.0, 1.0, 1)),
        _write(tmp_path, _body("person_t", 0.99, 0.9, 1, source="splits/v1 test (heridal)")),
    ]
    arg = "results=" + ",".join(str(f) for f in files)
    assert pd.main([arg, f"data={tmp_path}"]) == 0
    out = capsys.readouterr().out
    assert "Vald: person_a." in out
    assert f"SD_DETECTOR_MODEL={tmp_path / 'models' / 'detector' / 'person_a.json'}" in out


def test_results_from_different_pictures_are_never_ranked(tmp_path: Path) -> None:
    files = [
        _write(tmp_path, _body("person_a", 0.80, 0.5, 10)),
        _write(
            tmp_path, _body("person_b", 0.95, 0.5, 10, source="splits/v1 val (heridal) limit=10")
        ),
    ]
    arg = "results=" + ",".join(str(f) for f in files)
    assert pd.main([arg, f"data={tmp_path}"]) == 2


def test_an_unreadable_result_is_a_clear_error(tmp_path: Path) -> None:
    bad = tmp_path / "det_person_x_val.json"
    bad.write_text('{"schema": "bench.detector/v1"}', encoding="utf-8")
    gone = tmp_path / "det_gone_val.json"
    assert pd.main([f"results={bad}", f"data={tmp_path}"]) == 2
    assert pd.main([f"results={gone}", f"data={tmp_path}"]) == 2


def test_a_quick_look_alone_is_picked_with_a_warning(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    body = _body("person_a", 0.8, 0.5, 10, source="splits/v1 val (heridal) limit=4")
    assert pd.main([f"results={_write(tmp_path, body)}", f"data={tmp_path}"]) == 0
    assert "limit=" in capsys.readouterr().err


# --- a trained model in the bench, and the manifest detector-fetch writes -----------


def _tiny_yolox(path: Path, input_px: int = 64) -> None:
    """A YOLOX-shaped ONNX graph that always finds one person in the middle of its input."""
    import onnx
    from onnx import TensorProto, helper, numpy_helper

    cells = sum((input_px // s) ** 2 for s in (8, 16, 32))
    raw = np.zeros((1, cells, 6), np.float32)
    raw[0, :, 4:] = 0.0001  # objectness and class score (already after the sigmoid)
    cell = 3 * (input_px // 8) + 3  # stride 8, the cell at (24, 24)
    raw[0, cell] = [0.5, 0.5, np.log(2.0), np.log(2.0), 0.9, 0.9]
    x = helper.make_tensor_value_info("images", TensorProto.FLOAT, [1, 3, input_px, input_px])
    nodes = [
        helper.make_node("ReduceMean", ["images"], ["mean"], keepdims=0),
        helper.make_node("Mul", ["mean", "zero"], ["nothing"]),
        helper.make_node("Add", ["raw", "nothing"], ["output"]),
    ]
    inits = [
        numpy_helper.from_array(np.zeros((), np.float32), "zero"),
        numpy_helper.from_array(raw, "raw"),
    ]
    out = helper.make_tensor_value_info("output", TensorProto.FLOAT, list(raw.shape))
    graph = helper.make_graph(nodes, "tiny", [x], [out], inits)
    model = helper.make_model(graph, opset_imports=[helper.make_opsetid("", 13)], ir_version=8)
    onnx.save(model, path)


def _fetched(data: Path) -> Path:
    """What ``just detector-fetch`` leaves for a ready detector: the file and its
    manifest from ``manifest_for``."""
    folder = data / "models" / "detector"
    folder.mkdir(parents=True)
    onnx_path = folder / "coco_tiny.onnx"
    _tiny_yolox(onnx_path)
    row = {
        "name": "coco_tiny",
        "kind": "detector",
        "file": "detector/coco_tiny.onnx",
        "url": "https://example.invalid/coco_tiny.onnx",
        "size_bytes": onnx_path.stat().st_size,
        "sha256": hashlib.sha256(onnx_path.read_bytes()).hexdigest(),
        "licence": "Apache-2.0",
        "attribution": "test",
        "family": "yolox",
        "input_px": 64,
    }
    manifest = folder / "coco_tiny.json"
    manifest.write_text(json.dumps(fw.manifest_for(row)), encoding="utf-8")
    return manifest


def test_the_fetched_manifest_loads_on_the_drone(tmp_path: Path) -> None:
    source = load_model(_fetched(tmp_path))
    assert source.tiled.tile_px == 64 and source.manifest["schema"] == MANIFEST_SCHEMA


def test_the_bench_measures_a_trained_model_by_name(tmp_path: Path) -> None:
    _fetched(tmp_path)
    path = tmp_path / "p.png"
    PILImage.new("RGB", (128, 96), (30, 80, 30)).save(path)
    pic = ev.Picture(Image("t:p", "t", "s", path.name, 128, 96, ((10.0, 10.0, 30.0, 40.0),)), path)
    summary = ev.measure("coco_tiny", [pic], tmp_path)
    assert summary.n_pictures == 1 and summary.latency_ms["p50"] >= 0


@pytest.mark.parametrize(
    ("model", "why"),
    [("Bad-Name", "små bokstäver"), ("person_gone", "finns inte")],
)
def test_a_wrong_model_name_is_refused(tmp_path: Path, model: str, why: str) -> None:
    with pytest.raises(BenchError, match=why):
        ev.model_detector(model, tmp_path)


def test_a_broken_trained_model_is_a_bench_error(tmp_path: Path) -> None:
    manifest = _fetched(tmp_path)
    (manifest.with_suffix(".onnx")).write_bytes(b"changed")
    with pytest.raises(BenchError, match="kontrollsumma"):
        ev.model_detector("coco_tiny", tmp_path)


# --- models.toml ----------------------------------------------------------------------


def test_the_real_register_is_valid() -> None:
    rows = fw.load_models()
    assert {r["name"] for r in rows} >= {"yolox_s", "damoyolo_s", "coco_yolox_s"}


@pytest.mark.parametrize(
    ("over", "why"),
    [
        ({"licence": "CC-BY-NC-4.0"}, "licensen"),
        ({"licence": "MIT"}, "ska ha licensen"),  # a ready detector must be Apache-2.0
        ({"url": "http://example.invalid/m.onnx"}, "https"),
        ({"sha256": "abc"}, "SHA-256"),
        ({"sha256": "Z" * 64}, "SHA-256"),
        ({"kind": "other"}, "kind"),
        ({}, "family"),  # a ready detector needs family and input_px
    ],
)
def test_a_wrong_register_row_is_refused(tmp_path: Path, over: dict[str, str], why: str) -> None:
    row = {
        "name": "m",
        "kind": "detector",
        "file": "detector/m.onnx",
        "url": "https://example.invalid/m.onnx",
        "size_bytes": 1,
        "sha256": "0" * 64,
        "licence": "Apache-2.0",
        "attribution": "test",
    } | over
    lines = ["[[model]]"] + [f"{k} = {json.dumps(v)}" for k, v in row.items()]
    path = tmp_path / "models.toml"
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    with pytest.raises(BenchError, match=why):
        fw.load_models(path)


def test_without_yes_only_the_plan_is_shown(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    assert fw.main([f"data={tmp_path}"]) == 2
    assert "yes=1" in capsys.readouterr().out
    assert not (tmp_path / "models").exists()


def test_export_writes_the_schema_the_drone_reads() -> None:
    text = (TRAIN / "export_onnx.py").read_text(encoding="utf-8")
    found = re.search(r'^MANIFEST_SCHEMA = "([^"]+)"', text, re.MULTILINE)
    assert found is not None and found.group(1) == MANIFEST_SCHEMA
    cap = re.search(r"^MAX_MODEL_BYTES = (\d+) << (\d+)", text, re.MULTILINE)
    assert cap is not None and int(cap.group(1)) << int(cap.group(2)) == MAX_MODEL_BYTES


# --- datadir --------------------------------------------------------------------------


def test_datadir_prints_the_data_folder(tmp_path: Path, capsys: pytest.CaptureFixture[str]) -> None:
    assert datadir.main([f"data={tmp_path}", "model=yolox_s"]) == 0
    assert capsys.readouterr().out.strip() == str(tmp_path.resolve())
