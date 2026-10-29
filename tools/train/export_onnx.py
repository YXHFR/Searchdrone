"""Make the ONNX file from the best training epoch: ``just export-detector model=yolox_s``.

Keys: ``model`` (yolox_s, damoyolo_s), ``data``, ``train`` (the training folder,
default ``<data>/train/<model>``), ``name`` (default ``person_<model>``), ``out``
(default ``<data>/models/detector``).

Writes ``<name>.onnx`` and its manifest ``<name>.json`` (``sd.detector-model/v1``: the
family, the input size, the classes, the SHA-256 of the ONNX file, what it was trained
on and under which licences). Before writing, the ONNX file is run with ONNX Runtime
next to PyTorch on the same pictures; if they disagree, nothing is written. Point
``SD_DETECTOR_MODEL`` at the manifest to use the model (``tools/train/README.md``).
"""

from __future__ import annotations

import datetime as dt
import hashlib
import json
import re
import sys
from collections.abc import Sequence
from pathlib import Path
from typing import Any

import families
import numpy as np
import torch
from bench.common import BenchError, parse_args, say, warn
from bench.datasets import load_register

KEYS = ("model", "data", "train", "name", "out")
MANIFEST_SCHEMA = "sd.detector-model/v1"
MAX_MODEL_BYTES = 128 << 20  # sd_detect.onnx refuses a larger file on board
OPSET = 17
MAX_DIFF = 2e-2  # scores and pixels; fp32 both sides


def export(
    best: Path, out_dir: Path, name: str, check_images: int = 2, seed: int = 0
) -> tuple[Path, Path]:
    ck: dict[str, Any] = torch.load(best, map_location="cpu", weights_only=True)
    model, settings = ck["model"], ck["settings"]
    fam = families.FAMILIES[model]
    tile = int(settings["tile"])
    net = families.build(model)
    net.load_state_dict(ck["ema"])
    net.eval()
    wrapper = families.ExportWrapper(net, model).eval()
    out_dir.mkdir(parents=True, exist_ok=True)
    onnx_path = out_dir / f"{name}.onnx"
    tmp = out_dir / f".{name}.onnx.tmp"
    dummy = torch.zeros((1, 3, tile, tile))
    torch.onnx.export(
        wrapper,
        (dummy,),
        str(tmp),
        input_names=["images"],
        output_names=list(fam.outputs),
        opset_version=OPSET,
        dynamo=False,
    )
    _check(wrapper, tmp, tile, check_images, seed)
    if tmp.stat().st_size > MAX_MODEL_BYTES:
        tmp.unlink()
        raise BenchError("ONNX-filen är större än den inbyggda datorn tar emot (128 MB).", 1)
    tmp.replace(onnx_path)
    digest = hashlib.sha256(onnx_path.read_bytes()).hexdigest()
    register = load_register()
    # what the weights have seen in all: a run started from earlier weights (init=)
    # carries their sets too, so every one of them is credited
    sets = [s for s in ck.get("trained_on", settings["sets"]) if s in register]
    pretrained = ck.get("pretrained_on", ["simfarm"] if settings.get("simfarm_epochs") else [])
    manifest = {
        "schema": MANIFEST_SCHEMA,
        "name": name,
        "onnx": onnx_path.name,
        "sha256": digest,
        "family": fam.family,
        "model": model,
        "input_px": tile,
        "tile_px": tile,
        "classes": ["person"],
        "licence": fam.licence,
        "trained_on": [
            {"dataset": s, "licence": register[s].licence, "attribution": register[s].attribution}
            for s in sets
        ],
        "pretrained_on": list(pretrained),
        "base_weights": f"COCO {model} (Apache-2.0)",
        "val_ap50": ck.get("val_ap50"),
        "val_ap50_by_dataset": ck.get("val_ap50_by_dataset"),
        "epoch": ck.get("epoch"),
        "settings": settings,
        "exported": dt.datetime.now(dt.UTC).isoformat(timespec="seconds"),
        "publish": all(register[s].publish_weights for s in sets),
    }
    manifest_path = out_dir / f"{name}.json"
    manifest_path.write_text(json.dumps(manifest, indent=1) + "\n", encoding="utf-8")
    return onnx_path, manifest_path


def _check(wrapper: torch.nn.Module, path: Path, tile: int, n: int, seed: int) -> None:
    import onnxruntime as ort

    session = ort.InferenceSession(str(path), providers=["CPUExecutionProvider"])
    rng = np.random.default_rng(seed)
    for _ in range(n):
        x = rng.uniform(0, 255, size=(1, 3, tile, tile)).astype(np.float32)
        with torch.no_grad():
            want = wrapper(torch.from_numpy(x))
        want = [want] if isinstance(want, torch.Tensor) else list(want)
        got = session.run(None, {"images": x})
        for w, g in zip(want, got, strict=True):
            w = w.numpy()
            scale = np.maximum(np.abs(w), 1.0)
            diff = float(np.max(np.abs(w - g) / scale))
            if not diff <= MAX_DIFF:
                raise BenchError(f"ONNX-filen räknar inte som PyTorch (skillnad {diff:.4f}).", 1)


def main(argv: Sequence[str] | None = None) -> int:
    try:
        args = parse_args(sys.argv[1:] if argv is None else argv, KEYS, "export-detector")
        model = args.get("model", "")
        if model not in families.FAMILIES:
            raise BenchError(f"model= ska vara en av {', '.join(families.FAMILIES)}", 2)
        if not args.get("data"):
            raise BenchError("data= saknas (just export-detector fyller i den).", 2)
        data = Path(args["data"]).expanduser().resolve()
        train_dir = Path(args.get("train") or data / "train" / model)
        best = train_dir / "best.pt"
        if not best.is_file():
            raise BenchError(f"{best} finns inte. Kör först: just train-detector model={model}", 2)
        name = args.get("name") or f"person_{model}"
        if not re.fullmatch(r"[a-z0-9_]{1,40}", name):
            raise BenchError("name= får bara ha små bokstäver, siffror och _", 2)
        out = Path(args.get("out") or data / "models" / "detector")
        onnx_path, manifest = export(best, out, name)
        say(f"skrev {onnx_path} och {manifest}")
        say(f"Mät den: just bench-detector model={name} part=val")
        return 0
    except BenchError as exc:
        warn(str(exc))
        return exc.code


if __name__ == "__main__":
    raise SystemExit(main())
