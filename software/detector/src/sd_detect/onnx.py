"""``OnnxDetector``: a trained person detector (YOLOX or DAMO-YOLO) run with ONNX Runtime.

The same weights are compiled for the drone's AI board; this engine runs them on a
computer: for the image bank (``tools/bench``), the simulator and as a reserve. It takes
one picture (usually a 640 px tile from ``TiledDetector``) and gives person boxes.

Both families take the picture as BGR, 0-255, unnormalised, shrunk to fit
``input_px`` (never enlarged) and padded on the right and bottom: YOLOX with grey 114,
DAMO-YOLO with black, as they were trained. What comes out differs:

- ``yolox``: one tensor ``[1, n, 5 + classes]``, not yet decoded: per cell of the 8, 16
  and 32 px grids, the centre offset, the log of width and height, the objectness and
  the class scores (both after a sigmoid). The score is objectness times the person
  class score.
- ``damoyolo``: two tensors, class scores ``[1, n, classes]`` and boxes ``[1, n, 4]``
  (corners in input pixels).

Boxes below ``score_min`` are dropped, the rest go through non-maximum suppression and
back to the picture's pixels. The model file is checked against its SHA-256 before it
is opened; a file that does not match is never run.

``load_model`` reads a model's manifest (``<name>.json`` next to ``<name>.onnx``, written
by ``tools/train/export_onnx.py``) and gives the ``model`` source: the engine behind a
``TiledDetector``. Like every detector it gives hints, never decisions.
"""

from __future__ import annotations

import hashlib
import json
from collections.abc import Sequence
from pathlib import Path
from typing import Any, ClassVar, Literal

import numpy as np
import numpy.typing as npt

from sd_detect.api import Detection, Frame, SourceKind
from sd_detect.boxes import nms, to_bbox
from sd_detect.still import Still
from sd_detect.tiling import TiledDetector

__all__ = [
    "MANIFEST_SCHEMA",
    "Family",
    "ModelError",
    "ModelSource",
    "OnnxDetector",
    "load_model",
    "sha256_of",
]

Family = Literal["yolox", "damoyolo"]
FAMILIES: tuple[Family, ...] = ("yolox", "damoyolo")
STRIDES = (8, 16, 32)
PAD_VALUE: dict[Family, int] = {"yolox": 114, "damoyolo": 0}
SCORE_MIN = 0.05
NMS_IOU = 0.5
MANIFEST_SCHEMA = "sd.detector-model/v1"
MAX_MODEL_BYTES = 512 << 20


class ModelError(Exception):
    """The model file is missing, wrong or not what its manifest says (Swedish text)."""


def sha256_of(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as fh:
        while chunk := fh.read(1 << 20):
            digest.update(chunk)
    return digest.hexdigest()


def _resize(rgb: npt.NDArray[np.uint8], width: int, height: int) -> npt.NDArray[np.uint8]:
    from PIL import Image

    return np.asarray(Image.fromarray(rgb).resize((width, height), Image.Resampling.BILINEAR))


class OnnxDetector:
    """One ONNX model of a known family (module doc)."""

    name: ClassVar[str] = "onnx"
    license: ClassVar[str] = "Apache-2.0"

    def __init__(
        self,
        model_path: Path,
        family: Family,
        sha256: str,
        input_px: int = 640,
        providers: Sequence[str] = ("CPUExecutionProvider",),
        person_index: int = 0,
        score_min: float = SCORE_MIN,
        nms_iou: float = NMS_IOU,
        label: str = "person",
    ) -> None:
        if family not in FAMILIES:
            raise ModelError(f"okänd modellfamilj {family!r}")
        if input_px % 32 or not 64 <= input_px <= 2048:
            raise ModelError(f"input_px ska vara delbart med 32 (64-2048), inte {input_px}")
        model_path = Path(model_path)
        if not model_path.is_file():
            raise ModelError(f"Modellfilen finns inte: {model_path}")
        if model_path.stat().st_size > MAX_MODEL_BYTES:
            raise ModelError(f"Modellfilen är orimligt stor: {model_path}")
        got = sha256_of(model_path)
        if got != sha256.lower():
            raise ModelError(
                f"Modellfilens kontrollsumma stämmer inte ({model_path.name}: {got}); "
                "den körs inte."
            )
        import onnxruntime as ort

        options = ort.SessionOptions()
        options.log_severity_level = 3
        self.session: Any = ort.InferenceSession(
            str(model_path), sess_options=options, providers=list(providers)
        )
        self.input_name: str = self.session.get_inputs()[0].name
        self.family: Family = family
        self.input_px = input_px
        self.person_index = person_index
        self.score_min = score_min
        self.nms_iou = nms_iou
        self.label = label
        self._grids: tuple[npt.NDArray[np.float32], npt.NDArray[np.float32]] | None = None

    # --- in ---------------------------------------------------------------------------

    def preprocess(self, image: Frame) -> tuple[npt.NDArray[np.float32], float]:
        """The input tensor ``[1, 3, s, s]`` and the scale from picture to input."""
        rgb = np.frombuffer(image.rgb, dtype=np.uint8).reshape(image.height, image.width, 3)
        s = self.input_px
        r = min(s / image.width, s / image.height, 1.0)
        if r < 1.0:
            w, h = max(1, round(image.width * r)), max(1, round(image.height * r))
            rgb = _resize(rgb, w, h)
        h, w = rgb.shape[:2]
        canvas = np.full((s, s, 3), PAD_VALUE[self.family], dtype=np.uint8)
        canvas[:h, :w] = rgb[:, :, ::-1]  # BGR
        return np.ascontiguousarray(canvas.transpose(2, 0, 1)[None], dtype=np.float32), r

    # --- out --------------------------------------------------------------------------

    def _yolox_grids(self) -> tuple[npt.NDArray[np.float32], npt.NDArray[np.float32]]:
        if self._grids is None:
            cells, strides = [], []
            for stride in STRIDES:
                n = self.input_px // stride
                ys, xs = np.meshgrid(np.arange(n), np.arange(n), indexing="ij")
                cells.append(np.stack([xs, ys], axis=-1).reshape(-1, 2))
                strides.append(np.full((n * n, 1), stride))
            self._grids = (
                np.concatenate(cells).astype(np.float32),
                np.concatenate(strides).astype(np.float32),
            )
        return self._grids

    def decode(
        self, outputs: Sequence[npt.NDArray[np.float32]]
    ) -> tuple[npt.NDArray[np.float64], npt.NDArray[np.float64]]:
        """Boxes ``(x0, y0, x1, y1)`` in input pixels and person scores."""
        if self.family == "yolox":
            raw = np.asarray(outputs[0], dtype=np.float32)[0]
            cells, strides = self._yolox_grids()
            if raw.shape[0] != len(cells) or raw.shape[1] <= 5 + self.person_index:
                raise ModelError(
                    f"YOLOX-utdata har formen {raw.shape}, väntat ({len(cells)}, 5+klasser)"
                )
            xy = (raw[:, :2] + cells) * strides
            wh = np.exp(np.clip(raw[:, 2:4], -10, 10)) * strides
            scores = raw[:, 4] * raw[:, 5 + self.person_index]
            boxes = np.concatenate([xy - wh / 2, xy + wh / 2], axis=1)
        else:
            if len(outputs) < 2:
                raise ModelError("DAMO-YOLO-modellen ska ge två utdata: poäng och rutor")
            scores = np.asarray(outputs[0], dtype=np.float32)[0][:, self.person_index]
            boxes = np.asarray(outputs[1], dtype=np.float32)[0]
        return boxes.astype(np.float64), scores.astype(np.float64)

    def detect(self, image: Frame) -> list[Detection]:
        tensor, r = self.preprocess(image)
        outputs = self.session.run(None, {self.input_name: tensor})
        boxes, scores = self.decode(outputs)
        keep = scores >= self.score_min
        boxes, scores = boxes[keep] / r, scores[keep]
        boxes[:, [0, 2]] = np.clip(boxes[:, [0, 2]], 0, image.width)
        boxes[:, [1, 3]] = np.clip(boxes[:, [1, 3]], 0, image.height)
        real = (boxes[:, 2] - boxes[:, 0] >= 1) & (boxes[:, 3] - boxes[:, 1] >= 1)
        boxes, scores = boxes[real], scores[real]
        return [
            Detection(
                label=self.label,
                score=float(min(max(scores[i], 0.0), 1.0)),
                bbox=to_bbox(boxes[i]),
                source="model",
            )
            for i in nms(boxes, scores, self.nms_iou)
        ]


class ModelSource:
    """The register's ``model`` source: a trained detector over the still's tiles."""

    name: ClassVar[str] = "model"
    license: ClassVar[str] = "Apache-2.0"
    source: ClassVar[SourceKind] = "model"

    def __init__(self, tiled: TiledDetector, manifest: dict[str, Any]) -> None:
        self.tiled = tiled
        self.manifest = manifest

    def detect_still(self, still: Still) -> list[Detection]:
        return self.tiled.detect(still.frame)


def load_model(
    manifest_path: Path, providers: Sequence[str] = ("CPUExecutionProvider",)
) -> ModelSource:
    """The ``model`` source from a manifest (module doc). Any doubt is a ``ModelError``."""
    manifest_path = Path(manifest_path)
    try:
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    except (OSError, ValueError) as exc:
        raise ModelError(f"Kan inte läsa modellens manifest {manifest_path}: {exc}") from exc
    if manifest.get("schema") != MANIFEST_SCHEMA:
        raise ModelError(f"{manifest_path.name} är inget modellmanifest ({MANIFEST_SCHEMA})")
    onnx_name = str(manifest.get("onnx", ""))
    if not onnx_name or Path(onnx_name).name != onnx_name:
        raise ModelError("Manifestets onnx ska vara ett filnamn i samma mapp")
    classes = manifest.get("classes", [])
    if "person" not in classes:
        raise ModelError("Modellen hittar inte klassen person")
    engine = OnnxDetector(
        manifest_path.with_name(onnx_name),
        family=manifest["family"],
        sha256=str(manifest["sha256"]),
        input_px=int(manifest.get("input_px", 640)),
        providers=providers,
        person_index=classes.index("person"),
    )
    tiled = TiledDetector(engine, tile_px=int(manifest.get("tile_px", engine.input_px)))
    return ModelSource(tiled, manifest)
