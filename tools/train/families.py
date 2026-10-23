"""The two detector families: how each is built, started from its COCO weights, trained
and exported.

Both start from the published COCO weights (``just detector-fetch``) with one class,
person. Person is class 0 in COCO for both, so the new one-class head starts from the
COCO person channel instead of from nothing: the network already finds people on day
one and the training only has to teach it what they look like from above.

The exported ONNX graph is what ``sd_detect.onnx.OnnxDetector`` reads (its module doc
has the layout of each family's output).
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import torch
from torch import nn

PRETRAINED = {
    "yolox_s": "yolox_s.pth",
    "damoyolo_s": "damoyolo_tinynasL25_S.pt",
}


@dataclass(frozen=True)
class Family:
    model: str  # yolox_s | damoyolo_s
    family: str  # what sd_detect.onnx calls it: yolox | damoyolo
    pad: int  # letterbox fill, as the family was trained
    outputs: tuple[str, ...]
    licence: str = "Apache-2.0"


FAMILIES = {
    "yolox_s": Family("yolox_s", "yolox", 114, ("output",)),
    "damoyolo_s": Family("damoyolo_s", "damoyolo", 0, ("scores", "boxes")),
}


def build(model: str, num_classes: int = 1) -> nn.Module:
    if model == "yolox_s":
        from vendor.yolox import YOLOPAFPN, YOLOX, YOLOXHead

        net = YOLOX(YOLOPAFPN(0.33, 0.50, act="silu"), YOLOXHead(num_classes, 0.50, act="silu"))
        for m in net.modules():
            if isinstance(m, nn.BatchNorm2d):
                m.eps, m.momentum = 1e-3, 0.03
        net.head.initialize_biases(1e-2)
        return net
    if model == "damoyolo_s":
        from vendor.damoyolo import DamoYolo

        net = DamoYolo(num_classes=num_classes)
        net.init_model()
        return net
    raise ValueError(f"okänd modell {model!r}")


def _load(path: Path) -> dict[str, torch.Tensor]:
    ck: Any = torch.load(path, map_location="cpu", weights_only=True)
    state = ck.get("model", ck) if isinstance(ck, dict) else ck
    return {k.removeprefix("module."): v for k, v in state.items()}


def load_coco(net: nn.Module, model: str, weights: Path) -> list[str]:
    """Load the COCO weights into a one-class ``net``; the class layer takes the COCO
    person channel (0). Returns the names that were started from the person channel."""
    state = _load(weights)
    own = net.state_dict()
    person: list[str] = []
    cls_prefix = "head.cls_preds." if model == "yolox_s" else "head.gfl_cls."
    for name, value in state.items():
        if name not in own:
            continue
        if name.startswith(cls_prefix) and value.shape[0] != own[name].shape[0]:
            own[name] = value[: own[name].shape[0]].clone()  # person first
            person.append(name)
        elif value.shape == own[name].shape:
            own[name] = value
        else:
            raise ValueError(f"{name}: {tuple(value.shape)} passar inte {tuple(own[name].shape)}")
    missing = [k for k in own if k not in state and not k.endswith("num_batches_tracked")]
    if missing:
        raise ValueError(f"COCO-vikterna saknar {missing[:5]}")
    net.load_state_dict(own)
    return person


def train_targets(model: str, boxes: list[torch.Tensor], device: torch.device) -> Any:
    """Per-picture person boxes ``[n, 4]`` (corners, input pixels) in the form the
    family's loss takes."""
    if model == "yolox_s":
        most = max((len(b) for b in boxes), default=0)
        out = torch.zeros((len(boxes), max(most, 1), 5), device=device)
        for i, b in enumerate(boxes):
            if len(b):
                b = b.to(device)
                out[i, : len(b), 1] = (b[:, 0] + b[:, 2]) / 2
                out[i, : len(b), 2] = (b[:, 1] + b[:, 3]) / 2
                out[i, : len(b), 3] = b[:, 2] - b[:, 0]
                out[i, : len(b), 4] = b[:, 3] - b[:, 1]
        return out
    return [(b.to(device), torch.zeros(len(b), dtype=torch.long, device=device)) for b in boxes]


def loss(net: nn.Module, images: torch.Tensor, targets: Any) -> torch.Tensor:
    return net(images, targets)["total_loss"]


class ExportWrapper(nn.Module):
    """The graph as ``sd_detect.onnx`` reads it: YOLOX undecoded, DAMO-YOLO deployed."""

    def __init__(self, net: nn.Module, model: str) -> None:
        super().__init__()
        self.net = net
        self.model = model
        if model == "yolox_s":
            net.head.decode_in_inference = False  # type: ignore[union-attr]
        else:
            net.switch_to_deploy()  # type: ignore[operator]

    def forward(self, x: torch.Tensor) -> Any:
        return self.net(x)


def predict(net: nn.Module, model: str, images: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    """Boxes (corners, input pixels) and person scores ``[b, n, 4]``, ``[b, n]`` from a
    net in eval mode: the same decoding as ``sd_detect.onnx``."""
    if model == "yolox_s":
        net.head.decode_in_inference = True  # type: ignore[union-attr]
        out = net(images)
        xy, wh = out[..., :2], out[..., 2:4]
        boxes = torch.cat([xy - wh / 2, xy + wh / 2], dim=-1)
        return boxes, out[..., 4] * out[..., 5]
    scores, boxes = net(images)
    return boxes, scores[..., 0]
