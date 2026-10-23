# Copyright (C) Alibaba Group Holding Limited. All rights reserved.
# DAMO-YOLO-S as damo/detectors/detector.py and configs/damoyolo_tinynasL25_S.py build
# it, without the config system, distillation or distributed training. Written for
# Searchdrone from those two files (see NOTICE.md in tools/train/vendor).

import ast
from pathlib import Path

import torch.nn as nn

from .giraffe_fpn_btn import GiraffeNeckV2
from .tinynas_res import TinyNAS
from .zero_head import ZeroHead

STRUCTURE = Path(__file__).with_name("tinynas_L25_k1kx.txt")


def _structure():
    info = ast.literal_eval("".join(x.strip() for x in STRUCTURE.read_text().splitlines()))
    for layer in info:
        layer.pop("nbitsA", None)
        layer.pop("nbitsW", None)
    return info


class DamoYolo(nn.Module):
    def __init__(self, num_classes=80, legacy=False):
        # legacy: one class channel more, as the released COCO weights have
        super().__init__()
        self.backbone = TinyNAS(
            structure_info=_structure(),
            out_indices=(2, 4, 5),
            with_spp=True,
            use_focus=True,
            act="relu",
            reparam=True,
        )
        self.neck = GiraffeNeckV2(
            depth=1.0,
            hidden_ratio=0.75,
            in_channels=[128, 256, 512],
            out_channels=[128, 256, 512],
            act="relu",
            spp=False,
            block_name="BasicBlock_3x3_Reverse",
        )
        self.head = ZeroHead(
            num_classes=num_classes,
            in_channels=[128, 256, 512],
            stacked_convs=0,
            reg_max=16,
            act="silu",
            nms=False,
            legacy=legacy,
        )

    def init_model(self):
        for m in self.modules():
            if isinstance(m, nn.BatchNorm2d):
                m.eps = 1e-3
                m.momentum = 0.03
        self.backbone.init_weights()
        self.neck.init_weights()
        self.head.init_weights()

    def forward(self, x, targets=None):
        """Training: ``targets`` is a list of ``(boxes xyxy, classes)`` per picture and
        the result the loss dict. Eval: ``(scores [b, n, c], boxes xyxy [b, n, 4])``."""
        return self.head(self.neck(self.backbone(x)), targets)

    def switch_to_deploy(self):
        for m in self.modules():
            if hasattr(m, "switch_to_deploy") and m is not self:
                m.switch_to_deploy()
