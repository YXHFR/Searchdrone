# Copyright (c) Megvii Inc. All rights reserved.
# A minimal copy of YOLOX's model (https://github.com/Megvii-BaseDetection/YOLOX,
# commit 6ddff4824372906469a7fae2dc3206c7aa4bbaee, Apache-2.0): the network and its
# training loss, nothing else. See NOTICE.md in tools/train/vendor for what was changed.

from .darknet import CSPDarknet
from .yolo_head import YOLOXHead
from .yolo_pafpn import YOLOPAFPN
from .yolox import YOLOX

__all__ = ["YOLOX", "CSPDarknet", "YOLOPAFPN", "YOLOXHead"]
