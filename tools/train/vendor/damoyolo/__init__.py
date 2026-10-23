# Copyright (C) Alibaba Group Holding Limited. All rights reserved.
# A minimal copy of DAMO-YOLO's S model (https://github.com/tinyvision/DAMO-YOLO,
# commit 319572eef367340267ab6ab8ae253527a71d7c3a, Apache-2.0): the network and its
# training loss, nothing else. See NOTICE.md in tools/train/vendor for what was changed.

from .model import DamoYolo

__all__ = ["DamoYolo"]
