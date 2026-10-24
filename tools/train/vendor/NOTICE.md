# The model code in tools/train/vendor

The two detector families are trained with the network and loss code their authors
published, copied here so training needs neither package (both pull in a large
tree of other tools). Only what training and export need was taken. Both are under the
Apache License 2.0 (`LICENSE` in each folder; DAMO-YOLO's `NOTICE` is kept as it came).

## YOLOX (`yolox/`)

From <https://github.com/Megvii-BaseDetection/YOLOX>, commit
`6ddff4824372906469a7fae2dc3206c7aa4bbaee` (`yolox/models/`). Copyright (c) Megvii Inc.

- `darknet.py`, `network_blocks.py`, `yolo_pafpn.py`, `losses.py`: unchanged.
- `yolo_head.py`: imports its three helpers from `_utils.py` instead of `yolox.utils`,
  logs with `logging` instead of `loguru`, and drops `visualize_assign_result`. The
  label assignment turns autocast off with `torch.autocast(<device>, enabled=False)`
  instead of `torch.cuda.amp.autocast(enabled=False)`, so it is off on any device.
- `yolox.py`: drops `visualize`.
- `_utils.py`: `meshgrid`, `bboxes_iou` and `cxcywh2xyxy` from `yolox/utils/compat.py`
  and `yolox/utils/boxes.py`, unchanged.
- `__init__.py`: new; exports the four classes.

## DAMO-YOLO (`damoyolo/`)

From <https://github.com/tinyvision/DAMO-YOLO>, commit
`319572eef367340267ab6ab8ae253527a71d7c3a` (`damo/base_models/`). Copyright (C)
Alibaba Group Holding Limited. `core_utils.py`, `bbox_calculator.py`, `ota_assigner.py`
and `weight_init.py` carry OpenMMLab's copyright (Apache-2.0) as in the original;
`gfocal_loss.py` mainly comes from GFocalV2 (<https://github.com/implus/GFocalV2>,
Apache-2.0), as its own header says.

- All files sit in one folder, so the relative imports name `.ops`, `.ota_assigner` and
  so on instead of `..core.ops`.
- `ops.py`: `make_divisible` copied in from `damo/utils/model_utils.py`.
- `zero_head.py`: logs with `logging` instead of `loguru`; takes the targets as
  `(boxes, classes)` pairs instead of DAMO-YOLO's `BoxList`; returns the raw scores and
  boxes without its own NMS (the ONNX consumer, `sd_detect.onnx`, does the NMS);
  computes the class scores and the loss in float32 with autocast off (the label
  assignment and the quality focal loss call `binary_cross_entropy`, which mixed
  precision refuses).
- `core_utils.py`: `damo/base_models/core/utils.py`, unchanged.
- `tinynas_res.py`, `giraffe_fpn_btn.py`, `gfocal_loss.py`, `weight_init.py`: unchanged
  apart from the imports.
- `ota_assigner.py`: unchanged apart from the imports, except that the move to the CPU
  catches only `torch.OutOfMemoryError` instead of every `RuntimeError`, so other errors
  are not reported as running out of memory.
- `bbox_calculator.py`: unchanged apart from the imports, except that `batched_nms` and
  `multiclass_nms` are removed: nothing calls them, and `batched_nms` ran `eval` on a
  config string.
- `tinynas_L25_k1kx.txt`: the S backbone's structure, from
  `damo/base_models/backbones/nas_backbones/`, unchanged.
- `model.py`: new; builds DAMO-YOLO-S as `damo/detectors/detector.py` and
  `configs/damoyolo_tinynasL25_S.py` do, without the config system, distillation or
  distributed training.
- `__init__.py`: new.

The new files (`model.py`, both `__init__.py`) and the much changed `zero_head.py` are
also Searchdrone's (REUSE.toml lists both copyrights).
