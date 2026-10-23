# sd-detect (`sd_detect`)

The detector interface, the class register and the detector sources.

- **Class register** (`classes.toml`, read by `sd_detect.classes`): one row per class
  with its id (the detection's label), Swedish name, kind (`target` or `context`), the
  source that reports it, the plausible size on the ground and the source's settings.
  Start classes: `person`, `red_block`, `car`, `building`, `road`. A new class needs only
  a row there and a source that reports it. No row may describe a face, a licence plate
  or another identity; the loader refuses such ids.
- **Sources** (`sd_detect.sources.SOURCES`):
  - `color` (`ColorDetector`): a hue, saturation and value threshold on the real pixels,
    touching pixels joined into patches. Used for `red_block`.
  - `oracle` (`SimLabelDetector`): the simulator's own labelled boxes, marked
    `source = "oracle"`; runs only with `SD_SIM=1` and never stands for a real model.
    Used for `car`, `building` and `road`.
  - `model`: the trained person detector (`sd_detect.onnx`), built only when
    `SD_DETECTOR_MODEL` names its manifest (`tools/train/README.md`). Unset: `person` is
    not reported. Set but wrong (missing file, wrong SHA-256, unknown family or
    licence): perception is turned off and `sd_onboard` sends the reason to the ground
    (`executor.perception_off`), never a silent fallback to another detector. Flying
    goes on: `sd_onboard` may be restarted in the air, so a hint never stops it. The
    SHA-256 guards against a damaged or swapped file, not
    against someone who can write the model folder. What it is and is not for:
    [`MODEL_CARD.md`](MODEL_CARD.md).
- `build_sources`, `detect_all` and `keep_plausible`: build the sources the register
  names, run them on a `Still` (keeping only each source's own classes) and drop
  detections whose size in metres does not fit their class (a red car is no red block).
- `Detector`: the protocol for a frame-only engine, `detect(image: Frame) -> list[Detection]`.
  `Detection` has `label`, `score` (0..1), `bbox` (pixels), an optional `embedding` and
  `source` (`model`, `color` or `oracle`).
- `NullDetector`: an engine that never finds anything and says so (`built = False`).
- `TiledDetector(inner)`: runs a frame-only engine over overlapping tiles of a large
  picture (`tile_grid`: 640 px tiles, 15 % overlap, 45 tiles for 4608×2592), moves its
  boxes back to the picture, joins a box cut by a tile edge with its other half in the
  next tile and removes doubles (per-class NMS). An optional `skip_mask` leaves tiles out
  (for example the sky).
- `OnnxDetector` and `load_model` (`sd_detect.onnx`): run a YOLOX or DAMO-YOLO ONNX file
  with ONNX Runtime on the CPU (letterboxed BGR input, decoding, score floor, NMS),
  after checking its size and SHA-256 against the manifest (`sd.detector-model/v1`).
  The manifest names the file, family, input size, classes, licence and what the model
  was trained on. ONNX Runtime is imported only when a model is chosen.

A detection is only a hint for a person to look at: nothing here steers the vehicle or
reaches the safety gate, and only a person confirms a find. No face recognition.

The tests run the sources on pictures rendered by Gazebo in the simulator's world
(`tests/testdata/stills`, regenerated with `uv run --package sd-sim python -m
sd_sim.render_stills`).

Part of the [Searchdrone](../../README.md) monorepo. Licensed under Apache-2.0.

Run its checks from the repository root: `just test` (all checks: `just check`).
