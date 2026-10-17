# sd-detect (`sd_detect`)

The person-detector interface and a null detector. Real detector engines arrive in stage 2.

- `Detector`: the protocol every engine follows, `detect(image: Frame) -> list[Detection]`.
  It assumes no model type, so both a class-and-score detector and an embedding model fit.
- `Detection`: `label` (only `"person"`), `score` (0..1), `bbox` (pixels in the image) and an
  optional `embedding`. The fields are preliminary and may grow before stage 2.
- `NullDetector`: the only engine in v0.1.0. It never finds anything and says so
  (`built = False`).

A detection is only a hint for a person to look at: nothing here steers the vehicle or
reaches the safety gate, and only a person confirms a find. No face recognition.

Part of the [Searchdrone](../../README.md) monorepo. Licensed under Apache-2.0.

Run its checks from the repository root: `just test` (all checks: `just check`).
