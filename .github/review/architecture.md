# Review instructions: architecture and code quality

Role name: `architecture`. Read `common.md` first; its rules apply here too.

## What you review

- **Package boundaries.** The change belongs in the right package, and the import
  contracts in `.importlinter` still hold without new exceptions.
- **Tests.** New logic has tests at the lowest sensible level (unit before component
  before simulator). Pure logic in the safety gate, pattern generators and state
  machines meets the coverage thresholds.
- **Generated files.** Files generated from the schemas are never edited by hand; a
  schema change comes with regenerated output (`just gen`, checked by `just gen-check`).
- **Public interfaces.** Schemas, the HTTP API and the vehicle link protocol are
  versioned and backward compatible, or the change is marked breaking (`!` in the
  title) and says why.
- **Skill registry.** It stays easy to extend: one entry per skill, no special cases
  scattered across packages.
- **Clocks.** Packages that must be testable with a fake clock use the injected clock,
  not `loop.time()`, `time.time()` or `datetime.now()` directly.
- **Readability.** No dead code, no duplicated logic that already exists in a
  `software/common/` package, no needless complexity.
- **The gates themselves are not weakened.** `justfile`, `.importlinter`, `tools/**`,
  `.pre-commit-config.yaml`, `.github/workflows/**`, `.github/review/**` and
  `critical_paths.txt`. The hardware gate too: `software/hwcalc/**`,
  `tools/check_hw_files.py`, `.github/workflows/hardware.yml`.

## Blocking examples (`blocker` or `major`)

- New logic without tests.
- An import contract broken and hidden with `# noqa`, an exception or `ignore_imports`.
- A lowered coverage threshold, or a check removed from `just lint` or `just check`.
- A line removed or narrowed in `critical_paths.txt`.
- A schema change without regenerated files.
- Dead code, or logic duplicated from a common package.
- `loop.time()` or wall-clock calls in a clock-sensitive package.
- A changed formula, unit or limit in `sd_hwcalc` without the reference test showing
  the same reference values; a checked hardware file without a matching job in
  `hardware.yml`.

## stage_scope: v0.2.0

This release builds the search chain on top of the foundation. In scope, and not to be
rejected for scope: a person detector, image tiling, a tracker, a candidate queue where
a human confirms finds, the search profile, the probability map, cloud matching of
candidates against the profile, the skills `inspect_detection`, `orbit` and
`capture_stills`, and following a confirmed find (`follow_track`, `stop_follow`). The
detector on a companion computer with an AI accelerator, the bench and flight tests of
the 7-inch quad, the link over IP on a 4G modem, and importing an onboard recording
after landing are in scope as well.

**Reject (`major`) a software change that builds any of the following**, except as a
stub that returns "not implemented":

- A small local language model or vision-language model in the chain: for chat, for
  the profile proposal or for matching without a network. Without a network, matching
  is the colour score in code plus the operator's eyes. The person detector is not
  affected by this rule.
- A vision-language model on board, a Jetson, more than one drone or swarms.
- Thermal cameras. Interfaces may carry `sensor = "thermal"` as a value, nothing more.
- Camera gimbals or camera tilt (`set_camera_tilt` stays a stub).
- `follow_me`, `find_me` and the phone's position.
- Reinforcement learning for where the drone searches.
- An agent simulation of how the missing person moves.
- A national elevation model for georeferencing. Flat ground at home altitude plus the
  simulator's world elevation is used.
- The 14-inch drone. The 7-inch quad is the vehicle that flies in this stage.
- A wfb-ng video link or a separate ground radio. The link is IP over 4G.
- Uploading a photo of the missing person, or any API that takes a photo for comparison.
- Ultralytics, BoxMOT, `rfdetr[plus]` or any other AGPL component.

**Exception: `hardware/**` is never rejected for scope.** Hardware work (CAD, FEA,
circuit boards, the bill of materials) runs in parallel to the software. Pull requests
under `hardware/**` are governed by the hardware gate and the owner's approval, not by
this list.

This section is temporary. It is replaced in its own pull request when the search chain
stage is released. The rule below stays permanently.

## Permanent scope rule

**Face recognition and object search for identifying people: never, in any stage.**
Reject (`blocker`) any change that adds face recognition, face embeddings, person
re-identification by biometric features, or a search for a specific person by their
face.
