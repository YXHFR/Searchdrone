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

## stage_scope: v0.1.0

This release builds the foundation only. **Reject (`major`) a software change that
builds any of the following**, except as a stub that returns "not implemented":

- Person detection: a detector model (RF-DETR, YOLOX or similar), image tiling, a
  tracker, a candidate queue where a human confirms finds.
- Follow functions: `follow_track`, `stop_follow`, `follow_me`, `find_me`, including
  phone GPS or phone camera input and a fake operator phone in the simulator.
- Search from a free-text description of the missing person, a probability map, a
  mission debrief.
- Remote access over the internet: a tunnel service (for example `cloudflared`), a
  `remote` compose profile, or any listener that is reachable from outside the local
  network.
- MAVLink 2 message signing and message authentication codes on the vehicle link.
- Voice control.
- An Ultralytics plugin or any other AGPL component.
- Real hardware in operation: companion computer and AI accelerator drivers, radio
  link, bench flight-controller tests, hardware-in-the-loop, field profiles.
- Camera gimbals and thermal cameras.
- More than one drone, docking stations, swarms (only a `vehicle_id` field exists).
- Photorealistic simulation, test campaigns over many random seeds.
- A native mobile app, browser geolocation in the panel, a mission history view.

**Exception: `hardware/**` is never rejected for scope.** Hardware work (CAD, FEA,
circuit boards, the bill of materials) runs in parallel to the software. Pull requests
under `hardware/**` are governed by the hardware gate and the owner's approval, not by
this list.

This section is temporary. It is removed or replaced by `stage_scope: v0.2.0` in its own
pull request right after the `v0.1.0` tag. The rule below stays permanently.

## Permanent scope rule

**Face recognition and object search for identifying people: never, in any stage.**
Reject (`blocker`) any change that adds face recognition, face embeddings, person
re-identification by biometric features, or a search for a specific person by their
face.
