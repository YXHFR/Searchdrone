# sd-schemas (`sd_schemas`)

Shared message schemas (Pydantic models); the single source for every message between processes.

| Module | Contents |
|---|---|
| `base` | `SdModel`, `Id` (UUIDv7) and `new_id()`, positions, GeoJSON checks, `VehicleId` |
| `safety` | `SafetyLimits` (per mission) and `LimitCaps` (the outer cap a pilot may raise them to) |
| `charter` | `MissionCharter`, `CharterProposal`, `AutonomyMode`, sectors |
| `skills` | `SkillSpec`, `SkillRegistry`, `SKILL_REGISTRY` (18 skills) and their parameter models |
| `gate` | `GateRequest`, `GateContext`, `GateDecision` for `sd_safety.evaluate` |
| `mission` | `MissionState`, `Trigger`, `OperatorLatch` |
| `link` | SD-link `Envelope` and every message kind, including manual stick control |
| `api`, `commander` | Panel API and commander API models |
| `decide` | Typed questions and answers for the decision interface |
| `executor`, `vehicle` | Onboard executor states and the vehicle state picture |
| `reason_codes` | Every reason code with its Swedish text |

After changing a model, run `just gen` and commit the generated JSON Schema
(`generated/`) and TypeScript (`software/common/ts-schemas/`).

Part of the [Searchdrone](../../../README.md) monorepo. Licensed under Apache-2.0.

Run its checks from the repository root: `just gen-check` (all checks: `just check`).
