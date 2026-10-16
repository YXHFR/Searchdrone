# sd-safety (`sd_safety`)

Pure safety-gate logic. Every order to the vehicle, from the AI commander or the operator,
is evaluated here, on the ground and again onboard. No I/O and no clock reads: the caller
passes the time in `GateContext.now_mono_ms`.

```python
from sd_safety import evaluate

decision = evaluate(request, context)  # GateDecision: ALLOW, REJECT or PENDING_APPROVAL
```

| Module | Contents |
|---|---|
| `evaluate` | `evaluate(req, ctx) -> GateDecision`: runs every rule and collects every finding |
| `rules` | The rules (`R-*`), one function each, in table order |
| `priority` | `REASON_PRIORITY` and the operator text for a decision |
| `policy` | Hard-coded sets the gate applies whatever the registry says (forbidden names, always-approve skills, ...) |
| `limits` | Standard limits, the outer cap, the flight controller's fence for given limits, `check_limits` |
| `targets` | Geodesic distance (WGS84) and exact polygon checks for routes and areas, concave areas included |
| `constants` | Every time and margin the safety logic uses |

Routes are checked leg by leg against the charter's area, not sampled, and a polygon
that crosses itself is rejected. Anything the gate cannot work out counts as a rejection.

Part of the [Searchdrone](../../../README.md) monorepo. Licensed under Apache-2.0.

Run its checks from the repository root: `just test` (all checks: `just check`).
