# sd-onboard (`sd_onboard`)

The onboard process `sd-onboard`: it takes skill orders from the ground station over
SD-link, checks each one again onboard and flies it through the vehicle link (`sd_mav`).
In v0.1 it runs on a PC next to ArduPilot SITL.

```bash
uv run sd-onboard            # reads config/onboard.toml; exit code 2 when a start check fails
```

**Who steers.** The executor is a state machine (`IDLE`, `PREFLIGHT`, `TAKEOFF`,
`TRANSIT`, `SURVEY`, `HOLD`, `RTL`, `LANDED`, `HELD`, `FAILSAFE`). Every message that
steers passes compare-and-set: it only leaves while the executor is in charge and the
flight controller is in the mode the executor last asked for. A mode change from outside,
a flight-controller failsafe, or a STOP / FLY HOME notice from the ground makes it let go
at once (`HELD` or `FAILSAFE`); only the pilot's Resume click hands control back. STOP and
FLY HOME themselves reach the flight controller straight from the ground station.

**Checks onboard.** The shared safety gate (`sd_safety`) runs again on the handler's own
route, then the executor's own checks: a stale order, a missing or wrong approval, the
stop latch, the takeoff preconditions (preflight PF1-PF11, an approved charter and a
fresh "clear around the drone" from the site). The limit monitors in the air (battery,
range, area, navigation, height, speed, SD-link loss, a silent AI) work even without the
link to the ground. A silent AI never sends the vehicle home: the step finishes, then it
holds.

| Module | Contents |
|---|---|
| `executor/state.py` | States, triggers and the transition table |
| `executor/guards.py` | The guards that pick a transition's target or refuse |
| `executor/executor.py` | `Executor`: orders, checks, transitions, the tick, snapshots |
| `executor/authority.py` | Compare-and-set and the classification of mode changes |
| `executor/limits.py` | `LimitMonitor` and the effective pilot position |
| `executor/override.py` | STOP / FLY HOME notices, the stop latch, the pilot's takeover |
| `skills/` | The handlers: `takeoff`, `goto`, `survey_sector`, `hold`, `resume`, `return_home` |
| `io/` | `executor_state.json` (restart) and the parameter files (PF6) |
| `link_client.py` | SD-link onboard: inbound orders, outbound Acks, events and snapshots |
| `config.py`, `app.py`, `logs.py` | Configuration, start checks, the process, masked logs |

Unit tests run against an in-memory flight controller (`just test`); the SITL tests are
in `simulation/tests/onboard/` (`just test-sitl`).

Part of the [Searchdrone](../../README.md) monorepo. Licensed under Apache-2.0.
