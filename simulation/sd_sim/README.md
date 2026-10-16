# sd-sim (`sd_sim`)

Simulation tooling: SITL launcher, simulated radio link, MAVLink taps, test rig and scenario runner. Test tooling only; no production package imports it.

Part of the [Searchdrone](../../README.md) monorepo. Licensed under Apache-2.0.

## Commands

| Command | What it does |
|---|---|
| `just sim-setup` | Installs ArduPilot SITL (`~/ardupilot`), Gazebo Harmonic and ArduPilot's Gazebo plugin at the versions in `simulation/versions.lock`. Safe to run again. |
| `just sim-doctor [physics=builtin]` | Checks the environment and that every simulator port listens on 127.0.0.1 only. |
| `just sim-smoke [physics=gz\|builtin] [gui=false]` | GUIDED, arm, take off to 20 m, land. Writes `runs/<run>/result.json`; exit 0 when it passed. |
| `just sim` / `just sim-headless` | Starts the simulator stack for one instance in the foreground (Ctrl-C stops it). |
| `just stop` | Stops what `just sim` started. |
| `just test-sitl [sc=SC00] [physics=builtin\|gz] [mode=pr\|nightly] [workers=N] [shard=N shards=M]` | Runs the scenario tests (pytest with `sd_sim.pytest_plugin`). Each test writes `runs/<run>/result.json`; the run writes `runs/summary.md`. |
| `just ci-sitl [same keys]` | Runs the same tests inside the SITL image, exactly like the `sitl` CI job. |

## Parts

- `launch` builds the SITL command line (`sitl.build_sitl_argv`), starts each process with a minimal environment and waits for it to be healthy. Process ids go to `build/sim/i<N>/pids.json`, logs to `runs/<run>/logs/`.
- `linksim` is the simulated radio: MAVLink and video over UDP with delay, jitter and loss, SD-link over TCP with delay only. Its control API on 127.0.0.1 needs a bearer token from `build/sim/i<N>/secrets/linksim_token`.
- `mavtap` records the onboard computer's MAVLink channel without changing it.
- `harness` is the test rig's own MAVLink channel (system 253) on SITL serial0.
- `pytest_plugin` adds the simulator markers and options. Every `sitl` test has `@pytest.mark.budget_s(<s>)`; tests are spread over CI shards by longest budget first, and `test_sitl_budget` fails when a shard would not fit the job.
- `report` writes a scenario's `result.json`: one entry per criterion, with `passed: null` and a note for criteria whose part of the stack is not built yet.
- Ports come from `simulation/ports.yaml`; `simulation/listen_allowlist.yaml` lists the few sockets that ArduPilot and Gazebo bind to all addresses themselves, each with its reason; every other non-loopback socket on a simulator port or owned by a simulator process fails the check.

Run the checks from the repository root: `just check`. The SITL tests (`-m sitl`) run with `just test-sitl` when ArduCopter SITL is built, or with `just ci-sitl` in the image (tag from `tools/sitl_image_tag.sh`).
