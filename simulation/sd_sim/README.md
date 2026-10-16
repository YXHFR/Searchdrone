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

## Parts

- `launch` builds the SITL command line (`sitl.build_sitl_argv`), starts each process with a minimal environment and waits for it to be healthy. Process ids go to `build/sim/i<N>/pids.json`, logs to `runs/<run>/logs/`.
- `linksim` is the simulated radio: MAVLink and video over UDP with delay, jitter and loss, SD-link over TCP with delay only. Its control API on 127.0.0.1 needs a bearer token from `build/sim/i<N>/secrets/linksim_token`.
- `mavtap` records the onboard computer's MAVLink channel without changing it.
- `harness` is the test rig's own MAVLink channel (system 253) on SITL serial0.
- Ports come from `simulation/ports.yaml`; `simulation/listen_allowlist.yaml` lists the only socket allowed outside loopback, with the reason.

Run the checks from the repository root: `just check`. The SITL smoke test (`-m sitl`) runs when ArduCopter SITL is built.
