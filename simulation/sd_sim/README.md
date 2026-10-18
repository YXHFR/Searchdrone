# sd-sim (`sd_sim`)

Simulation tooling: SITL launcher, simulated radio link, MAVLink taps, test rig and scenario runner. Test tooling only; no production package imports it.

Part of the [Searchdrone](../../README.md) monorepo. Licensed under Apache-2.0.

## Commands

| Command | What it does |
|---|---|
| `just sim-setup` | Installs ArduPilot SITL (`~/ardupilot`), Gazebo Harmonic and ArduPilot's Gazebo plugin at the versions in `simulation/versions.lock`. Safe to run again. |
| `just sim-doctor [physics=builtin]` | Checks the environment and that every simulator port listens on 127.0.0.1 only. |
| `just sim-smoke [physics=gz\|builtin] [gui=false]` | GUIDED, arm, take off to 20 m, land. Writes `runs/<run>/result.json`; exit 0 when it passed. |
| `just sim` / `just sim-headless` | Starts the whole stack for one instance in the foreground (Ctrl-C stops it): Postgres, MediaMTX, `sd-ground`, SITL, `linksim`, `sd-onboard`, `sd-brain` and the video, then prints the panel's one-time login link and how long the start took. |
| `just demo [lan=1]` | The same, locked to the public CMAC home of `simulation/scenarios/demo_fixture.yaml` (no local `config/sim.toml`). Every service must be ready within 180 s. |
| `just e2e [runs=N] [offline=1]` | The demo (SC01) on the whole stack as separate programs, then the emergency buttons (SC02, SC03, SC15, SC08), headless with the scripted commander. Uses the SITL image when SITL is not built here. `offline=1` first shows the internet cannot be reached. Writes `runs/e2e-summary.json`. |
| `just stop` | Stops what `just sim` started. |
| `just test-sitl [sc=SC00] [physics=builtin\|gz] [mode=pr\|nightly] [workers=N] [shard=N shards=M]` | Runs the scenario tests (pytest with `sd_sim.pytest_plugin`). Each test writes `runs/<run>/result.json`; the run writes `runs/summary.md`. |
| `just test-video [physics=builtin\|gz]` | Video tests against MediaMTX (`just video-up`; the browser test also needs `just db-up migrate` and Google Chrome): publishing password, loopback-only ports, the delay read from the time code strip, recordings per mission, LAN mode. |
| `just ci-sitl [same keys]` | Runs the same tests inside the SITL image, exactly like the `sitl` CI job. |

## Parts

- `launch` builds the SITL command line (`sitl.build_sitl_argv`), starts each process with a minimal environment and waits for it to be healthy. Process ids go to `build/sim/i<N>/pids.json`, logs to `runs/<run>/logs/`.
- `services` starts the real programs around SITL (`sd-ground serve`, `sd-onboard`, `sd-brain run`) with this instance's ports and files as `SD_<PART>__<KEY>` variables, makes a new commander token and SD-link token per run, and creates the instance's database (instance 0: `searchdrone`; others: `searchdrone_i<N>` and their own data folder). The local model server is never started: the AI commander uses the cloud model when its key is in `.env`, otherwise its command mode.
- `rc` plays the pilot's RC transmitter in SITL (throttle centred in the air, down on the ground); the launcher's observer holds it.
- `e2e` runs the end-to-end suite; `mp4` reads the codec and length of a recorded MP4.
- `linksim` is the simulated radio: MAVLink and video over UDP with delay, jitter and loss, SD-link over TCP with delay only. Its control API on 127.0.0.1 needs a bearer token from `build/sim/i<N>/secrets/linksim_token`.
- `testvideo` is the synthetic camera for `video=test` (builtin physics, CI): GStreamer's moving ball with a machine-readable time code strip (`timecode`: the wall clock in ms as 48 black/white boxes at the top of the picture), H.264 over RTP to the camera's port.
- `video_relay` takes the camera's RTP after `linksim` and publishes it to MediaMTX (`rtsp://127.0.0.1:8554/drone/sim-01/main`, instance 0 only). The password is read from `build/sim/i0/secrets/mediamtx_publish`, never from the command line. Without a running MediaMTX (`just video-up`) the launcher skips it.
- `mavtap` records the onboard computer's MAVLink channel without changing it.
- `harness` is the test rig's own MAVLink channel (system 253) on SITL serial0.
- `pytest_plugin` adds the simulator markers and options. Every `sitl` test has `@pytest.mark.budget_s(<s>)`; tests are spread over CI shards by longest budget first, and `test_sitl_budget` fails when a shard would not fit the job.
- `report` writes a scenario's `result.json`: one entry per criterion, with `passed: null` and a note for criteria whose part of the stack is not built yet.
- Ports come from `simulation/ports.yaml`; `simulation/listen_allowlist.yaml` lists the few sockets that ArduPilot and Gazebo bind to all addresses themselves, each with its reason; every other non-loopback socket on a simulator port or owned by a simulator process fails the check.

Run the checks from the repository root: `just check`. The SITL tests (`-m sitl`) run with `just test-sitl` when ArduCopter SITL is built, or with `just ci-sitl` in the image (tag from `tools/sitl_image_tag.sh`).
