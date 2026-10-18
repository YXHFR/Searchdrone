# sd-sim (`sd_sim`)

Simulation tooling: SITL launcher, simulated radio link, MAVLink taps, test rig and scenario runner. Test tooling only; no production package imports it.

Part of the [Searchdrone](../../README.md) monorepo. Licensed under Apache-2.0.

## Commands

| Command | What it does |
|---|---|
| `just sim-setup` | Installs ArduPilot SITL (`~/ardupilot`), Gazebo Harmonic and ArduPilot's Gazebo plugin at the versions in `simulation/versions.lock`. Safe to run again. |
| `just sim-doctor [physics=builtin]` | Checks the environment and that every simulator port listens on 127.0.0.1 only. |
| `just sim-smoke [physics=gz\|builtin] [gui=false] [world=sd_hellasgarden]` | GUIDED, arm, take off to 20 m, land, with ArduPilot's quad at CMAC or with `world=` our vehicle in a terrain world. Writes `runs/<run>/result.json` with Gazebo's real-time factor while hovering with the camera on; exit 0 when it passed. |
| `just world-snapshot [world=sd_hellasgarden]` | Pictures from the vehicle's camera at 40 m and 100 m and from the side, with the road's width and the map overlap measured in them (`runs/<run>/snapshots/`). |
| `just sim-farm [commander=script\|sd-brain] [rounds=N] [max_minutes=N] [seed=N] [idle_only=true] [update=off]` | The sim farm: randomized search rounds from the newest `main`, one after another, until Ctrl+C. Each passed round keeps camera pictures with YOLO and COCO person boxes and a metadata file per picture. Pauses while the GPU or CPU is busy; stops after `max_failures` (3) failed rounds in a row, and at `max_minutes` (the running round is stopped and dropped, exit 0). Settings in `config/simfarm.toml` (copy `config/simfarm.example.toml`). |
| `just sim-farm-status` | What the farm is doing: running or paused and why, pictures and labels kept, rounds per hour, AI-commander exams and their cost, the upload backlog, the latest errors. |
| `just sim-farm-setup` | Fetches the person models (pinned by SHA-256) from Gazebo Fuel into `build/simfarm/assets/`. |
| `just world-fetch site=<site>` | Fetches a site's heights and map layers (see Worlds). |
| `just sim` / `just sim-headless` | Starts the simulator stack for one instance in the foreground (Ctrl-C stops it). |
| `just stop` | Stops what `just sim` started. |
| `just test-sitl [sc=SC00] [physics=builtin\|gz] [mode=pr\|nightly] [workers=N] [shard=N shards=M]` | Runs the scenario tests (pytest with `sd_sim.pytest_plugin`). Each test writes `runs/<run>/result.json`; the run writes `runs/summary.md`. |
| `just test-video [physics=builtin\|gz]` | Video tests against MediaMTX (`just video-up`; the browser test also needs `just db-up migrate` and Google Chrome): publishing password, loopback-only ports, the delay read from the time code strip, recordings per mission, LAN mode. |
| `just ci-sitl [same keys]` | Runs the same tests inside the SITL image, exactly like the `sitl` CI job. |

## Worlds

| World | Where | Used by |
|---|---|---|
| `sd_hellasgarden` | A real place: the public Hellasgården outdoor area in Nacka, 1.2 × 1.2 km around its home pad | `just sim` (default), `just sim-smoke world=sd_hellasgarden`, `just world-snapshot` |
| `sd_lake_forest` | A flat made-up world at ArduPilot's public SITL site (CMAC) | `just demo` and the test suites (`just test-sitl`, the model and video tests) |

`just demo` and `brain=scripted` stay in the flat world: the scripted commander's scenarios
(SC01 and the scenario tests) have pins, charters and a 40 m search height for that world,
and the launcher refuses `brain=scripted` in a terrain world. A local site (your own area)
counts as a personal home: `just sim` warns, and the demo refuses it.

A world from a real site is built from three files in `simulation/gazebo/sites/<site>/`:
`site.toml` (home, size, height source), `dem.asc` (heights every 20 m from Copernicus
DEM GLO-30) and `osm.geojson` (forest, water, open land, roads, buildings and parking from
OpenStreetMap). `just world-fetch site=<site>` fetches the last two again (needs GDAL's
`gdal-bin` and internet). `sd_sim.terrain` turns them into a Gazebo model in
`build/sim/models/sd_site_<site>/`, without downloads and without third-party textures or
meshes:

- The terrain is a textured mesh every 5 m, in 16 tiles of 300 m with a 2048-pixel texture
  each (about 0.15 m per pixel). The texture is painted from the map: forest floor, meadow,
  wetland, lake, asphalt with edge and centre lines, and gravel paths.
- About 25 000 spruce, pine and birch trees stand on the ground at real heights (10–26 m).
- Buildings are the map's footprints at their tagged or typical height, with gabled roofs on
  rectangular houses. Cars of 4.5 × 1.8 × 1.5 m are parked in the parking areas.
- The heights come from a surface model, so the forest canopy (`canopy_m`, 12 m) is taken
  off inside the forest. Lakes are set below their water level, and the pad around home is
  flat.
- The ground collides as a heightmap, which is cheap enough to keep the real-time factor of
  the flat world. Trees, cars and water are visual only, while buildings and the pad collide.
- The same inputs always give the same files (`python -m sd_sim.terrain site=<site> twice=1`
  checks this). A build is reused until the inputs change.

The world's origin is home on the pad: Gazebo's `<spherical_coordinates>`, SITL's `--home`
(the pad's height above sea level) and the panel's map pack (`just maps-fetch
site=<site>`) all use the same point. No ground station sends SITL terrain data, and terrain following
(`TERRAIN_FOLLOW`, `RTL_ALT_TYPE`) stays at ArduPilot's default, off. Heights are therefore relative to home and the ground under the vehicle rises and falls
with the hills (−13 to +37 m from home in `sd_hellasgarden`). The site's fixture
(`build/sim/worlds/<world>.fixture.yaml`) therefore sets the search height at least 25 m
above the highest ground plus the tallest tree under `charter_test`; it is written for
scenarios that will fly in this world (none reads it yet). Return home (RTL) climbs to
ArduPilot's default 15 m above home, so where the ground on the way home lies higher than
that, the simulated vehicle can hit it: give a charter in the terrain world a return
altitude above its highest ground.

Your own area: copy `config/site.example.toml` to `config/site.local.toml`, set your area
there and run `SD_LM_USER=... SD_LM_PASSWORD=... just world-fetch site=<name>` with your
Geotorget login (read only from the environment, never from a file). Lantmäteriet's 1 m height model and
orthophoto (CC BY 4.0, © Lantmäteriet) are written to `build/sites/<name>/` only, and the
orthophoto becomes the ground's texture. Nothing of it is committed or uploaded.

## Parts

- `launch` builds the SITL command line (`sitl.build_sitl_argv`), starts each process with a minimal environment and waits for it to be healthy. Process ids go to `build/sim/i<N>/pids.json`, logs to `runs/<run>/logs/`.
- `linksim` is the simulated radio: MAVLink and video over UDP with delay, jitter and loss, SD-link over TCP with delay only. Its control API on 127.0.0.1 needs a bearer token from `build/sim/i<N>/secrets/linksim_token`.
- `testvideo` is the synthetic camera for `video=test` (builtin physics, CI): GStreamer's moving ball with a machine-readable time code strip (`timecode`: the wall clock in ms as 48 black/white boxes at the top of the picture), H.264 over RTP to the camera's port.
- `video_relay` takes the camera's RTP after `linksim` and publishes it to MediaMTX (`rtsp://127.0.0.1:8554/drone/sim-01/main`, instance 0 only). The password is read from `build/sim/i0/secrets/mediamtx_publish`, never from the command line. Without a running MediaMTX (`just video-up`) the launcher skips it.
- `world` renders the Gazebo world and the vehicle model per instance into `build/sim/` (deterministic, offline).
- `objects` is the hook for anything a scenario puts in the world for the camera to find (people, decoys, a red test block). Pass `placed=[...]` to `world.render_world`, or `objects=<file.yaml>` to `just sim` (example: `simulation/scenarios/objects/red_block.yaml`). Each object is placed on the ground at metres east/north of home, gets an optional Gazebo semantic label, and its facit (world position, WGS84 position, label and free-form `truth`) is written to `build/sim/worlds/<world>_i<N>.truth.json`. Bring your own class (a person mesh, an `<actor>`) by implementing the `WorldObject` protocol.
- `farm` is the sim farm. `scenario` draws a round from its seed (world, people and decoys, poses, clothes, sun and weather, height); `people` is the catalogue of person models (CC0 and CC BY 4.0, credited in `simulation/farm/people/ATTRIBUTION.md`; no face recognition, and the models are never committed); `camera` and `labels` turn the facit and the camera's pose into boxes, checked against Gazebo's bounding-box camera; `qc` passes or fails a round; `round` is one round in its own process (`spec.json` in, `round.json` out); `loop` runs rounds from a detached worktree at `origin/main` (`build/simfarm/main`) under `nice`; `host` decides when to pause; `store` uploads passed rounds to Cloudflare R2 in the EU jurisdiction when `[r2]` in `config/simfarm.toml` has a key (once per file: a ledger, the object's SHA-256 and a manifest written last), otherwise they stay in `build/simfarm/rounds/`. The decision maker is `commander=script` (the scripted search) or `sd-brain` (the AI commander); when `sd_brain.simeval.farm` exists, each passed round is graded by `sd-brain eval grade --round <dir>`. The Gazebo round itself is not built yet: `just sim-farm` exits 2 until it is.
- `mavtap` records the onboard computer's MAVLink channel without changing it.
- `harness` is the test rig's own MAVLink channel (system 253) on SITL serial0.
- `pytest_plugin` adds the simulator markers and options. Every `sitl` test has `@pytest.mark.budget_s(<s>)`; tests are spread over CI shards by longest budget first, and `test_sitl_budget` fails when a shard would not fit the job.
- `report` writes a scenario's `result.json`: one entry per criterion, with `passed: null` and a note for criteria whose part of the stack is not built yet.
- Ports come from `simulation/ports.yaml`; `simulation/listen_allowlist.yaml` lists the few sockets that ArduPilot and Gazebo bind to all addresses themselves, each with its reason; every other non-loopback socket on a simulator port or owned by a simulator process fails the check.

Run the checks from the repository root: `just check`. The SITL tests (`-m sitl`) run with `just test-sitl` when ArduCopter SITL is built, or with `just ci-sitl` in the image (tag from `tools/sitl_image_tag.sh`).
