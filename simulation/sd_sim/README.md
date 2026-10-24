# sd-sim (`sd_sim`)

Simulation tooling: SITL launcher, simulated radio link, MAVLink taps, test rig and scenario runner. Test tooling only; no production package imports it.

Part of the [Searchdrone](../../README.md) monorepo. Licensed under Apache-2.0.

## Commands

| Command | What it does |
|---|---|
| `just sim-setup` | Installs ArduPilot SITL (`~/ardupilot`), Gazebo Harmonic and ArduPilot's Gazebo plugin at the versions in `simulation/versions.lock`. Safe to run again. |
| `just sim-doctor [physics=builtin]` | Checks the environment and that every simulator port listens on 127.0.0.1 only. |
| `just sim-smoke [physics=gz\|builtin] [gui=false] [world=sd_hellasgarden] [vehicle=fpv7]` | GUIDED, arm, take off to 20 m, land, with ArduPilot's quad at CMAC or with `world=` our vehicle in a terrain world; `vehicle=fpv7` flies the 7-inch `sd_fpv7` (in `sd_lake_forest` at CMAC without `world=`). Writes `runs/<run>/result.json` with Gazebo's real-time factor while hovering with the camera on, and the hover current where the physics reports one (builtin); exit 0 when it passed. |
| `just world-snapshot [world=sd_hellasgarden] [gi=off\|vct]` | Pictures from the vehicle's camera at 40 m and 100 m and from the side, with the road's width and the map overlap measured in them, and Gazebo's real-time factor and the graphics card's memory in use while the cameras render (`runs/<run>/snapshots/`). |
| `just ortho-fetch site=<site>` | Fetches the site's orthophoto with your own Geotorget login to `build/ortho/<site>/` (see Photo mode). |
| `just sim-photo [site=hellasgarden] [gi=vct\|off]` | `just sim` in photo mode: the orthophoto on the open ground, the real trees on it, global illumination for the camera. Says what to run when the photo is missing. |
| `just sim-farm [commander=script\|sd-brain] [rounds=N] [max_minutes=N] [seed=N] [idle_only=true] [update=off]` | The sim farm: randomized search rounds from the newest `main`, one after another, until Ctrl+C. Each passed round keeps camera pictures with YOLO and COCO person boxes and a metadata file per picture. Pauses while the GPU or CPU is busy; stops after `max_failures` (3) failed rounds in a row, and at `max_minutes` (the running round is stopped and dropped, exit 0). Settings in `config/simfarm.toml` (copy `config/simfarm.example.toml`). |
| `just sim-farm-status` | What the farm is doing: running or paused and why, pictures and labels kept, rounds per hour, AI-commander exams and their cost, the upload backlog, the latest errors. |
| `just sim-farm-setup` | Fetches the person models (pinned by SHA-256) from Gazebo Fuel into `build/simfarm/assets/`. |
| `just world-fetch site=<site>` | Fetches a site's heights and map layers (see Worlds). |
| `just sim` / `just sim-headless` | Starts the simulator stack for one instance in the foreground (Ctrl-C stops it). |
| `just stop` | Stops what `just sim` started. |
| `just test-sitl [sc=SC00] [physics=builtin\|gz] [mode=pr\|nightly\|search\|search-nightly\|e2e-search] [seeds=1-50] [workers=N] [shard=N shards=M]` | Runs the scenario tests (pytest with `sd_sim.pytest_plugin`). Each test writes `runs/<run>/result.json`; the run writes `runs/summary.md`. The search chain's sets run only in their own modes: `search` (the pull request set), `search-nightly` (the nightly campaign over `seeds=`, `A-B` or `1,2,3`, default 1-3) and `e2e-search` (the search demo). |
| `just test-video [physics=builtin\|gz]` | Video tests against MediaMTX (`just video-up`; the browser test also needs `just db-up migrate` and Google Chrome): publishing password, loopback-only ports, the delay read from the time code strip, recordings per mission, LAN mode. |
| `just ci-sitl [same keys]` | Runs the same tests inside the SITL image, exactly like the `sitl` CI job. |
| `just e2e-search [runs=N] [detector=model] [model=<.json>]` | The search demo end to end in the SITL image with the database from `compose.yaml`, like the `e2e-search` CI job; writes `runs/<run>/search_report.md` and `search_metrics.json`. `detector=model` flies it on this computer in Gazebo with the SK01 persons as models and the owner's trained detector (`sd_sim.detector_choice`, `sd_sim.people_scene`; `tools/train/README.md`). |
| `just demo-search [detector=model] [model=<.json>]` | The e2e-search run (SK01, scripted AI and operator) locally with Gazebo physics and its window. |

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
`build/sim/models/sd_site_<site>/`, without downloads:

- The terrain is a textured mesh every 5 m, in 16 tiles of 300 m with a 2048-pixel texture
  each (about 0.15 m per pixel). The texture is painted from the map: forest floor, meadow,
  wetland, lake, asphalt with edge and centre lines, and gravel paths. With a fetched
  orthophoto the open ground is the photo instead (see Photo mode).
- The real trees, when they are fetched: `just trees-fetch site=<site>` reads every tree's
  position and height from Skogsstyrelsen's tree height raster (Trädhöjd, 1 m, from
  Lantmäteriet's laser scanning) with a local-maximum filter whose radius grows with the
  height, and its species (spruce, pine or deciduous) from Naturvårdsverket's land cover map
  NMD 2023 (CC0). `just tree-pictures-fetch` makes one texture atlas (colour with alpha,
  normal, occlusion-roughness-metallic) from Poly Haven's CC0 tree and bark scans. Trees
  within 150 m of home are card models of 580–950 triangles, the rest 6-triangle
  impostors, in 64 glTF tiles sharing one material. Both go to `build/` only, never into
  the repository (see `ATTRIBUTION`): `sd_hellasgarden` gets about 14 700 trees of up to
  34 m, with a mean height error of 0.3 m against the raster.
- Without them, about 25 000 generated spruce, pine and birch trees (10–26 m) stand in the
  map's forest instead.
- Buildings are the map's footprints at their tagged or typical height, with gabled roofs on
  rectangular houses. Cars of 4.5 × 1.8 × 1.5 m are parked in the parking areas.
- With the real trees the heights are Copernicus's as they are. Copernicus is a surface
  model, but the old fixed 12 m taken off inside the forest made forest lower than the open
  ground beside it, which the raw model did not. Only the generated trees still take
  `canopy_m` off. `[dem] source = "lantmateriet-mhm"` with `surface = "dtm"` uses
  Lantmäteriet's 1 m ground model instead (Geotorget login). Lakes are set below their water
  level, and the pad around home is flat.
- The ground collides as a heightmap, which is cheap enough to keep the real-time factor of
  the flat world. Trees, cars and water are visual only, while buildings and the pad collide.
- The same inputs always give the same files (`python -m sd_sim.terrain site=<site> twice=1`
  checks this). A build is reused until the inputs change.

The world's origin is home on the pad: Gazebo's `<spherical_coordinates>`, SITL's `--home`
(the pad's height above sea level) and the panel's map pack (`just maps-fetch
site=<site>`) all use the same point. No ground station sends SITL terrain data, and terrain following
(`TERRAIN_FOLLOW`, `RTL_ALT_TYPE`) stays at ArduPilot's default, off. Heights are therefore relative to home and the ground under the vehicle rises and falls
with the hills (−5 to +49 m from home in `sd_hellasgarden` with the real trees). The site's fixture
(`build/sim/worlds/<world>.fixture.yaml`) therefore sets the search height at least 25 m
above the highest ground plus the tallest tree under `charter_test`; it is written for
scenarios that will fly in this world (none reads it yet). Return home (RTL) climbs to
ArduPilot's default 15 m above home, so where the ground on the way home lies higher than
that, the simulated vehicle can hit it: give a charter in the terrain world a return
altitude above its highest ground.

Your own area: copy `config/site.example.toml` to `config/site.local.toml`, set your area
there and run `just world-fetch site=<name>` with your Geotorget login (see Photo mode for
where it goes). Lantmäteriet's 1 m height model and orthophoto (CC BY 4.0, © Lantmäteriet)
are written to `build/sites/<name>/` only, and the orthophoto is draped over the ground as
in photo mode. Nothing of it is committed or uploaded.

## Photo mode

The camera sees the real place from above: Lantmäteriet's newest colour orthophoto over the
site (0.16 m per pixel where it has one) on the open ground, the real trees standing on it.

1. Your Geotorget login: copy `config/geotorget.example.toml` to
   `config/geotorget.local.toml` (gitignored), fill in `user` and `password` and run
   `chmod 600 config/geotorget.local.toml`. `SD_LM_USER` and `SD_LM_PASSWORD` in the
   environment win over the file. The login goes to GDAL's environment only; error messages
   have it cut out.
2. `just ortho-fetch site=hellasgarden` (needs `gdal-bin`): the photo is written to
   `build/ortho/hellasgarden/ortho.npy`, with where and when it came from in `ortho.toml`.
   It refuses to run in CI.
3. `just trees-fetch site=hellasgarden` and `just tree-pictures-fetch` once, for the real
   trees.
4. `just sim-photo`.

Under every tree's crown (15 % wider plus 0.5 m, as the photo is not a true orthophoto
and its crowns lean away from where it was taken) the painted forest floor stays, so the
photo's trees do not lie flat under the world's own. Buildings, cars and water stand on
the photo as before. Without `ortho.npy` the world is exactly the same as without photo
mode, file for file; a photo in a public site's committed folder is never used. The
texture size is the same either way (16 tiles of 2048 × 2048, about 270 MB, 360 MB with
mipmaps), so photo mode needs no more graphics memory than the painted ground.

`gi=vct` (the default in `just sim-photo`, also `just sim gi=vct world=<site world>`)
turns on Gazebo's voxel cone tracing for the cameras: light bounces off the ground and the
trees into the shadows. It works with `physics=gz` in a site world only. In a container
without a graphics card (software rendering) it lowered the real-time factor while four
cameras rendered from 0.68 to 0.47 and lit the shadows slightly; measure on your own card
with `just world-snapshot gi=off` and `gi=vct` (the real-time factor and the card's memory
in use go to `metrics.json`) and with `just sim-smoke world=sd_hellasgarden gi=off` and
`gi=vct` (the real-time factor while hovering with the vehicle's camera on, in
`result.json`).

While `ortho.npy` is there every use of the site world has the photo, not only `just
sim-photo` (`just sim`, `just world-snapshot`, `just sim-smoke`), and each says so when it
builds the world; move `build/ortho/<site>/` away for the painted ground CI has.

The photo, the world built from it and pictures of that world stay on your computer:
`tools/check_no_orthophotos.py` fails CI if any of these is in git, its history, an
uploaded artifact or the cache: a picture-shaped NumPy array, a GeoTIFF, a symbolic link,
or a world, snapshot or ground texture marked as built from a photo. It cannot recognise
a photo saved by hand as an ordinary picture with no such mark; no photo is ever fetched
in CI, and images outside the test data are kept out of git anyway. The sim farm never
uploads a round flown over the photo: its facit says so.

## Parts

- `launch` builds the SITL command line (`sitl.build_sitl_argv`), starts each process with a minimal environment and waits for it to be healthy. Process ids go to `build/sim/i<N>/pids.json`, logs to `runs/<run>/logs/`.
- `linksim` is the simulated radio: MAVLink and video over UDP with delay, jitter and loss, SD-link over TCP with delay only. Its control API on 127.0.0.1 needs a bearer token from `build/sim/i<N>/secrets/linksim_token`.
- `testvideo` is the synthetic camera for `video=test` (builtin physics, CI): GStreamer's moving ball with a machine-readable time code strip (`timecode`: the wall clock in ms as 48 black/white boxes at the top of the picture), H.264 over RTP to the camera's port.
- `video_relay` takes the camera's RTP after `linksim` and publishes it to MediaMTX (`rtsp://127.0.0.1:8554/drone/sim-01/main`, instance 0 only). The password is read from `build/sim/i0/secrets/mediamtx_publish`, never from the command line. Without a running MediaMTX (`just video-up`) the launcher skips it.
- `world` renders the Gazebo world and the vehicle model per instance into `build/sim/` (deterministic, offline).
- `fpv7` is the 7-inch `sd_fpv7` (`just sim vehicle=fpv7`; the default stays `sd_quad14`). Its mass, centre of gravity, inertia matrix, rotor, leg and camera positions and its meshes come from the CAD model `hardware/cad/fpv7` through `simulation/gazebo/models/sd_fpv7/cad.json` and `meshes/`, written by `just sim-model-fpv7`; the hw-cad CI job runs `just sim-model-fpv7 --check` and fails when they no longer match the CAD model. The Gazebo links add up to the CAD totals, and the builtin-physics frame file (`sd_fpv7.json`, written into SITL's working directory) takes its mass and inertia from the same file. Its parameter layer is `simulation/sitl/params/sd_fpv7-sim.parm` (X frame, `MOT_SPIN_MAX` 0.90, `ATC_ANGLE_MAX` 35, and the simulated 6S 10 Ah pack through `SIM_BATT_VOLTAGE` and `SIM_BATT_CAP_AH`). The pack size and reserves (`BATT_*`) come from the safety file, as for every model.
- `objects` is the hook for anything a scenario puts in the world for the camera to find (people, decoys, a red test block). Pass `placed=[...]` to `world.render_world`, or `objects=<file.yaml>` to `just sim` (example: `simulation/scenarios/objects/red_block.yaml`). Each object is placed on the ground at metres east/north of home, gets an optional Gazebo semantic label, and its facit (world position, WGS84 position, label and free-form `truth`) is written to `build/sim/worlds/<world>_i<N>.truth.json`. Bring your own class (a person mesh, an `<actor>`) by implementing the `WorldObject` protocol.
- `farm` is the sim farm. `scenario` draws a round from its seed (world, people and decoys, poses, clothes, sun and weather, height); `people` is the catalogue of person models (CC0 and CC BY 4.0, credited in `simulation/farm/people/ATTRIBUTION.md`; no face recognition, and the models are never committed); `camera` and `labels` turn the facit and the camera's pose into boxes, checked against Gazebo's bounding-box camera; `qc` passes or fails a round; `round` is one round in its own process (`spec.json` in, `round.json` out); `loop` runs rounds from a detached worktree at `origin/main` (`build/simfarm/main`) under `nice`; `host` decides when to pause; `store` uploads passed rounds to Cloudflare R2 in the EU jurisdiction when `[r2]` in `config/simfarm.toml` has a key (once per file: a ledger, the object's SHA-256 and a manifest written last), otherwise they stay in `build/simfarm/rounds/`. The decision maker is `commander=script` (the scripted search) or `sd-brain` (the AI commander); when `sd_brain.simeval.farm` exists, each passed round is graded by `python -m sd_brain eval grade --round <dir>`. The Gazebo round itself is not built yet: `just sim-farm` exits 2 until it is.
- `mavtap` records the onboard computer's MAVLink channel without changing it.
- `harness` is the test rig's own MAVLink channel (system 253) on SITL serial0.
- `pytest_plugin` adds the simulator markers and options. Every `sitl` test has `@pytest.mark.budget_s(<s>)`; tests are spread over CI shards by longest budget first, and `test_sitl_budget` fails when a shard would not fit the job.
- `report` writes a scenario's `result.json`: one entry per criterion, with `passed: null` and a note for criteria whose part of the stack is not built yet.
- Ports come from `simulation/ports.yaml`; `simulation/listen_allowlist.yaml` lists the few sockets that ArduPilot and Gazebo bind to all addresses themselves, each with its reason; every other non-loopback socket on a simulator port or owned by a simulator process fails the check.

Run the checks from the repository root: `just check`. The SITL tests (`-m sitl`) run with `just test-sitl` when ArduCopter SITL is built, or with `just ci-sitl` in the image (tag from `tools/sitl_image_tag.sh`).
