# Changelog

All notable changes to this project are listed here. The format follows
[Keep a Changelog](https://keepachangelog.com/en/1.1.0/) and the project uses
[Semantic Versioning](https://semver.org). Hardware revisions have their own tags (`hw-revA`, ...).

## [Unreleased]

### Changed

- The cloud model is the AI commander and the local language model is gone: no
  llama-server in `compose.yaml`, no `just models-fetch`, `just llm-up`, `just check-model`
  or `just brain-eval-model`, and no local decision engine. Without network or key, or
  when the cloud does not answer in time, the fixed command mode answers the same turn;
  the brain's heartbeat, `chat.brain_status` and `/system/status` say which engine
  answers (`cloud` or `command`), and the panel shows "AI:n svarar inte" with fixed
  command buttons. `[cloud]` gains `fallback_model` (asked when the main model errors),
  `reasoning_effort` (default `low`) and `[cloud.provider]` (OpenRouter provider routing,
  `data_collection = "deny"` by default). Defaults: `z-ai/glm-5.3-flash` with
  `anthropic/claude-haiku-5.5` as fallback. The backend kind `llm_local` is now `cloud`
  (old `brain.toml` and `sim.toml` files still load). `just perf` measures the cloud model's answer time when a key
  is set and skips with a line otherwise; `just brain-eval-cloud` runs the chat scenarios
  against it.
- **Breaking (`commander/v1`):** the decision engine `local` is gone. `DecisionLog.engine`
  and `SystemStatus.brain_engine` now report `command` where they reported `local`, and
  readers that matched on `local` must match on `command`.

### Added

- Perception chain: `sd-onboard` takes stills from a camera folder (`[perception]` in
  `config/onboard.toml`, off by default), runs the class register's detector sources,
  places each detection on the ground with the flight controller's position and attitude
  (`ATTITUDE` is now read) and sends a `PerceptionReport` followed by a `CropData` JPEG
  for each target over SD-link, in its own lowest-priority queue. A detection may now
  carry its class (`GeoDetectionMsg.label` and `source`). `sd-ground` queues reports per
  vehicle off the link's inbound loop, waits a moment for a report's crops, stores the
  still and its classed detections (`perception_stills` and `detections`; migration
  `0004_detections` adds `label`, `kind`, `source` and `find_id`), joins them into one
  find per object (rebuilt from the table by find id after a restart), saves thumbnails
  in the mission folder and serves
  `GET /missions/{id}/detections`, `/finds` and `/detections/{id}/crop`; live events
  `detection.new`, `find.new` and `find.updated` on the `detections` topic. The AI
  commander gets `event.detection` for new finds and wakes on targets (trigger
  `detection`); the scripted commander can fly to a find (`at: "event:find"`) and ask the
  pilot to confirm. The panel draws finds on the map and opens a card with the class,
  score, thumbnail and source; the simulator's labels are marked "facit".
- Class register and detector sources (`sd_detect`): a table of classes (`person`,
  `red_block`, `car`, `building`, `road`) with Swedish names, kind, source and plausible
  size; a colour detector on real pixels for red blocks; the simulator's labelled boxes as
  an oracle source (simulator only, marked `oracle`); a size check that drops a red car
  posing as a block. Georeferencing in `sd_geo` (pixel to ground on flat ground and back,
  footprint, pixels per metre, 95 % position error). The SK-RED scene as world objects
  (`sd_sim.targets`, placed with `render_world(placed=)`) and `python -m
  sd_sim.render_stills`, which renders the detector's test pictures headless in Gazebo.
- A simulator world from a real place: `sd_hellasgarden`, 1.2 × 1.2 km of the public
  Hellasgården outdoor area (Nacka), built from Copernicus DEM GLO-30 heights and
  OpenStreetMap layers (`simulation/gazebo/sites/hellasgarden/`). It has textured terrain
  with roads and road markings, about 25 000 spruce, pine and birch trees, buildings from
  the map's footprints, parked cars and lakes. It is the default for `just sim`;
  `just demo`, `brain=scripted` and the test suites keep the flat world at CMAC. Gazebo,
  SITL's home and the panel's map pack (`just maps-fetch site=hellasgarden`) share one origin.
  `just world-fetch` refreshes a site, or fetches your own area from Lantmäteriet into
  `build/` only. `just world-snapshot` takes pictures at 40 m and 100 m and from the side,
  and measures a road's width and the map overlap in them. `just sim-smoke world=...`
  flies in a terrain world and reports Gazebo's real-time factor.
- Fly a drawn path: the pilot draws a path on the map ("Rita väg", `pa-1`, ...: a
  `path` annotation, a GeoJSON LineString of 2-50 points) and asks for it in the chat. The
  AI proposes `follow_path` by the path's name only; sd-brain takes the points from the
  drawn path, so the model never writes coordinates. The order always needs the pilot's
  click, and the drone flies the points as an AUTO mission and then hovers at the last
  one. The gate checks the whole line against the charter, the height and speed limits,
  a 5 km cap (`PATH_TOO_LONG`) and the battery (`ROUTE_BATTERY`: the path, the flight
  home and the landing must leave the reserve, at an assumed 5 % per minute), with the
  numbers in the Swedish message. The panel shows drawn paths, the planned path
  (dashed), the flown part (solid) and the progress in %, and refuses a path outside the
  search area before saving it. SITL scenarios SC_PATH (every point within 3 m, then
  hover) and SC_PATH_stop (STOP halfway: BRAKE at once, standing still within 5 s as
  SC02).
- Scenario objects in the simulator world (`sd_sim.objects`): boxes from a YAML file
  (`just sim objects=<file>`) or any `WorldObject` from code are placed on the ground at
  metres east/north of home, can carry a Gazebo semantic label, and their facit (world and
  WGS84 position) is written to `build/sim/worlds/<world>_i<N>.truth.json`.
- Status page on GitHub Pages (`.github/workflows/status-page.yml`, `tools/status_page/`):
  one row per capability (flight, safety, AI, link, video) with its expected and measured
  value, scenario, time and commit, the last 20 runs per capability and a demo film from
  the panel's SITL browser test. Rebuilt after every push to `main` and nightly. Every
  SITL and video test now writes `runs/<run>/result.json` with the values it measured.
- Live video and recordings: MediaMTX 1.21.1 in `compose.yaml` (`just video-up`) with
  `config/mediamtx.yml`, every listener on 127.0.0.1 (only the WebRTC media port moves
  to the LAN in LAN mode, Media over QUIC off) and publishing only with the password
  from `.env`. The simulator's test picture carries a time code strip; `video_relay`
  publishes the camera through the simulated radio. The ground station proxies WHEP
  (`/api/v1/video/{vehicle_id}/main/whep`), reports the video server in `/health` and puts
  each mission's video segments into its folder. The panel's video view plays WebRTC, falls
  back to native HLS (Safari, iPhone) once the server serves HLS, and shows "no video"
  with a retry. `just test-video` measures the delay in
  Google Chrome from the strip.
- AI mission commander `sd-brain`: chat and situation in, one gated order per turn out.
  The cloud model (any OpenAI-compatible API, OpenRouter by default) decides when a key is
  set, and the local Qwen3-VL-4B in llama-server takes over automatically in the same turn.
  Charter proposals from the chat, the take-off rule (approved charter, "clear around the
  drone" and the operator's request), a fixed Swedish command mode while the model is down,
  a scripted commander for tests, the decision interface with a quick stop check on every
  chat line (local model, Jev, OpenAI Decisions), 33 chat scenarios with an answer key,
  `just models-fetch`, `just llm-up`, `just brain`, `just check-model` and `just brain-token`.
- Onboard computer `sd-onboard` (`sd_onboard`): the executor that takes skill orders over
  SD-link, checks them again onboard (the shared safety gate on the handler's own route,
  plus its own checks: stale orders, approvals, the stop latch, takeoff preconditions with
  a fresh "clear around the drone") and flies them through `sd_mav`. Implemented skills:
  `takeoff`, `goto`, `survey_sector` (lawnmower), `hold`, `resume` and `return_home`. A
  state machine with compare-and-set on the flight mode: a mode change from outside, a
  flight-controller failsafe or a STOP/FLY HOME notice makes it let go at once and send
  nothing that steers until the pilot's Resume. Limit monitors in the air (battery,
  range, area, navigation, height, speed, SD-link loss, a silent AI), preflight checks,
  restart safety (HELD when armed or in the air) and a state snapshot at 2 Hz. SITL tests
  for the demo flight, an outside mode change and SD-link loss.
- ArduCopter safety parameters (`simulation/sitl/params/searchdrone-safety.parm`):
  failsafes, battery thresholds, the fence just outside the hard limits, no stick arming
  (`ARMING_RUDDER 0`) and stick values that expire after 1 s (`RC_OVERRIDE_TIME 1`).
- `sd_geo`: the lawnmower search pattern (concave areas too) and the coverage measure.
- `sd_testing`: a test-only package with the project's one fake clock.
- Vehicle link (`sd_mav`): MAVLink to the flight controller for the onboard computer and
  the ground station, behind a default-deny allowlist with number checks against the
  mission's limits and a deny list of the most dangerous messages (no parameter writes,
  no geofence switch, no forced arming). Telemetry, stream requests, lost-link detection
  within 3 s, mission and fence upload with read-back hashes, failsafe signals and an
  in-memory fake flight controller. SITL tests for connection, telemetry rate, missions
  and link loss.
- Monorepo skeleton: uv workspace for the Python packages, pnpm workspace for TypeScript,
  `justfile` with every recipe, pre-commit hooks, import contracts and example configuration.
- Forbidden-file guard that blocks planning files, agent files, secrets, personal config,
  media, large files, coordinates outside public test sites and private search patterns.
- `tools/bootstrap.sh` and `tools/setup/wsl-setup.sh` for a fresh WSL2 Ubuntu 24.04.
- Simulator environment: `just sim-setup` installs ArduPilot SITL (Copter-4.7.1) and Gazebo
  Harmonic, `just sim-doctor` checks the setup, `just sim-smoke` flies a take-off and landing.
  Simulated radio link (`linksim`), MAVLink tap, test rig, SITL container image and
  `simulation/ports.yaml`. Simulator ports listen on 127.0.0.1; the few sockets ArduPilot
  and Gazebo bind to all addresses themselves are listed with their reasons and checked.
- Project documents: package READMEs, security policy, contributing guide, code of conduct,
  issue templates and this changelog.
- Shared message schemas (`sd_schemas`): ids, limits and their outer cap, mission charter,
  gate input and output, mission states, SD-link messages (including manual stick control),
  panel and commander API models, reason codes with Swedish texts, and the decision
  interface. Generated JSON Schema and TypeScript types (`@searchdrone/schemas`) via
  `just gen`, checked by `just gen-check`.
- Skill registry with all 18 skills, every one a stub until its executor is built.
- Local Postgres 18 + PostGIS 3.6 in `compose.yaml` (loopback port 5433, image pinned by
  digest), `just db-up` / `db-down` / `db-check`, and the first Alembic migration with the
  ground station tables, append-only guards and the restricted `sd_app` role.
- Ground station server `sd-ground`: panel API and live WebSocket on `127.0.0.1:8080`
  (TLS and PIN pairing on `:8443` only in LAN mode), commander API for the AI on
  `127.0.0.1:7700`, one-time login links and PINs, sessions that never expire mid-flight,
  browser protection (Host allow-list, CSRF header, Origin check, no CORS, security
  headers, body limits), STOP / RETURN HOME handled in their own ingress thread, telemetry
  to the panel at 5 Hz and into the database at 2 Hz, the `sd-ground` command line and
  `just migrate` / `just ground`. Tests can only reach loopback.
- Web panel shell (`software/panel`): chat, map, video, candidates, status and manual-control
  views that can be combined or shown in full screen, tabs and picture-in-picture on phones,
  and the always-visible STOP / RETURN HOME bar with honest acknowledgement, retries and
  polling. Placeholders for take-over, on-screen sticks, game controller and order buttons.
  Playwright tests against a fake ground station; works fully offline.
- CI on every pull request and push to `main`: lint, type checks, unit and contract tests,
  license policy, REUSE, secret scanning and build, summed up in the required check `ci-ok`.
  Code and dependency scanning (CodeQL, OSV-Scanner) in `security-ok`, the commit statuses
  `gate/pr-title` and `gate/review`, Dependabot and a pull request template.
- Safety gate (`sd_safety.evaluate`): every rule for orders from the AI and the operator,
  exact route and area checks against concave charter areas, geodesic distances, limits
  with an outer cap and the matching flight-controller fence, manual control and the
  flight-controller fence radius as reasons to reject.
- Hardware gate: `hardware/drone.yaml` and `hardware/limits.yaml` (simulated quad, about
  2.3 kg and 33 min hover), `sd_hwcalc` with `just hw-check` for mass, hover power, hover
  time and thrust margins, and the required check `hardware-ok`. Every other file under
  `hardware/` fails closed until its own check exists. Changes to `hardware/` and
  `.github/CODEOWNERS` also need the owner's OK on the exact head commit (`gate/owner`).
- Flight tests in CI without graphics: `sitl.yml` runs the SITL scenario tests (builtin
  physics) in the SITL image on every pull request, summed up in the required check
  `sitl-ok`, plus nightly suites and a Gazebo server run. `sitl-image.yml` builds the image,
  scans its file system for secrets and only then pushes it to GHCR from `main`. The image
  tag comes from `tools/sitl_image_tag.sh`. `just test-sitl` runs the scenario tests,
  `just ci-sitl` runs them in the image like CI; the first scenario is the SC00 smoke test.
- Review instructions for the three reviewer roles (architecture, IT security, flight
  safety) in `.github/review/`, the list of safety-relevant paths every reviewer must read
  (`critical_paths.txt`), and instructions for a retrospective review of merged code.

### Changed

- The panel keeps a drawn path on the map (planned, flown part and progress) while the
  pilot's Resume flies the rest of it after STOP, and shows it as flown when the drone
  hovers at its end.
- The onboard computer reads the flight controller's fence back right before every
  takeoff and refuses with `FENCE_NOT_VERIFIED` when it differs from the charter's.
- When the onboard battery rule flies home by itself, the mission returns (cause
  battery), the pilot gets the `BATTERY_RTL` alert, open orders are cancelled and the
  AI's orders meet the failsafe.
- The ground keeps the flight controller's failsafe texts (a fence breach comes as a
  notice) at any severity.
- A ready-made charter proposal builds on the charter in force.
- The safety parameters pin `AVOID_ENABLE 3` (ArduPilot's default, braking before the
  fence), so the onboard computer reads it back before every takeoff. SC16b runs on its own
  SITL with `AVOID_ENABLE 2` (`simulation/sitl/params/tests/fence_no_brake.parm`): the
  climb no longer stops just below `FENCE_ALT_MAX`, so the height fence is breached every run.

- One clock abstraction: SD-link takes its time from `sd_obs.clock.Clock`; the two fake
  clocks are merged into `sd_testing.FakeClock`, which production packages may not import.
- The SD-link client refuses to send its token over `ws://` to anything but loopback.
- `sd_mav.Vehicle` takes an optional send gate that the owner of the link (the onboard
  executor) asks before every single write, retries included.
- The six onboard skills of the foundation are `implemented` in the skill registry.

### Fixed

- The Gazebo model `sd_quad14` can take off: each rotor now has its own two lift surfaces
  (they all sat on rotors 0 and 1, where the clockwise pair cancelled the lift). The
  vehicle now starts with its legs on the ground in every world; in a site world they were
  inside the landing pad's box and stuck to it.
