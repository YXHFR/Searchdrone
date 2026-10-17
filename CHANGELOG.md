# Changelog

All notable changes to this project are listed here. The format follows
[Keep a Changelog](https://keepachangelog.com/en/1.1.0/) and the project uses
[Semantic Versioning](https://semver.org). Hardware revisions have their own tags (`hw-revA`, ...).

## [Unreleased]

### Added

- Live video and recordings: MediaMTX 1.21.1 in `compose.yaml` (`just video-up`) with
  `config/mediamtx.yml`, every listener on 127.0.0.1 (only the WebRTC media port moves
  to the LAN in LAN mode, Media over QUIC off) and publishing only with the password
  from `.env`. The simulator's test picture carries a time code strip; `video_relay`
  publishes the camera through the simulated radio. The ground station proxies WHEP
  (`/api/v1/video/{vehicle_id}/main/whep`), reports the video server in `/health` and puts
  each mission's video segments into its folder. The panel's video view plays WebRTC, falls
  back to HLS and shows "no video" with a retry. `just test-video` measures the delay in
  Google Chrome from the strip.
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

- One clock abstraction: SD-link takes its time from `sd_obs.clock.Clock`; the two fake
  clocks are merged into `sd_testing.FakeClock`, which production packages may not import.
- The SD-link client refuses to send its token over `ws://` to anything but loopback.
- `sd_mav.Vehicle` takes an optional send gate that the owner of the link (the onboard
  executor) asks before every single write, retries included.
- The six onboard skills of the foundation are `implemented` in the skill registry.
