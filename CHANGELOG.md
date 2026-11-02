# Changelog

All notable changes to this project are listed here. The format follows
[Keep a Changelog](https://keepachangelog.com/en/1.1.0/) and the project uses
[Semantic Versioning](https://semver.org). Hardware revisions have their own tags (`hw-revA`, ...).

## [Unreleased]

### Added

- Two training-only sets of real pictures for the person detector (the owner's
  decision of 2026-11-01): KIOS Aerial Dataset for Car and Person Detection (University
  of Cyprus, Zenodo 21216709, CC BY 4.0; people on a parking lot, every tenth video
  frame) and SeaDronesSee Object Detection v2 (University of Tübingen, CC0 1.0;
  swimmers in open water). Fetched only by name (`just bench-fetch yes=1
  sets=kios_carperson,seadronessee`) and trained on with `just train-detector
  sets=heridal,kios,kios_carperson,seadronessee`. They are never measured on: the new
  split `tools/bench/splits/v3.json` keeps every v2 row as it was and gives the two
  sets only train and val parts, split by video. The register now takes a set in
  several archives (`more_archives`), zips inside an archive, labels in one COCO file
  for all pictures, and a set of loose files on a WebDAV share pinned by one SHA-256
  over all of them. `just bench-fetch yes=1` without `sets=` now fetches only the
  measured sets.
- `just dev`: one command for the local development stack. It starts the database,
  `sd-ground`, `sd-onboard`, `sd-search`, the scripted AI commander and the panel's Vite
  dev server (hot reload), then prints the login link. The default `physics=mock`
  replaces ArduPilot with the new `sd_sim.mockfc`, sd_mav's fake flight controller on
  SITL's ports, so it needs no simulator build and no API key. Only `just dev` accepts
  `physics=mock`, so no smoke flight, demo or test can pass without ArduPilot;
  `physics=builtin` or `gz` flies SITL as `just sim` does. The panel's dev server port follows the instance
  (`SD_PANEL_PORT`, 5173 + 10N).
- `tools/check_forbidden_mavlink.py` (in `just lint`): reads the repository's Python and
  fails when code outside the MAVLink gate `sd_mav` names a message or command from its
  deny list (parameter writes, switching off the fence, flight termination, forced
  arming with 21196 ...), also under an import alias, as a `getattr` string or by
  message id (`MAVLINK_MSG_ID_*`, `mavlink_map[...]`), or when production code could
  write on a pymavlink connection itself instead of through `Vehicle.send`
  (`pymavlink.mavutil`, also through `importlib`/`__import__`, `mavlink_connection`, a
  raw `MAVLink` encoder, `mavlink_map`, `_conn` or `write_message` anywhere,
  `pymavlink.mavutil` as a string (`sys.modules[...]`), any message's
  `*_send`/`*_encode` method, `send` on a connection's `mav` (also assigned to a name or
  attribute first, or with `:=`), mavfile helpers such as `reboot_autopilot` or `set_mode`, called, only
  referenced or as a `getattr` string). Tests outside `src/` packages are not read; of
  the denied messages only the simulator harness may send `PARAM_SET`: from a
  `set_sim_param` that first refuses any name but `SIM_*` and then sends that name (if
  it is never bound again) or a literal `SIM_*` one, or elsewhere in the harness with a
  literal `SIM_*` name (a `param_set_send` on any owner but a literal `<x>.mav` needs it
  in every place it may sit: first, as mavfile's, and third, as the dialect's; a
  `*args`/`**kwargs` that may fill it is refused). The mavfile helpers that send
  denied traffic themselves (`reboot_autopilot`, `set_servo`, `calibrate_level` ...)
  are refused everywhere outside the gate. Names built at run time (f-strings,
  arithmetic) are not seen.
- The safety manifest `tools/safety_manifest.toml` and `tools/check_safety_manifest.py`
  (in `just test-contract`): the manifest names the safety parameter layers of the
  simulator and real-flight profiles and pins the content of the safety files, the Lua
  script, the deny list, the parameter allowlist, the safety rules themselves
  (`sd_sim/params.py`, `sd_safety/constants.py`) and what enforces them (the gate's
  `vehicle.py`, `connection.py` and `allowlist.py`, `sd_sim/sitl.py`,
  `tools/params_check.py` and both checks); these
  files must stay pinned, so changing one needs `--update` in the same PR. The check
  fails when a pinned file changes without the manifest, when a failsafe, fence,
  battery, arming, RTL or scripting line in either profile has no rule, when a profile
  breaks the safety rules or the values the ground station reads back, when SITL or
  `sd-onboard` load other layers than the manifest names, or when a sim-only layer (or
  the allowlist) sets a parameter the safety layers set or any safety-relevant name
  (`FS_`, `FENCE_`, every `BATTn_` and `GPSn_`, `ARMING_`, `FLTMODE`, `EK3_SRC`,
  `MOT_SAFE_`, `RCn_OPTION`, `SYSID_MYGCS` ..., in any case); a parameter value that
  is not a number is a finding with its file and line, not a crash; Python files are
  pinned byte for byte, so a comment change needs `--update` too. `--update` writes the new digests after
  a reviewed change and prints which pinned files changed.
- Release tools. `just release-prep X.Y.Z` (or `X.Y.Z-rc.N`) refuses a dirty tree or an
  existing tag, runs `tools/check_merged_pr_gates.py`, sets the version in every
  `pyproject.toml` (PEP 440) and `package.json`, refreshes the uv lock files, runs
  `check_versions --expect` and turns `[Unreleased]` into the release's section, with
  the commit list from git-cliff (`cliff.toml`, pinned in `simulation/versions.lock`).
  It never commits or tags. `tools/check_merged_pr_gates.py` checks that every commit on
  main (the whole first-parent history minus `tools/accept/pr_gate_exceptions.txt`; tags
  do not shorten it) is the merge commit of a pull request into `main` whose gates pass
  when worked out again, since any workflow can write a status: `gate/review` with
  `check_review_comments.evaluate` over the comments written before the merge,
  `gate/owner` from the merge commit's own files (local git) and
  `check_owner_gate.approved_by_label_events` (the owner's newest `owner-ok`, added after
  the head commit's first check suite, so a head without one fails, and after any base
  change, force push or reopening), and `gate/pr-title` from the title at the merge. Workflow files the merge
  added or changed are read as YAML: none may be able to write a gate (a job without
  `permissions` at either level counts, since it gets the default token), and the gate
  workflows may only run on `pull_request_target`/`issue_comment`
  (`tools/workflow_rules.py`, shared with `tools/tests/test_workflows.py`). The exceptions
  file, the gate scripts and the gate and release workflows now need the owner's
  `owner-ok` (`check_owner_gate.PROTECTED`).
  Each merge is judged by the protected paths in force before it (its first parent's
  `PROTECTED` as well as the checker's own; a parent list it cannot read in full makes
  every changed path need `owner-ok`), and no merged workflow outside the gates may hold
  `pull-requests: write`, `issues: write`, `secrets: inherit` or any secret other than
  `GITHUB_TOKEN` and `OPENROUTER_API_KEY` (`workflow_rules.ALLOWED_SECRETS`).
  `.github/workflows/release.yml`, on a pushed `v[0-9]*` tag, first runs a job that
  refuses a tag that is not on main and runs the gate check from a clean copy of the
  protected scripts, with PyYAML pinned by hash (Python 3.12), so no other repository
  code runs before it. The workflow finds the newest published release (rc included)
  on the tag's line with `gh` and git alone; the scripts come from that release, first
  confirm that `release.yml` itself made it, with a successful run for that tag and
  commit whose `gates` job passed, and the check starts after it. A different answer,
  or any error, stops the job, and only the first release (no release and no release
  tag on the line) checks the whole history with the tagged commit's own copy. Past
  the API's hourly limit the check waits for the reset instead of failing, and a test
  runs the workflow's own anchor loop against the check's choice. New rules therefore
  apply from the release after the one that ships them; so do the `gates` job check and
  the waiting, which run in the anchor's copy. A deleted release and a `release.yml`
  without its gates job are left to the owner's ruleset on `v*` tags and immutable
  releases. While the check waits it shares the token's hourly limit with the gate
  workflows, so a long first release is best started when no pull request is under
  review. Then it checks the
  versions, builds the packages and the panel and publishes the
  GitHub release only if the tag still names the commit it built; only its publish job
  can write. `just showcase` is still not built.
- `tools/tests/test_workflows.py` fails when any workflow other than `review-gate.yml`,
  `owner-gate.yml` and `pr-title.yml` grants `statuses: write`, `checks: write` or
  `write-all` (at the top or in a job), leaves a job without `permissions` or names a job
  `gate/...` or `review/...`, so no
  other workflow can write a gate (A18). The workflow rules now read `*.yaml` files too.
- The image bank's second open dataset: KIOS Small Object Aerial Person Detection
  (University of Cyprus, Zenodo 7740081, CC BY 4.0; 3,136 pictures, 61,708 people),
  fetched with `just bench-fetch yes=1 sets=kios` from its two pinned archives (pictures
  and labels) and split by site in the new locked split `tools/bench/splits/v2.json`.
  v2 keeps HERIDAL's and WiSARD's parts from v1 unchanged, so HERIDAL numbers stay
  comparable. `just train-detector` mixes every fetched set that may be trained on,
  prints the val AP50 per dataset and keeps the epoch whose weakest dataset is best;
  `init=<best.pt>` starts from an earlier run's weights, and the manifest credits every
  set those weights saw. `just bench-detector` prints the recall per dataset and size,
  each at that dataset's own false-alarm threshold, and `just pick-detector` ranks each
  model by its weakest dataset's recall, so KIOS's many people cannot hide a model
  getting worse on HERIDAL; `compare=` in `just bench-detector` fails when that
  weakest-dataset recall or any one dataset's drops more than 2 points, or a dataset is
  no longer measured. A detector profile for the simulator is made only from a
  result measured on one dataset. `just data-dir` prints the
  data folder. JKU's Airborne Optical Sectioning data is not added:
  it has no person boxes for single colour pictures (`tools/bench/README.md`).
- `detector=model` for `just sim-headless`, `just demo`, `just demo-search` and `just
  e2e-search`: the simulator runs the owner's trained person detector
  (`SD_DETECTOR_MODEL` or `model=<manifest .json>`) instead of the stand-in, and passes it
  on to `sd-onboard`. The manifest and the model's SHA-256 are checked before anything
  starts; a missing or changed model stops the start, never a fallback to the stand-in
  (`sd_sim.detector_choice`, `sd_detect.onnx.check_model`). With Gazebo the search
  scenario's persons stand in the world as Gazebo Fuel models where the oracle has them
  (`sd_sim.people_scene`) and the drone carries the stills camera. The default and CI
  keep the stand-in. How to: `tools/train/README.md`.
- The search chain end to end in the simulator: synthetic stills of a seeded scenario
  world (`sd_sim.search_world`, `simulation/scenarios/SK*.yaml`), the detector and
  image-model oracles (`det_oracle_v0`, `vlm_oracle_v0`), the product's perception and
  keep-out cylinders, `sd-search`, `sd-brain` with a search script, and a scripted
  operator in the panel that views a picture before it confirms and only runs with
  `SD_SIM=1` (`sd_sim.operator`). SK01, SK03, SK04 and SK09 fly the whole chain and SK08,
  SK11 and SK12 their parts, each for three fixed seeds, in pull request CI
  (`sitl-search`, three shards) and must break no invariant (also: nothing confirmed
  without a person, no candidate vanishes or goes back, no cylinder entered again). The
  nightly `sitl-search-nightly` flies SK01-SK04 and SK06 with and without the
  probability map over seeds 1-50 and checks the campaign's requirements with `python
  -m sd_sim.eval_search aggregate`; its numbers are labelled "modellerat, ej uppmätt".
  `just e2e-search` runs the search demo (SK01: the confirmed find, home, landing) and
  writes `search_report.md`; `just demo-search` runs it locally with Gazebo physics.
- `sd-brain` hears about candidates: the situation lists the high ones and the operator's
  decisions, and the commander API's events `candidate` and `candidate_decided` wake a
  turn (once per candidate); the AI can only fly over a candidate or ask, never confirm
  or reject one.
- The image bank for the person detector (`tools/bench`, README there): `just bench-fetch
  yes=1` downloads HERIDAL (CC BY 3.0) to `<data_dir>/datasets/` (the visible-light part
  of WiSARD is registered for measuring, and fetched once its archive's SHA-256 is pinned), checks each archive's size and SHA-256, unpacks only pictures
  and labels under checked names and indexes them; it downloads nothing without `yes=1`,
  resumes a stopped download and sets a wrong file aside unused. `sets=simfarm` indexes
  the sim farm's renders for pre-training only. The split `tools/bench/splits/v1.json`
  (70/10/20) is by sequence (place or flight), so no near-copy of a test picture is
  trained on; the test part is locked. `just bench-detector` measures a detector on it:
  recall per person size (under 16, 16-32 and over 32 px) at 0.5 false alarms per 12 MP,
  AP50, precision at score 0.25 and time per picture, numbers only, to `build/bench/`;
  `tools/bench/profiles.py` turns a result into the simulator's detector profile
  (`det_oracle_v0` is the unmeasured start). A nightly job, `bench-mini`, reads 64 checked
  test pictures straight out of the archive with byte ranges, runs the self-check (nothing
  finds 0, the answer key finds all) and uploads only the numbers.
- `sd_detect.TiledDetector`: runs any detector over overlapping tiles of a large picture
  (640 px, 15 % overlap: 45 tiles for 12 MP), moves the boxes back, joins a person cut
  by a tile edge into one box and removes doubles.
- The person detector (`tools/train`, README there): `just detector-fetch yes=1` fetches
  the COCO weights of YOLOX-s and DAMO-YOLO-S (Apache-2.0, size and SHA-256 checked);
  `just train-detector model=yolox_s|damoyolo_s` fine-tunes one on the image bank's
  train part in 640 px tiles (fits an 8 GB graphics card, saves every 10 minutes and
  carries on after a stop); `just export-detector` writes the ONNX file and a manifest
  with its SHA-256, training sets, licences and credit, after checking ONNX Runtime
  against PyTorch; `just bench-detector model=<name>` measures it and `just
  pick-detector` chooses on the val part (recall over 32 px, then 16–32 px, then speed).
  `sd_detect.onnx` runs such a model on board with ONNX Runtime only, when
  `SD_DETECTOR_MODEL` names its manifest; a missing or changed file turns perception off
  and the reason goes to the ground (`executor.perception_off`, also when detection keeps
  failing), never a silent fallback; the panel shows it in the event log and as a toast,
  and again when detection works once more (`executor.perception_on`).
  Flying goes on. ONNX Runtime uses at most two threads.
  Model card: `software/detector/MODEL_CARD.md`.
- Photo mode for the simulator: `just ortho-fetch site=hellasgarden` fetches
  Lantmäteriet's newest colour orthophoto over the site (0.16 m per pixel, all tiles of the
  newest flight year, colour only) with the owner's own Geotorget login from a gitignored
  `config/geotorget.local.toml` (example in `config/geotorget.example.toml`) to
  `build/ortho/<site>/`, never in CI. The terrain then shows the photo on the open ground
  and keeps the painted forest floor under every tree's crown, where the world's own trees
  stand; without the photo the world is the same file for file. `just sim-photo` checks
  that the photo is there (and says how to get it) and starts `just sim` in the site world
  with `gi=vct`: Gazebo's voxel cone tracing for the cameras, also `just sim gi=vct` in a
  site world. `just world-snapshot` takes `gi=` and writes the real-time factor and the
  graphics card's memory in use while the cameras render. `tools/check_no_orthophotos.py`
  fails CI when a picture-shaped array, a GeoTIFF, a symbolic link or a world, snapshot or
  ground texture marked as built from a photo is in git (also its history), an uploaded
  artifact or the cache; every upload and cache in the workflows runs it on what it
  uploads or caches. The sim farm never uploads to R2 a round flown over the photo.
- Simulator exams for the AI commander: `sd-brain eval --backend sim` (`just
  brain-eval-sim`) flies random search scenarios with an answer key in a fast 2D world
  through the real commander loop and a gate with the area, height and speed rules. The
  cloud model decides; the executor flies. Each flight gets a JSON grade (safety, found or
  not, time to find, coverage, pilot questions, cost in kronor) under `data/brain-scores/`;
  any safety failure (a rejected or forbidden order, an order during STOP, a geofence
  breach) fails the flight. The cloud model has a hard cap of 30 kr per day counted on the
  provider's stated price per call (`usage.cost`), shared by all runs on the machine; at
  the cap, without a key, with a refused key (`key_invalid`) or when the cloud fails, a
  fixed search script takes over and the flight is not counted as an exam. A cost the
  provider did not state is shown as not measured, never guessed. Good and failed exams go
  to an example bank (a gitignored local folder, or an R2 bucket in the EU,
  `SD_BANK_R2_*`): `sd-brain bank export-golden` turns a good one into a replay fixture
  (`tests/golden/sim/`, only the seed and the model's answers) that `just brain-eval`
  flies again in CI without a network, and `sd-brain bank failures` lists the failed ones
  with what went wrong. `sd-brain eval grade --round <dir>` grades a simulator-farm round
  and writes `grade.json` in its folder.
  `just brain-scores` shows the grade per day and model and flags a worse day. The
  decision maker (`sd_brain.simeval.make_decision_maker`) and the grading
  (`score_flight`) are plain interfaces for a simulator farm. A nightly workflow
  (`brain-nightly`) runs a short cloud exam from main when the repository has the
  `OPENROUTER_API_KEY` secret and uploads the grades as an artifact.

### Changed

- The panel's choice lists (a drawn area's hint, the search profile's fields) work
  from the keyboard like a native select: ↓/↑ on the button opens the list on the
  current choice, ↓/↑, Home and End move, Enter or Space chooses, Escape and Tab close,
  and the focus returns to the button. The component moved to `src/ui/Picker.tsx`.
- Flight safety: arming needs exactly `sd_gcs_loiter_cap.lua` loaded and running
  (`SCR_LD_CHECKSUM` / `SCR_RUN_CHECKSUM` in `searchdrone-safety.parm`); a unit test
  fails when the script changes without a new checksum. LOITER's braking
  (`LOIT_BRK_ACC_M`, `LOIT_BRK_JRK_M`, `LOIT_BRK_DELAY`) is set explicitly to the stop
  model E8 assumes, and a test fails when they drift apart.
- The ground waits for the onboard executor's answer to every `KeepoutInstall`. A
  refused, failed, expired or unanswered keep-out list is sent again (above a stale
  sequence number when the executor holds a newer one), and the operator gets one red,
  sticky alarm in Swedish (`KEEPOUTS_NOT_INSTALLED`) until the list is on board
  (`KEEPOUTS_INSTALLED`). The sequence number stops at `INSTALL_SEQ_MAX`, logged once
  per run of refusals.
- A drone hovering for a keep-out cylinder (`executor.keepout_hold`) alarms the operator:
  orange short of a cylinder, red and sticky over a person (`KEEPOUT_HOLD`), saying
  whether it climbs straight up first. The fields are shared (`KeepoutHoldField`). One
  delivered after its mission closed is only logged. Every such alarm carries the
  drone's `sent_utc` in its details, so its record shows when a late one was sent (the
  panel does not show it yet).
- The tlog writer's queue has a cap (`QUEUE_MAX` frames); when the disk falls behind,
  new frames are dropped, numbered gaps show where, and `<stem>.lost.json` counts them.
- The chain scenarios' E8 oracle is as strict as the executor: a drone inside a cylinder
  when it came may leave it but never go deeper, and a cylinder's new shape gives no new
  grace. FLY HOME pressed in a chain run must be under way within its limit, and the
  drone must land at home. New SITL tests fly the margin case from #96 and require the
  inspection's `keepout_replan`.
- Flight safety: an inspection inside a cylinder over a person with every way out
  refused climbs straight up to the cylinder's top plus a margin, never above the
  geofence or the charter's height, then hovers and alarms; when it cannot climb it
  hovers and alarms. An inspection plans its way at most `REPLANS_MAX` times, and a
  goto's deadline runs from its first target, so replans cannot keep it flying. A way
  out planned again gets its own deadline, `goto.start` can no longer overwrite a
  renewed one, and every keep-out hover tells the ground whether the drone is inside a
  cylinder. The AI's next order after such a hover waits for the onboard snapshot after
  it also when the hover's red alarm names the turn (before, it was refused onboard as
  `STATE_CHANGED` and the drone kept hovering). STOP and FLY HOME still go past the AI
  and the reflex.
- Flight safety: any other order that would hover inside a person's cylinder (a search
  with no way around, a goto stopped short of the next cylinder) now climbs straight up
  the same way first, as an internal goto that replaces the order, and ends in the
  keep-out hover up there (`hold_reason` `limit`), never above the charter's height less
  5 m or the geofence. When it cannot climb it hovers where it is, as before. STOP, FLY
  HOME and manual take-over are unchanged.
- Flight safety: every end of that climb is the keep-out hover (`hold_reason` `limit`):
  its deadline, a mode not confirmed, and a charter hold when the AI goes quiet. While it
  climbs, a link loss's find move never takes the drone sideways out of the cylinder. A
  full list of the drone's own cylinders also climbs first when it is over a person, and
  the altitude limit's way back down ends in its own hover when the AI is quiet.
- Flight safety: a link loss detected during that climb no longer cuts it short or
  sends the drone sideways out of the cylinder below its top: the climb goes on and the
  find hover starts in the keep-out hold up there, also past the find move's deadline.
  The AI going quiet in the keep-out hold keeps it the keep-out hold.
- The ground station's wind check reads the flight controller's own `ATC_ANGLE_MAX`,
  `RTL_SPEED_MS` (`WP_SPD` when it is 0), `BATT_CAPACITY` and `BATT_LOW_MAH` on each
  connection, with the fence check's parameter reading. A parameter that cannot be read
  keeps its constant from `sd_safety.constants`, and `wind.param_fallback` is logged.
  `ENERGY_HOME` now also counts the wind on the way home: with a fresh MAVLink WIND the
  ground speed home is the airspeed minus the headwind, with the crosswind crabbed out,
  and the current comes from the hover current scaled to the lean that airspeed needs
  (the drag is measured while the drone holds position; until then, full lean at
  `RTL_SPEED_MS` through the air). The 1.3 margin stays. The strictest of the estimate
  as before, the same with the flight controller's numbers and the one with the wind
  decides the warning, so the warning never comes later than before, and a wind too
  strong to get home in (once the drag is measured) warns at once. The wind used is the
  mean over the last 60 s, direction and speed from the same samples, so one gust does
  not count. `ENERGY_HOME` also starts the battery guard's FLY HOME (`battery_rtl`,
  `vehicle.battery_rtl` and `alarm.BATTERY_RTL` with `reason: energy_home`), but only
  from the estimate with the flight controller's numbers, or from the wind estimate once
  the drag is measured, and both only with `BATT_CAPACITY` and `BATT_LOW_MAH` read from
  the flight controller (with the constant 10000 mAh or 1500 mAh reserve, before the
  read, after a reconnect, when one is not answered or `BATT_LOW_MAH` is 0 or not below
  the capacity, they only warn, so the 1500 mAh constant never flies a small pack
  home); an uncalibrated drag and a wind too strong to get
  home in only warn. Not while the pilot flies by hand. A ground speed of 0 or NaN does
  not count as holding position for the drag. Parameters outside ArduCopter's ranges
  (`RTL_SPEED_MS` and `WP_SPD` 0.1-20 m/s), a capacity outside 100-1,000,000 mAh and a
  `BATT_LOW_MAH` not below the capacity keep their constants; parameters not answered
  are asked for again on the same connection. A reconnect or a reboot of the flight
  controller (`FcTelemetry.fc_link_epoch`, also counting ATTITUDE's `time_boot_ms` going
  back) forgets the parameters and the measured drag. The guard's percentage level and the flight
  controller's failsafes are unchanged: an energy flight home under a STOP blocks only
  the energy trigger, so the percentage level still flies home after STOP and "Ta över".
  Resume's refusal and the alarm say when the wind and the battery sent the drone home.
  SITL W5 shows the ground station using sd_fpv7's `ATC_ANGLE_MAX` 35; W6 has
  `ENERGY_HOME` fly home (reason `energy_home`, Resume refused, STOP and "Ta över" leave
  the drone in LOITER).
  The battery guard's reason is set together with its latch (`Latches.start`), so no
  moment shows the latch without its reason. A late ATTITUDE (up to 1 s back) no longer
  moves `fc_boot_ms` back: the reboot check compares with the newest. The ground's
  README says to set `BATT_CAPACITY` and `BATT_LOW_MAH` on the flight controller, since
  without them `ENERGY_HOME` only warns.

- The nightly search campaign's targets (`python -m sd_sim.eval_search aggregate`):
  SK02's "never in the low list" still passes on a share of at least 0.95 and now also
  shows its Wilson lower bound. SK04 counts every verdict of the hiker's top that saw it,
  `conflict` / (`conflict` + `match`) at least 0.80, and needs at least 10 of them over
  the campaign; with fewer the row says "för få data" and does not pass. To get them, the
  SK04 runs search on after the confirmation: the operator asks the AI to search sector A
  again, through the gate as any order, for at most 240 s and never past the scenario's
  time limit, until the hiker's top has been judged or the drone heads home on its own;
  then the operator asks the drone home as before. SK06 asks the map for at least 10 %
  (time with the map at most 0.9 x without, one-sided 95 % bound as before) over seeds
  1-100 (`sk06_seeds=` in `just test-sitl` and `just ci-sitl`); the nightly job has 52
  shards.
- A detector manifest's `input_px` and `tile_px` must be integers. A manifest with, for
  example, `"input_px": 640.0` or `"640"` is now refused when the model is loaded
  ("Manifestets input_px och tile_px ska vara heltal"); write the plain number.
- The panel's look is one design system (`software/panel/src/theme/tokens.css`): spacing on
  a 4 px grid, one type scale, one corner radius, one height for inline chips and segments
  (touch targets keep their 44 px) and one for pane headers. The drone's state, the
  pending-approval badge, the candidates badge and alarm banners are filled in their colour
  so they read first; speed, heading, distance, GPS and the clock in the top bar step back
  a shade until they warn. Layout, texts and behaviour are unchanged; the STOP / RETURN
  HOME bar only takes the shared radius and title size.
- The nightly Gazebo job caches only the CC0 tree pictures (`build/tree_models`). The
  real trees (`build/trees`, derived from Skogsstyrelsen's tree height raster, which may
  not be redistributed) are fetched again on every run and never stored in the GitHub
  Actions cache, which pull request workflows, also from forks, can restore.
  The CAD job no longer caches the manufacturer models (`build/cad/vendor`) either, as
  some of them may not be redistributed. A workflow test fails when a cache or artifact
  stores anything outside a short list of folders.
- The simulated and planned pack is 10 000 mAh (6S2P Li-ion), not 9 000: `BATT_CAPACITY
  10000` in `searchdrone-safety.parm` with the reserves kept at 15 % and 7 %
  (`BATT_LOW_MAH 1500`, `BATT_CRT_MAH 700`), `capacity_mah` in `hardware/drone.yaml` and
  `battCapacityAh` in the SITL frame `sd_quad14.json`.
- Real flights only use zero-data-retention cloud endpoints: `[cloud.provider] zdr` now
  defaults to `true` when `profile` is `bench` or `field` and to `false` in `sim`, so
  OpenRouter gets `provider.zdr = true` for both the main and the fallback model. An
  explicit `zdr` still wins. `config/brain.toml` files copied from the old example carry
  `zdr = false`, which keeps zero retention off; remove that line (or set `zdr = true`)
  before flying. `sd-brain config-check` and the brain's start warn when a bench or
  field profile has `zdr = false`.
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
- `tools/bench/profiles.py` also refuses a result that does not name its dataset
  (`recall_by_dataset` missing or empty) or names one no longer in `datasets.toml`, so the
  `publish_results` check can no longer be skipped. A yes/no field in `datasets.toml` must
  be TOML `true` or `false`; a quoted `"false"` is refused instead of read as true.

### Added

- `just perf` (alias `just bench`, keys `runs`, `gui`, `eval`): the performance run on
  the owner's computer. It starts the demo with Gazebo (without a window unless
  `gui=true`), camera video and the cloud AI commander, flies SC01 with a test rig pilot
  and measures cold start, Gazebo's real time factor, video frames in headless Google
  Chrome, memory in WSL, Windows and the graphics card, STOP latency from the radio log,
  telemetry latency and the chat's answer time, then times the cloud model's answers
  (`sd-brain perf-cloud`). The report goes to `data/reports/perf-<time>.json` and `.md`;
  the run exits 0 only when every limit holds.
- Both simulator worlds step at 1.25 ms (800 Hz) and the drone's IMU runs at 800 Hz: at
  1 ms Gazebo in WSL2 reached only about 0.7 real time in lockstep with SITL, and
  ArduPilot arms only with the gyro at 1.8 times its 400 Hz loop.
- Real trees in `sd_hellasgarden`: `just trees-fetch site=<site>` takes every tree's
  position and height from Skogsstyrelsen's tree height raster (1 m) and its species from
  Naturvårdsverket's NMD 2023, and `just tree-pictures-fetch` makes a PBR texture atlas from
  Poly Haven's CC0 tree scans. The world then has about 14 700 trees of up to 34 m (mean
  height error 0.3 m against the raster), detailed card models within 150 m of home and
  cheap impostors further out, in glTF tiles. The tree list and pictures stay in `build/`
  and the nightly Gazebo job fetches them; its snapshot artifact has pictures from before
  and after. With the real trees the ground heights are Copernicus's as they are (no fixed
  12 m canopy taken off), and a public site can use Lantmäteriet's 1 m ground model with
  `[dem] source = "lantmateriet-mhm"`, `surface = "dtm"`. Sources and licences are in
  `ATTRIBUTION`.
- The simulator can fly the owner's 7-inch quad: `just sim vehicle=fpv7` and
  `just sim-smoke vehicle=fpv7` use `sd_fpv7`, whose mass, centre of gravity, inertia and
  looks are generated from the CAD model `hardware/cad/fpv7` (`just sim-model-fpv7`); the
  hw-cad CI job fails when the two drift apart. It has its own SITL parameter layer and
  builtin-physics frame (6S 10 000 mAh, 7x4 three-blade propellers). `sd_quad14` stays the
  default. The smoke flight's `result.json` gains `vehicle` and, with builtin physics,
  `hover_current_a`.
- Perception chain: `sd-onboard` takes stills from a camera folder (`[perception]` in
  `config/onboard.toml`, off by default), runs the class register's detector sources,
  places each detection on the ground with the flight controller's position and attitude
  and sends a `PerceptionReport` followed by a `CropData` JPEG for each target over
  SD-link, in its own lowest-priority queue. A detection may now
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
- `just sim-farm`, `just sim-farm-status` and `just sim-farm-setup`: a loop that runs
  randomized search rounds from the newest `main` and keeps camera pictures with person
  boxes (YOLO and COCO) computed from the simulator's facit and the camera's pose, plus a
  metadata file per picture. Each round passes a quality check; three failed rounds in a
  row stop the farm, `max_minutes=N` caps its run time, and it pauses while the GPU or
  CPU is busy (optionally only when the computer is idle). Passed rounds stay in
  `build/simfarm/` and, with a key in `config/simfarm.toml`, are also uploaded once to
  Cloudflare R2 in the EU. Person models (CC0, CC BY 4.0) are fetched pinned from Gazebo
  Fuel and credited. A round flies a seeded search in `sd_lake_forest` with SITL, the
  ground station and perception, lit by the round's drawn sun, ambient light and haze
  (clouds and haze change only the sky colour for now). Each still carries the camera's
  pose from Gazebo, matched to the picture within 20 ms. `commander=script` flies the
  scripted search (`SIMFARM.yaml`) at the round's height; `commander=sd-brain` hands the
  flight to sd-brain's exam decision maker on the cloud model. The flight's record for
  grading is kept under `runs/<run>/` in the round. People are seen through the
  simulator's labels until a person detector exists. A round needs `just db-up`. The
  lake-forest world's ground is now a generated meadow (grass, dry streaks and bare soil,
  drawn from noise and repeated every 60 m) instead of one flat green, so pictures from
  the air show ground texture; the colour stays away from red, and SK-RED still passes on
  it. A readable `config/simfarm.toml` that holds an R2 key now stops the farm
  (`chmod 600`) instead of warning, and uploads never follow a link. A round's drawn
  height is raised to clear the highest ground or tree top where it can fly (home, the
  search box and the straight way home, 15 m wider) by the site fixtures' 25 m, in whole
  5 m, below the test charter's ceiling; with `commander=sd-brain` the charter's lowest
  height is the round's, so the gate rejects lower orders. A round that cannot clear the
  trees is skipped before the simulator starts, with the reason in `round.json`.
- "Too windy" warning on the ground station (`sd_ground.services.wind`), a warning only:
  nothing on board or on the ground acts on it, and STOP, FLY HOME and the flight
  controller's failsafes are unchanged. `Telemetry.wind` (`WindCheck`) carries three
  checks: the mean tilt over 10 s in the air against `ATC_ANGLE_MAX` - 10 deg
  (`TILT_MARGIN`), the energy to fly home now at `RTL_SPEED_MS` and the measured current
  x 1.3 against what is left above `BATT_LOW_MAH` (`ENERGY_HOME`), and the flight
  controller's MAVLink WIND over the stage's limit (`WIND_LIMIT`; `[wind] stage` in
  `ground.toml`: `first_flights` 4 m/s, `until_calibrated` 8 m/s mean and 12 m/s gusts).
  ArduCopter sends WIND only with `EK3_DRAG_*` set, so that check is silent until the drag
  is measured. Nothing is calibrated yet (`calibrated: false`). The event log and the
  `vehicle` topic get `vehicle.wind_warning` and `vehicle.wind_ok` when the reasons change.
  The ground station now also reads the battery current and the WIND message.
- SITL wind tests with ArduPilot's own wind (`simulation/tests/model/test_wind_sitl.py`,
  all marked uncalibrated): W1 hold in 10 m/s (and with turbulence), W2 RTL home against
  12 m/s within 1.5 x the model's time and its mAh + 20 %, W3 a 10 s gust from 8 to 15 m/s
  on a survey leg, W4 the flight controller's battery failsafe 500 m out against 10 m/s.
  The nightly SITL job sweeps W2 over 0-18 m/s and is red only if the drone does not get
  home in 12 m/s or less.
- Class register and detector sources (`sd_detect`): a table of classes (`person`,
  `red_block`, `car`, `building`, `road`) with Swedish names, kind, source and plausible
  size; a colour detector on real pixels for red blocks (with a lowest score per class,
  so a sliver of a red car is no block); the simulator's labelled boxes as an oracle
  source (simulator only, marked `oracle`); a size check that drops a red car posing as a
  block. Georeferencing in `sd_geo` (pixel to ground on flat ground and back,
  footprint, pixels per metre, 95 % position error). The SK-RED scene as world objects
  (`sd_sim.targets`, placed with `render_world(placed=)`) and `python -m
  sd_sim.render_stills`, which renders the detector's test pictures headless in Gazebo.
- `sd-searchsim`: a fast 2D search simulator (probability map with terrain and canopy,
  a lost-person motion model, battery and mission time) and a small learned policy that
  suggests which sector to search next. `just rl-train` trains it on CPU with evolution
  strategies (`size=ci` for a short run); `just rl-eval` compares it with lawnmower,
  spiral and two greedy rules on the same held-out seeded scenarios and writes
  `build/rl/eval.json`. Weights stay in `build/rl/`, never in git. Advisory only: no
  flight-path package imports it (import contract), and nothing about flight safety
  changes. CI trains and evaluates the CI size and uploads the comparison.
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

- The cloud model's zero-data-retention default (`[cloud.provider] zdr` unset) follows
  the ground station's profile from its `server.hello`, not the brain's own: on until the
  ground station has said it, off only against a `sim` ground station, and on again while
  the stream is down. The offline evaluations count as `sim`. The AI configuration hash is
  computed again after the hello. The brain logs a line when the ground station's
  profile turns zero retention off.
- `sd-brain bank export-golden` and `eval-profiles record=1` refuse answers holding the
  value of any key the AI reads (`SD_JEV_API_KEY` and `SD_OPENAI_API_KEY` were missing);
  the list is built from the brain's own key list, and the keys in `.env` are checked too,
  `SD_BRAIN_TOKEN` and the `SD_BANK_R2_*` keys included.
- The cloud matcher refuses a crop larger than 512 px instead of scaling it down: the
  link caps crops at 512 px, so a larger image is not a crop.
- The cloud image policy passes a text part on with its text only, no other keys.
- The probability map is frozen only with a confirmed find checked under the map's write
  lock, and a sector's `psr` is at most 100 (`PSR_MAX`); a sector stored before that is
  read with its `psr` clamped.
- Simulator exams: the closer look flies inside the charter's height band, the farm grade
  takes `flew_on` from the track flown on a search order after the raise (the exam's
  rule, from `sd_schemas.simfarm`) instead of counting every raised candidate as flown
  on, and the scripted decider raises the ground's high candidates from their
  `event.candidate` bodies instead of reading the situation's text.
- The sim farm moves a round flown over the orthophoto from `rounds/` to `draped/` once
  instead of trying to upload it on every turn. `ROUND_SCHEMA` has one home
  (`sd_schemas.simfarm`), the workflow test fails a storing step without a `path`, and
  `PerceptionLoop` takes prebuilt detector sources (`sources=`).
- The image bank's downloads (`just bench-fetch`, the mini bank and `make_splits`) and
  the detector weights' (`just detector-fetch`, `tools/train/fetch_weights.py`) try a
  request up to four times, after 2, 4 and 8 s, when the network fails on the way (a
  timed-out TLS handshake, a dropped connection, a cut read or a 5xx). A 4xx answer, a
  certificate that does not verify or a refused file stops at once, as before.
- The simulator farm's grade (`sd-brain eval grade`) now counts a candidate as raised only
  when the AI named it (`K3`) to the operator in a chat message or a report, as the exam
  does: `flight.json` carries the ground's candidate queue with when each was raised
  (`sd_schemas.candidate_ai.named_seqs` is the one rule for both). Without a queue the
  ground's finds stand in as before. The CANDIDATES rule gives way to an operator who asks
  the drone to hold, hover or circle a candidate and to a confirmed find. The exam flies
  the closer look (`inspect_detection`, reduced: near the candidate, low, then hovering),
  and a test flies the look and the interrupted search ordered again.
- A keep-out way around now keeps the margin (`KEEPOUT_ARC_MARGIN_M`) from every
  cylinder, not only from the ones it was made for: a leg of a way around that passes
  closer than the margin to another cylinder gets a way around that one too. A SITL test
  flies a survey past two overlapping cylinders, passed as one group.
- The simulator exam (`just brain-eval-sim`) now measures the path the real drone uses.
  A sighting becomes a search candidate: a high one wakes the AI with the ground's own
  `candidate` event and every open one is in the "Kandidater" line, both built by the
  same functions as the ground station (now in `sd_schemas.candidate_ai`). The
  simulator-only warning `POSSIBLE_PERSON_<pin>`, its map pins and close-up verdicts are
  gone, and so is the prompt's SIGHTINGS rule. A find is a candidate near the person that
  the AI named to the operator (`K3` in a report or a chat message) after which the
  search flew on (a flight to the candidate itself does not count); map marks no longer
  count (grade version 2, shown apart from older grades in `just brain-scores`). The
  CANDIDATES rule says that raising a candidate never pauses the search; the one
  exception is the closer look on a candidate seen in one still, after which the AI
  orders the interrupted search again.
- The nightly SK06 comparison no longer ends a run when the keep-out reflex stops a
  sector's search (`KEEPOUT_VIOLATION`). The test then stopped ordering sectors, the
  drone flew home after one sector and the run counted as not found (900 s): two of the
  50 runs without the map in the nightly of 24 October. Such a sector is now ordered
  again when it comes next, planned around the cylinders; stopped twice, it is passed over.
- Gazebo's camera stream starts also on a busy computer. `gz topic -p` publishes the
  request once after a fixed 0.8 s and exits 0 even when the camera plugin had not been
  found yet (gz-transport 13, its issue #47), so the stream sometimes never started and
  the nightly smoke flight counted 0 frames. The simulator now sends the request again
  until linksim sees frames, within the same 30 s. In the nightly Gazebo job the terrain
  world's steps no longer depend on the stock smoke flight, and every flight in the
  terrain world runs only after the world is built: the sd_fpv7 flight there built it
  itself after a red smoke flight, and that took it past its 3 minutes.
- The panel no longer logs a 404 in the browser's console for every mission without a
  probability map. `GET /missions/{id}/probability-map` now answers 200 with `null` before
  the first map (like the search profile); the map's picture still answers 404 then. The
  nightly panel test in a browser on SITL failed on that console line.
- A search no longer stops in front of one person seen again and again. Each sighting
  brings its own keep-out cylinder, and the way around one of them ran into the next, so
  no way around was found: the drone hovered, the new search was turned down and SK09
  flew home without its find. Cylinders whose margins overlap are now passed as one
  group, along the shortest way that keeps the 5 m margin from each of them and stays
  inside the charter; a cluster of more than 20 has no way around (the drone holds).
- Onboard events held back by a lost link land in the mission they happened in. They
  were filed under whatever mission was open when they arrived, so a find hold reported
  after landing (and after the mission closed) was in no mission's log (LL4). Each event
  now carries its mission id.
- `just train-detector model=damoyolo_s` no longer stops on a graphics card with "unsafe
  to autocast" and then "Found dtype Float but expected Half". DAMO-YOLO's label
  assignment and quality focal loss call `binary_cross_entropy`, which mixed precision
  refuses; the head now computes its scores and loss in float32 with autocast off. The
  assignment's move to the CPU no longer catches every error as running out of memory
  (the misleading OOM warning before the second error). YOLOX turns autocast off for its
  assignment on every device, not only CUDA. A test runs both families' training step
  under autocast.
- The link test `test_client_unreachable_returns_none` no longer fails on WSL2 with
  mirrored networking, where a connection to a closed port hangs instead of being
  refused: it dials a port held bound without listening and allows the client's own open
  timeout. A new test checks that the client gives up on a server that accepts TCP but
  never answers, within its open timeout.
- The AI's charter proposals no longer fail five times on the demo request. The brain
  turns every draft into the proposal inside the turn, so an unknown pin, a rectangle
  without its size, a skill name the registry does not have, a value beyond the mission
  limits, a height band the ground would refuse, a stale `base_version`, home outside the
  flight area or an area beyond the range is sent back to the model at once, every fault
  in one repair text with the ground station's reason code (up to three repair rounds,
  `behaviour.max_proposal_repair_rounds`, within the turn's deadline) without a POST. A
  draft still refused when the rounds or the turn's time run out, or when the model stops
  answering, counts as a refused attempt instead of ending in "Jag hann inte svara".
  So does the same refused draft sent again (no more repairs then), and a turn that ends
  with a text after a refused draft instead of a proposal (unless the text explains a
  mission limit the request exceeds), and a retry turn that proposes nothing: the AI no
  longer goes quiet after it, and a retry that ends without a proposal resets the
  attempts for the next request. The repair text, and the retry turn's context, list the
  map pins and drawn areas in the situation (or "none"), the shape of a rectangle on a
  pin and, when the operator's line names a box and a pin, the search area it describes
  (without a height or a size the mission would refuse). The tool schema offers a drawn
  area and `operating_area` only when an area is drawn, with the known labels and pin ids
  as its only values; without pins, an area needs `center`.
  The AI can no longer draw the flight area: it names only the search
  areas (or an area the operator drew) and the system computes the flight area around
  home. The tool schema gives each area kind its own required fields, a description on
  every field, the registry's skill names as `allowed_skills` values and the mission's
  height and speed limits. A refused proposal may be corrected up to three times per
  request (an operator line no longer drops the errors); after the third the AI asks the pilot
  one question with `refs.charter_failed` and the error codes. `just perf` no longer
  answers "Ja, godkänn." to questions; it fails at once on that line, or on an answer to
  its request marked `refs.turn_failed` (no answer in time, an unreadable reply), instead
  of waiting for a card. The chat
  is read before the situation in every turn, the charter error line shows the field and
  a hint, and the decision log stores `retry_of`, `attempt`, `proposal_errors` and
  `proposal_draft`. `HOME_OUTSIDE_AREA` and `SECTOR_OUTSIDE_AREA` now say "utanför
  flygområdet".
- The perf run's mission now ends at home. Fortsätt right after STOPP takes effect at
  once: the onboard executor ends the stop's notice window and its fallback chain, so
  LOITER is sent instead of the drone staying in BRAKE (a new STOPP still blocks), and a
  refused mode write is logged with its reason. The brain keeps the operator's latest
  requests in the mission memory and shows them on every turn as `UPPDRAG` (quoted as
  data; a "ja kör" or a "fortsätt" replaces nothing; cleared after landing), shows the
  sectors' size and position and the charter notes on the charter line, and the system
  prompt says to call `return_home` when the request is done. A finish backstop
  (`behaviour.finish_idle_s`, default 180 s, 60-1800 s): airborne with nothing running or
  waiting and no operator input, find or AI order for that long, the brain says so in the
  chat and orders `return_home` through the gate. A charter is proposed only in a turn an
  operator line started (or the retry of a refused proposal), an approved charter wakes
  the AI once the drone has installed it, and the situation text says "godkänd,
  installeras" (or that the install failed), names a pending change once and drops
  "Fritt runt drönaren" in the air. The goto schema requires its target (a map pin, or
  `KOORDINATER` with lat and lon), the same refused order twice ends the turn, a plan
  note sent with an order is kept, each model call is bounded as a whole (a trickling
  cloud starts the cloud pause), and the decision log keeps the prompt, the offered tools
  and the calls when a turn hits its deadline. `just perf` still reports the browser
  measurements (P2, P7), the chat answer times (P8) and the model's answer time when the
  flight breaks off, and its notes say what happened.
- The Gazebo model `sd_quad14` can take off: each rotor now has its own two lift surfaces
  (they all sat on rotors 0 and 1, where the clockwise pair cancelled the lift). The
  vehicle now starts with its legs on the ground in every world; in a site world they were
  inside the landing pad's box and stuck to it.
- `sitl-nightly-gz` also flies `sd_quad14` (the parity flight), even when the smoke
  flight's checks fail.
