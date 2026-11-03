# Searchdrone

Searchdrone is an open-source search-and-rescue drone that finds missing people from a description, directed by an AI mission commander you talk to in chat.

[![REUSE](https://img.shields.io/badge/REUSE-compliant-green)](https://reuse.software)
[![Software: Apache-2.0](https://img.shields.io/badge/software-Apache--2.0-blue)](LICENSES/Apache-2.0.txt)
[![Hardware: CERN-OHL-S-2.0](https://img.shields.io/badge/hardware-CERN--OHL--S--2.0-blue)](LICENSES/CERN-OHL-S-2.0.txt)

## Status

| Stage | What | Status | Release |
|---|---|---|---|
| 0 | Workshop: repository, tools, CI, review gate | In progress | v0.1.0 |
| 1 | First capability in the simulator: chat, take off, fly to a point, survey an area, land | Planned | v0.1.0 |
| 2 | Person detection and following | Planned | v0.2.0 |
| 3 | Search from a description | Planned | v0.3.0 |

There is no flying code yet. This repository currently holds the project skeleton and its tooling.

## Safety notice

Searchdrone is experimental. Never fly a real drone with it without a qualified safety pilot holding their own RC transmitter, and always follow local rules. The STOP and RETURN HOME buttons always bypass the AI. The AI can never change flight-controller parameters, disable the geofence, raise its own autonomy level or confirm a find. Searchdrone does not use face recognition.

## How it works

```
 Operator (browser, mobile friendly)
     │  chat with the AI ── STOP / RETURN HOME (bypass the AI)
     ▼
 Panel ──► Ground station ──► safety gate ──► radio link ──► onboard computer ──► flight controller
               ▲                                                                    (ArduPilot)
               └── AI mission commander                    Simulator: ArduPilot SITL + Gazebo
```

Seven parts: flight controller, onboard computer, radio link, ground station, AI mission commander, panel and simulator. Every order from the AI goes through a safety gate in the ground station that rejects anything outside the mission frame the operator approved. Below that, the flight controller's own geofence and failsafes are a second, independent layer.

## Quick start (development)

Requirements: Windows 11 with WSL2 Ubuntu 24.04 (or Ubuntu 24.04), git and curl. Docker Engine, an NVIDIA GPU with 8 GB VRAM and 16 GB RAM are needed for the simulator in later stages. The AI chat uses a cloud model (`SD_CLOUD_API_KEY` in `.env`); without network or key a fixed command mode answers, and STOP and RETURN HOME never need either.

```bash
git clone https://github.com/YXHFR/Searchdrone.git ~/Searchdrone
cd ~/Searchdrone
bash tools/setup/wsl-setup.sh   # once: Docker Engine, uv, just
bash tools/bootstrap.sh         # same as `just bootstrap`
just check
```

The git hooks need a file with your own private search patterns (home address, real coordinates, e-mail), one per line, so they can never be committed by mistake. It is gitignored and may be empty:

```bash
mkdir -p config/local && touch config/local/private-patterns.txt
```

Common commands (`just --list` shows all):

| Command | What it does |
|---|---|
| `just check` | Everything that must be green before a pull request |
| `just lint` / `just test` | Lint, or unit tests only |
| `just fmt` | Format code |
| `just setup-config` | Create local config and random secrets |
| `just dev` | Database, ground station, panel (hot reload), onboard computer and AI against a fake flight controller, then the login link; Ctrl-C stops it. Needs Docker for the database, no simulator and no API key (`physics=builtin` or `gz` flies ArduPilot SITL instead) |

Settings are passed as `key=value` after the recipe name, for example `just test-sitl physics=builtin sc=SC04`.

### Try your trained detector in the simulator

The simulator uses a stand-in person detector unless you choose your own trained one
([`tools/train/README.md`](tools/train/README.md)) with `detector=model`:

```bash
just sim-farm-setup                                   # once: the 3D person models
export SD_DETECTOR_MODEL=/path/to/person_yolox_s.json # the line pick-detector printed
just demo detector=model          # Gazebo with the SK01 persons; sd-onboard runs your model
just e2e-search detector=model    # the search demo with your model, report in runs/<run>/
```

`model=/path/to/manifest.json` works instead of the variable. A missing or changed model
stops the start; it never falls back to the stand-in. Details in
[`tools/train/README.md`](tools/train/README.md#try-your-trained-detector-in-the-simulator).

### Height map for a closer look (floor pack)

Before the drone goes down for a closer look at a possible person, it needs to know how
high the ground, the trees and the known power lines are, without internet. `just
floor-fetch` makes that map (a *floor pack*) for an area of at most 10 x 10 km from
Lantmäteriet's forest laser scan and ground model, plus power lines from OpenStreetMap.
Every user fetches their own data with their own login; no map data is ever in this
repository.

1. Create a free account at Lantmäteriet's [Geotorget](https://geotorget.lantmateriet.se)
   and order (free of charge) **Laserdata Nedladdning, skog** and **Markhöjdmodell
   Nedladdning, grid 1**.
2. Put your login in `config/geotorget.local.toml` (copy `config/geotorget.example.toml`,
   then `chmod 600`), or set `SD_LM_USER` and `SD_LM_PASSWORD`. Without either, the tool
   asks.
3. Draw your area as one polygon in a GeoJSON file outside the repository
   (`config/floor_area.example.geojson` shows the format), then:

```bash
just floor-fetch area=~/areas/lake.geojson name=lake       # to ${SD_DATA_DIR}/floor/lake/
just floor-push name=lake host=pi@drone.local dir=/srv/sd/onboard/sd1  # its state_dir
```

The tool downloads only the tiles that cover the area (about 70-130 MB of point cloud
per 2.5 km square and 250-300 MB of ground model per 10 km square, cached in
`${SD_DATA_DIR}/floor/.cache/`), deletes the point clouds when it has read them (they
may hold personal data under Lantmäteriet's terms) and prints the pack's
`floor_pack_sha256` for the mission. `lines=0` skips OpenStreetMap and says so.

The data is CC BY 4.0. A pack carries its attribution, and the panel shows it: "©
Lantmäteriet, Laserdata Nedladdning, skog och Markhöjdmodell (grid 1), CC BY 4.0.
Bearbetad (trädhöjd) av Searchdrone." Power lines are © OpenStreetMap contributors,
ODbL 1.0. The map only plans how low the drone may go; it does not show every wire,
mast or tree that grew since the scan.

## Repository layout

```
software/common/   shared packages: schemas, safety gate, geometry, MAVLink, link, observability
software/onboard/  onboard process (sd-onboard)
software/ground/   ground station (sd-ground)
software/brain/    AI mission commander (sd-brain)
software/detector/ person-detector interface (stage 2)
software/hwcalc/   hardware calculations
simulation/        simulator tooling, versions.lock
hardware/          open hardware (CERN-OHL-S-2.0)
config/            example configuration (local copies are gitignored)
tools/             setup scripts and repository checks
```

Each package has a short README.

## Development

Run `just check` before every pull request. Pull requests follow [Conventional Commits](https://www.conventionalcommits.org) titles and are reviewed by three AI reviewers (architecture, IT security, flight safety) in addition to the automated tests. Changes to `hardware/` also need the maintainer's approval. See [CONTRIBUTING.md](CONTRIBUTING.md), [SECURITY.md](SECURITY.md) and [CODE_OF_CONDUCT.md](CODE_OF_CONDUCT.md). Changes are listed in [CHANGELOG.md](CHANGELOG.md).

## Hardware

The simulated v1 drone is a quadcopter of about 2.3 kg with 14" propellers and about 33 minutes of estimated flight time. Hardware revision A is planned. Hardware files live in `hardware/` and are checked by automated calculations plus the maintainer's approval.

## Roadmap

- Stage 0: the workshop: repository, CI and review gate.
- Stage 1: chat-directed flight in the simulator (v0.1.0).
- Stage 2: detect and follow a person.
- Stage 3: search for a missing person from a description.

## License

Searchdrone is open source and open hardware.

- **Software** is licensed under [Apache-2.0](LICENSES/Apache-2.0.txt).
- **Hardware** (everything under `hardware/`: PCB designs, mechanical/CAD files, BOMs) is licensed under the [CERN Open Hardware Licence v2 – Strongly Reciprocal](LICENSES/CERN-OHL-S-2.0.txt).

- **Simulator site data** under `simulation/gazebo/sites/` keeps its own licence: map layers © [OpenStreetMap contributors](https://www.openstreetmap.org/copyright) ([ODbL-1.0](LICENSES/ODbL-1.0.txt)); heights produced using Copernicus WorldDEM-30 © DLR e.V. 2010-2014 and © Airbus Defence and Space GmbH 2014-2018 provided under COPERNICUS by the European Union and ESA; all rights reserved ([licence](LICENSES/LicenseRef-Copernicus-WorldDEM-30.txt)). The organisations in charge of the Copernicus programme by law or by delegation do not incur any liability for any use of the Copernicus WorldDEM-30. Nothing in the project is endorsed by them.

Licensing follows the [REUSE](https://reuse.software) specification; see [`REUSE.toml`](REUSE.toml). A possible Ultralytics detector plugin (later) would be AGPL and kept separate from the Apache-2.0 packages.
