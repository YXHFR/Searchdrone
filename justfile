# Searchdrone task runner. Run `just --list` to see every recipe.
#
# Secrets in .env are never exported to the shell: services read their own secrets,
# and docker compose gets them with --env-file. Settings are passed as key=value
# after the recipe name (e.g. `just test-sitl physics=builtin sc=SC04`), never --flags.

set shell := ["bash", "-euo", "pipefail", "-c"]

typos_version := "1.50.3"
zizmor_version := "1.30.1"

# Recipes whose ticket is not built yet. A direct call prints "inte byggt än (<ticket>)"
# and exits 2; composite recipes skip them. Remove a name when its ticket lands.
unbuilt := "db-up db-down db-check migrate gen gen-check test-panel-e2e panel-build"

_just := just_executable() + " --justfile " + quote(justfile()) + " --working-directory " + quote(justfile_directory())

# --- installation --------------------------------------------------------------------

# Install every tool on a fresh WSL2 Ubuntu 24.04 (same as `bash tools/bootstrap.sh`)
bootstrap:
    bash tools/bootstrap.sh

# Check tool versions, install dependencies and git hooks, then create local config
setup:
    #!/usr/bin/env bash
    set -euo pipefail
    uv --version | grep -Eq '^uv 0\.11\.' || { echo "Kräver uv 0.11.x. Kör bash tools/bootstrap.sh."; exit 1; }
    node --version | grep -Eq '^v24\.' || { echo "Kräver Node 24. Kör bash tools/bootstrap.sh."; exit 1; }
    pnpm --version >/dev/null || { echo "pnpm saknas. Kör bash tools/bootstrap.sh."; exit 1; }
    docker info >/dev/null 2>&1 || echo "Varning: docker fungerar inte utan sudo (behövs från databas-steget)."
    uv sync --all-packages --frozen
    pnpm install --frozen-lockfile
    uv tool run --from "pre-commit==$(python3 tools/versions.py pre_commit_version)" pre-commit install --hook-type pre-commit --hook-type pre-push
    {{_just}} setup-config

# Copy example config to config/<part>.toml and fill .env with random secrets
setup-config:
    uv run --no-sync python tools/setup_config.py

# --- everyday checks -----------------------------------------------------------------

# Format Python and TypeScript/JSON
fmt:
    uv run ruff format
    uv run ruff check --fix
    pnpm exec biome check --write

# Lint, format check, spelling and the forbidden-file guard
lint:
    #!/usr/bin/env bash
    set -euo pipefail
    uv run ruff check
    uv run ruff format --check
    pnpm exec biome ci
    uvx --from "typos=={{typos_version}}" typos
    patterns=()
    [ -f config/local/private-patterns.txt ] && patterns=(--private-patterns config/local/private-patterns.txt)
    uv run tools/check_forbidden_files.py "${patterns[@]}"
    for step in check_forbidden_mavlink check_versions check_compose_ports; do
        if [ -f "tools/${step}.py" ]; then uv run "tools/${step}.py"; else echo "hoppar över ${step} (inte byggt än)"; fi
    done
    if compgen -G ".github/workflows/*.yml" >/dev/null; then
        uvx "zizmor=={{zizmor_version}}" --offline .github
    else
        echo "hoppar över zizmor (inte byggt än)"
    fi
    if [ -n "${SD_BASE_SHA:-}" ]; then
        if [ -f tools/check_commit_emails.py ]; then uv run tools/check_commit_emails.py --since "$SD_BASE_SHA"; else echo "hoppar över check_commit_emails (inte byggt än)"; fi
    fi

# Type-check Python (pyright) and TypeScript (tsc)
typecheck:
    uv run pyright
    pnpm -r exec tsc --noEmit

# Unit tests without coverage requirements (Python and TypeScript)
test:
    uv run pytest -m "not contract and not component and not db and not sitl"
    pnpm -r --if-present test

# Contract, component and unit tests with coverage, plus import contracts
test-contract:
    #!/usr/bin/env bash
    set -euo pipefail
    {{_just}} _step db-up
    {{_just}} _step migrate
    uv run pytest -m "not sitl" \
      --cov=sd_safety --cov=sd_ground --cov=sd_onboard \
      --cov-branch --cov-report=json:coverage.json --cov-report=term
    uv run lint-imports
    for step in check_coverage check_safety_manifest; do
        if [ -f "tools/${step}.py" ]; then
            if [ "$step" = check_coverage ]; then uv run tools/check_coverage.py coverage.json tools/thresholds.toml; else uv run "tools/${step}.py"; fi
        else
            echo "hoppar över ${step} (inte byggt än)"
        fi
    done

# Everything that must be green before a pull request
check:
    {{_just}} lint
    {{_just}} typecheck
    {{_just}} test
    {{_just}} _step gen-check
    {{_just}} test-contract
    uv run reuse lint

# Build wheels into dist/ and the panel
build:
    uv build --all-packages -o dist
    {{_just}} _step panel-build

# --- schemas -------------------------------------------------------------------------

# Generate JSON Schema, TypeScript types and OpenAPI snapshots
gen: (_not-built "B0-03")

alias gen-schemas := gen

# Fail if generated files are out of date
gen-check: (_not-built "B0-03")

# --- database and services -----------------------------------------------------------

# Start Postgres + PostGIS on 127.0.0.1:5433
db-up: (_not-built "B0-04")

# Stop the database
db-down: (_not-built "B0-04")

# Show the PostGIS version of the running database
db-check: (_not-built "B0-04")

# Create or upgrade the database schema
migrate: (_not-built "B1-05")

# Start the ground station
ground: (_not-built "B1-05")

# Write a commander token for sd-brain
brain-token: (_not-built "B1-10")

# Panel end-to-end tests against a mock server
test-panel-e2e: (_not-built "B1-06")

[private]
panel-build: (_not-built "B1-06")

# --- simulator -----------------------------------------------------------------------

# Run the whole stack against a fake vehicle (keys: physics, gui, brain, instance)
dev *args:
    uv run --no-sync python tools/just_args.py dev {{args}}

# Start the simulator stack with Gazebo (keys: physics, gui, video, brain, ...)
sim *args:
    uv run --package sd-sim python -m sd_sim.launch up {{args}}

alias sim-up := sim

# Like `sim` but builtin physics and a test picture as video
sim-headless *args:
    SD_SIM__HEADLESS=1 uv run --package sd-sim python -m sd_sim.launch up video=test {{args}}

# Stop everything this instance started
stop *args:
    uv run --package sd-sim python -m sd_sim.launch down {{args}}

alias sim-down := stop

# Check the simulator environment
sim-doctor: (_not-built "B0-05")

# Install ArduPilot SITL and Gazebo
sim-setup: (_not-built "B0-05")

# GUIDED, arm, take off to 20 m, land (keys: physics, gui)
sim-smoke *args:
    uv run --no-sync python tools/just_args.py sim-smoke {{args}}

# Scenario tests, headless (keys: sc, variant, physics, instance, shard, shards, ...)
test-sitl *args:
    uv run --package sd-sim python -m sd_sim.testrun {{args}}

# Run the CI SITL job locally in the same container
ci-sitl: (_not-built "B0-08")

# Full demo locked to the public CMAC home (keys: lan, record, ...)
demo *args:
    uv run --package sd-sim python -m sd_sim.launch up --demo {{args}}

# Headless end-to-end run with the scripted commander (keys: runs, offline)
e2e *args:
    uv run --no-sync python tools/just_args.py e2e {{args}}

# Performance measurement on the owner's computer (keys: runs)
perf *args:
    uv run --no-sync python tools/just_args.py perf {{args}}

alias bench := perf

# Download the local AI model
models-fetch: (_not-built "B1-10")

# Download the offline map extract
maps-fetch: (_not-built "B1-11")

# --- hardware, acceptance and release ------------------------------------------------

# Check mass and flight time against hardware/drone.yaml
hw-check: (_not-built "B0-06")

# Acceptance checks for a release (keys: phase, sha)
accept *args:
    uv run --no-sync python tools/just_args.py accept {{args}}

# Local traceability report between tests and the internal plan (local only)
check-traceability: (_not-built "B1-16")

# Record the showcase GIF for a stage (keys: brain)
showcase STAGE *args:
    uv run --no-sync python tools/just_args.py showcase {{args}}

# Set one version everywhere before a release
release-prep VERSION: (_not-built "B1-18")

# --- helpers -------------------------------------------------------------------------

[private]
_not-built ticket:
    @echo "inte byggt än ({{ticket}})"; exit 2

[private]
_step recipe:
    #!/usr/bin/env bash
    set -euo pipefail
    if [[ " {{unbuilt}} " == *" {{recipe}} "* ]]; then
        echo "hoppar över {{recipe}} (inte byggt än)"
    else
        {{_just}} {{recipe}}
    fi
