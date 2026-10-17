# Searchdrone task runner. Run `just --list` to see every recipe.
#
# Secrets in .env are never exported to the shell: services read their own secrets,
# and docker compose gets them with --env-file. Settings are passed as key=value
# after the recipe name (e.g. `just test-sitl physics=builtin sc=SC04`), never --flags.

set shell := ["bash", "-euo", "pipefail", "-c"]
# Recipe arguments reach the shell as "$@", never pasted into the command line.
set positional-arguments

typos_version := `python3 tools/versions.py typos_version`
zizmor_version := `python3 tools/versions.py zizmor_version`

# Recipes whose ticket is not built yet. A direct call prints "inte byggt än (<ticket>)"
# and exits 2; composite recipes skip them. Remove a name when its ticket lands.
# tools/tests/test_justfile.py checks that this list matches the (_not-built ...) recipes.
unbuilt := ""

# Check scripts in tools/ that are not built yet. Composite recipes skip only these;
# any other missing script fails. The test checks that none of them exists yet.
unbuilt_tools := "check_forbidden_mavlink check_coverage check_safety_manifest"

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
        if [[ " {{unbuilt_tools}} " == *" ${step} "* ]]; then echo "hoppar över ${step} (inte byggt än)"; else uv run "tools/${step}.py"; fi
    done
    if compgen -G ".github/workflows/*.yml" >/dev/null; then
        uvx "zizmor=={{zizmor_version}}" --offline .github
    else
        echo "hoppar över zizmor (inte byggt än)"
    fi
    if [ -n "${SD_BASE_SHA:-}" ]; then
        if [[ " {{unbuilt_tools}} " == *" check_commit_emails "* ]]; then echo "hoppar över check_commit_emails (inte byggt än)"; else uv run tools/check_commit_emails.py --since "$SD_BASE_SHA" --head "${SD_HEAD_SHA:-HEAD}"; fi
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
      --cov=sd_safety --cov=sd_ground --cov=sd_onboard --cov=sd_brain \
      --cov-branch --cov-report=json:coverage.json --cov-report=term
    uv run lint-imports
    for step in check_coverage check_safety_manifest; do
        if [[ " {{unbuilt_tools}} " == *" ${step} "* ]]; then
            echo "hoppar över ${step} (inte byggt än)"
        elif [ "$step" = check_coverage ]; then
            uv run tools/check_coverage.py coverage.json tools/thresholds.toml
        else
            uv run "tools/${step}.py"
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
gen:
    uv run --no-sync python tools/gen_schemas.py

alias gen-schemas := gen

# Fail if generated files are out of date
gen-check:
    #!/usr/bin/env bash
    set -euo pipefail
    {{_just}} gen
    changed="$(git status --porcelain -- software/common/schemas/generated \
      software/common/ts-schemas/src/generated software/ground/tests/contract/snapshots)"
    if [ -n "$changed" ]; then
        echo "Genererade filer är inte i takt med källan. Kör 'just gen' och committa:"
        echo "$changed"; git --no-pager diff --stat; exit 1
    fi

# --- database and services -----------------------------------------------------------

# Start Postgres + PostGIS on 127.0.0.1:5433 and wait until it is healthy
db-up: _env-file
    docker compose --env-file .env up -d --wait db

# Stop the database (the data volume is kept)
db-down: _env-file
    docker compose --env-file .env down db

# Show the PostGIS version of the running database
db-check: _env-file
    docker compose --env-file .env exec -T db psql -U sd_owner -d searchdrone -v ON_ERROR_STOP=1 -tAc 'SELECT postgis_full_version()'

# Create or upgrade the database schema (creates the role sd_app if missing)
migrate:
    uv run --package sd-ground sd-ground db setup

# Start MediaMTX: live video and per-mission recordings, every port on 127.0.0.1 (B1-09)
video-up: _env-file
    #!/usr/bin/env bash
    set -euo pipefail
    # The recordings folder must exist before the container starts, or Docker creates
    # it owned by root and MediaMTX (running as the owner's user) cannot write there.
    data="$(sed -n 's/^SD_DATA_DIR=//p' .env | tail -n 1)"
    [ -n "$data" ] || { echo "SD_DATA_DIR saknas i .env. Kör: just setup-config"; exit 1; }
    mkdir -p "$data/video"
    docker compose --env-file .env up -d --wait mediamtx

# Stop MediaMTX (the recordings are kept)
video-down: _env-file
    docker compose --env-file .env down mediamtx

# Video tests against MediaMTX (needs just video-up, db-up and migrate; keys: physics)
test-video *args:
    #!/usr/bin/env bash
    set -euo pipefail
    selection="video and not gz"
    for arg in "$@"; do
        case "$arg" in
            physics=builtin) ;;
            physics=gz) selection="video" ;;
            *) echo "test-video: okänd inställning '$arg' (physics=builtin eller physics=gz)"; exit 2 ;;
        esac
    done
    pnpm --filter @searchdrone/panel build
    uv run pytest -p sd_sim.pytest_plugin -m "$selection" simulation/tests/video

# Start the ground station (prints the login link)
ground:
    uv run --package sd-ground sd-ground serve

# Write a new commander token for sd-brain (restart sd-ground and sd-brain afterwards)
brain-token:
    uv run --no-sync python tools/brain_token.py

# --- AI commander (sd-brain) -----------------------------------------------------------

# Start the AI commander (reads config/brain.toml; keys from .env)
brain:
    uv run --package sd-brain sd-brain run --config config/brain.toml

# Start the local model server (llama-server, GPU) on 127.0.0.1:8091
llm-up: _env-file
    docker compose --env-file .env --profile ai up -d --wait llm

# Stop the local model server
llm-down: _env-file
    docker compose --env-file .env --profile ai stop llm

# Golden evaluation of the AI commander (no GPU; same as CI)
brain-eval:
    uv run --package sd-brain sd-brain eval --backend replay

# Model evaluation against the real local model, 3 runs (owner's computer, GPU)
brain-eval-model:
    #!/usr/bin/env bash
    set -euo pipefail
    gpu="$(nvidia-smi --query-gpu=name --format=csv,noheader 2>/dev/null | head -1 || true)"
    uv run --package sd-brain sd-brain eval --backend llm_local --repeat 3 --gpu-name "$gpu"

# Check the local model for 5 minutes: valid tool calls and graphics memory <= 7000 MB
check-model:
    #!/usr/bin/env bash
    set -euo pipefail
    mkdir -p data/reports
    log="$(mktemp)"
    nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits -lms 500 > "$log" &
    smi=$!
    trap 'kill "$smi" 2>/dev/null || true; rm -f "$log"' EXIT
    image="$(python3 -c 'import tomllib;print(tomllib.load(open("software/brain/runtime.lock.toml","rb"))["llama_cpp"]["image"])')"
    uv run --package sd-brain sd-brain check-model --duration-s 300 --interval-s 10 --vram-log "$log" --image "$image"

# --- panel ---------------------------------------------------------------------------

# Panel dev server on http://127.0.0.1:5173 (loopback only, proxies to sd-ground)
panel:
    pnpm --filter @searchdrone/panel dev

# Panel end-to-end tests against a fake ground station (same four steps as CI)
test-panel-e2e:
    pnpm --filter @searchdrone/panel build
    pnpm --filter @searchdrone/panel check
    pnpm --filter @searchdrone/panel build:test
    pnpm --filter @searchdrone/panel e2e

# The stack runs in the SITL container and Playwright beside it on this machine.
# Panel in a browser against SITL and the scripted commander (keys as ci-sitl)
test-panel-sitl *args:
    #!/usr/bin/env bash
    set -euo pipefail
    pnpm --filter @searchdrone/panel build:test
    ex=build/panel-sitl
    mkdir -p "$ex"
    rm -f "$ex/stack.json" "$ex/stop-brain" "$ex/done.json"
    {{_just}} ci-sitl mode=panel physics=builtin brain=scripted "$@" &
    stack=$!
    trap 'kill "$stack" 2>/dev/null || true' EXIT
    set +e
    pnpm --filter @searchdrone/panel e2e:sitl
    code=$?
    echo "{\"code\": $code}" > "$ex/done.json"
    wait "$stack"
    stack_code=$?
    set -e
    trap - EXIT
    [ "$code" -eq 0 ] && [ "$stack_code" -eq 0 ]

[private]
panel-build:
    pnpm --filter @searchdrone/panel build

# --- simulator -----------------------------------------------------------------------

# Run the whole stack against a fake vehicle (keys: physics, gui, brain, instance)
dev *args:
    uv run --no-sync python tools/just_args.py dev "$@"

# Start the simulator stack with Gazebo (keys: physics, gui, video, brain, ...)
sim *args:
    uv run --package sd-sim python -m sd_sim.launch up "$@"

alias sim-up := sim

# Like `sim` but builtin physics and a test picture as video
sim-headless *args:
    SD_SIM__HEADLESS=1 uv run --package sd-sim python -m sd_sim.launch up video=test "$@"

# Stop everything this instance started
stop *args:
    uv run --package sd-sim python -m sd_sim.launch down "$@"

alias sim-down := stop

# Check the simulator environment (keys: physics)
sim-doctor *args:
    uv run --package sd-sim python -m sd_sim.doctor "$@"

# Install ArduPilot SITL and Gazebo; safe to run again (asks for the Ubuntu password)
sim-setup:
    bash tools/setup/wsl-setup.sh sim

# GUIDED, arm, take off to 20 m, land (keys: physics, gui, instance)
sim-smoke *args:
    uv run --package sd-sim python -m sd_sim.smoke "$@"

# Scenario tests, headless (keys: sc, variant, physics, instance, shard, shards, ...)
test-sitl *args:
    uv run --package sd-sim python -m sd_sim.testrun "$@"

# Run the CI SITL job locally in the same container (keys as test-sitl, e.g. shard=1 shards=4)
ci-sitl *args:
    #!/usr/bin/env bash
    set -euo pipefail
    image="${SD_SITL_IMAGE:-$(bash tools/sitl_image_tag.sh --image)}"
    if ! docker image inspect "$image" >/dev/null 2>&1; then
        docker pull "$image" || docker build -f simulation/docker/sitl.Dockerfile -t "$image" .
    fi
    # The container runs as the calling user so files in runs/ and build/ stay theirs;
    # its virtualenv lives in the container, never in the checkout's .venv.
    # The Docker socket gives the container control of the host's Docker, so it is only
    # mounted for tests that start their own database container (private_db=true).
    sock=()
    if [[ " $* " == *" private_db=true "* ]]; then
        sock=(-v /var/run/docker.sock:/var/run/docker.sock \
            --group-add "$(stat -c %g /var/run/docker.sock)")
    fi
    docker run --rm --network host --user "$(id -u):$(id -g)" "${sock[@]}" \
        -v "$PWD:/w" -w /w -e SD_SIM__HEADLESS=1 -e HOME=/tmp/home \
        -e UV_PROJECT_ENVIRONMENT=/tmp/venv -e GITHUB_SHA -e SD_SIM__SPEEDUP -e SD_SIM__REQUIRE_SITL \
        "$image" bash -lc 'uv sync --all-packages --frozen && uv run --package sd-sim python -m sd_sim.testrun "$@"' testrun "$@"

# Full demo locked to the public CMAC home (keys: lan, record, ...)
demo *args:
    uv run --package sd-sim python -m sd_sim.launch up --demo "$@"

# Headless end-to-end run with the scripted commander (keys: runs, offline)
e2e *args:
    uv run --no-sync python tools/just_args.py e2e "$@"

# Performance measurement on the owner's computer (keys: runs)
perf *args:
    uv run --no-sync python tools/just_args.py perf "$@"

alias bench := perf

# Download the local AI model to models/ and check its sha256 (keys: fallback)
models-fetch *args:
    uv run --no-sync python tools/models_fetch.py "$@"

# Download the offline map pack to <data_dir>/tiles/ (keys: bbox, maxzoom, name, dry)
maps-fetch *args:
    uv run --no-sync python tools/maps/fetch.py "$@"

# --- hardware, acceptance and release ------------------------------------------------

# Check mass and flight time against hardware/drone.yaml
hw-check:
    uv run --package sd-hwcalc python -m sd_hwcalc check hardware/drone.yaml --limits hardware/limits.yaml

# Acceptance checks for a release (keys: phase, sha)
accept *args:
    uv run --no-sync python tools/just_args.py accept "$@"

# Traceability report for the scenario tests (local only)
check-traceability: (_not-built "B1-16")

# Record the showcase GIF for a stage (keys: brain)
showcase STAGE *args:
    uv run --no-sync python tools/just_args.py showcase stage="$1" "${@:2}"

# Set one version everywhere before a release
release-prep VERSION: (_not-built "B1-18")

# --- helpers -------------------------------------------------------------------------

[private]
_not-built ticket:
    @echo "inte byggt än ({{ticket}})"; exit 2

[private]
_env-file:
    @[ -f .env ] || { echo "Filen .env saknas. Kör: just setup-config"; exit 1; }

[private]
_step recipe:
    #!/usr/bin/env bash
    set -euo pipefail
    if [[ " {{unbuilt}} " == *" {{recipe}} "* ]]; then
        echo "hoppar över {{recipe}} (inte byggt än)"
    else
        {{_just}} {{recipe}}
    fi
