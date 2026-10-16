#!/usr/bin/env bash
# One-time setup of WSL2 Ubuntu 24.04 for Searchdrone. Run inside Ubuntu, in ~/Searchdrone.
#
#   bash tools/setup/wsl-setup.sh        Docker Engine, uv and just (before bootstrap.sh)
#   bash tools/setup/wsl-setup.sh sim    the simulator: ArduPilot SITL, Gazebo Harmonic,
#                                        ArduPilot's Gazebo plugin (same as `just sim-setup`)
#
# Both modes are idempotent: steps that are already done are skipped, so running the
# script twice in a row is safe. Versions come from simulation/versions.lock.
set -euo pipefail

cd "$(dirname "${BASH_SOURCE[0]}")/../.."
REPO="$(pwd)"
export PATH="$HOME/.local/bin:$PATH"
MODE="${1:-base}"
ver() { python3 tools/versions.py "$1"; }
say() { printf '\033[1m==> %s\033[0m\n' "$*"; }
warn() { printf '\033[33mVarning: %s\033[0m\n' "$*"; }

# 0. Checks
grep -q 'VERSION_ID="24.04"' /etc/os-release || { echo "Kräver Ubuntu 24.04"; exit 1; }
[ "$(pwd)" = "${HOME}/Searchdrone" ] || warn "repot bör ligga i ~/Searchdrone (inte under /mnt/c)"
case "$(pwd)" in /mnt/*) echo "Klona repot inuti Linux (~/Searchdrone), inte under /mnt/c."; exit 1 ;; esac

base_setup() {
    if [ -f /etc/wsl.conf ] && ! grep -q "systemd=true" /etc/wsl.conf; then
        warn "systemd är av i WSL (behövs för Docker)"
    fi

    # 1. Base packages
    sudo apt-get update
    sudo apt-get install -y git curl ca-certificates gnupg lsb-release build-essential \
        python3-venv python3-dev iproute2 jq openssl

    # 2. Docker Engine in WSL (not Docker Desktop), as documented on docs.docker.com for Ubuntu
    if ! docker info >/dev/null 2>&1; then
        if ! command -v docker >/dev/null 2>&1; then
            sudo install -m 0755 -d /etc/apt/keyrings
            sudo curl -fsSL https://download.docker.com/linux/ubuntu/gpg -o /etc/apt/keyrings/docker.asc
            sudo chmod a+r /etc/apt/keyrings/docker.asc
            # Docker's published signing key fingerprint (docs.docker.com)
            gpg --show-keys --with-colons /etc/apt/keyrings/docker.asc | grep -q '^fpr:::::::::9DC858229FC7DD38854AE2D88D81803C0EBFCD88:' \
                || { echo "Fel fingeravtryck på Dockers nyckel"; exit 1; }
            echo "deb [arch=$(dpkg --print-architecture) signed-by=/etc/apt/keyrings/docker.asc] https://download.docker.com/linux/ubuntu $(. /etc/os-release && echo "$VERSION_CODENAME") stable" \
                | sudo tee /etc/apt/sources.list.d/docker.list > /dev/null
            sudo apt-get update
            sudo apt-get install -y docker-ce docker-ce-cli containerd.io docker-buildx-plugin docker-compose-plugin
        fi
        if ! id -nG "$USER" | grep -qw docker; then
            # Note: membership in the docker group gives root-equivalent access on this machine.
            sudo usermod -aG docker "$USER"
            echo "Kör wsl --shutdown i PowerShell, öppna Ubuntu och kör skriptet igen."
            exit 0   # the group applies only after a new login
        fi
    fi

    # 3. uv (exact version, checksum) and just
    UV_VERSION="$(ver uv_version)"
    if ! uv --version 2>/dev/null | grep -q "^uv ${UV_VERSION}\b"; then
        tmp="$(mktemp -d)"
        tgz="uv-x86_64-unknown-linux-gnu.tar.gz"
        curl -fsSL -o "$tmp/$tgz" "https://github.com/astral-sh/uv/releases/download/${UV_VERSION}/${tgz}"
        echo "$(ver uv_sha256)  $tmp/$tgz" | sha256sum -c - || { echo "Fel kontrollsumma för uv"; exit 1; }
        tar -xzf "$tmp/$tgz" -C "$tmp"
        mkdir -p "$HOME/.local/bin"
        install -m 0755 "$tmp/uv-x86_64-unknown-linux-gnu/uv" "$tmp/uv-x86_64-unknown-linux-gnu/uvx" "$HOME/.local/bin/"
        rm -rf "$tmp"
    fi
    uv tool install --force "rust-just==$(ver just_version)"

    echo "Klart. Kör nu: bash tools/bootstrap.sh"
}

sim_setup() {
    local tag commit gz_commit ap="$HOME/ardupilot" agz="$HOME/gz_ws/src/ardupilot_gazebo"
    tag="$(ver ardupilot_tag)"
    commit="$(ver ardupilot_commit)"
    gz_commit="$(ver ardupilot_gazebo_commit)"

    # 1. Packages: GStreamer (test video, camera stream), mesa-utils (glxinfo)
    say "Paket för video och grafik"
    sudo apt-get update
    sudo apt-get install -y git curl ca-certificates gnupg lsb-release build-essential cmake \
        mesa-utils python3-venv python3-dev iproute2 \
        gstreamer1.0-tools gstreamer1.0-plugins-base gstreamer1.0-plugins-good \
        gstreamer1.0-plugins-bad gstreamer1.0-plugins-ugly gstreamer1.0-libav \
        gstreamer1.0-gl gstreamer1.0-rtsp python3-gi gir1.2-gstreamer-1.0

    # 2. Graphics check (WSLg); only warnings
    glxinfo -B 2>/dev/null | grep -E "OpenGL renderer" || warn "glxinfo visar ingen grafik"
    command -v nvidia-smi >/dev/null && nvidia-smi -L || warn "inget NVIDIA-kort syns i WSL (se reservvägarna i guiden)"

    # 3. ArduPilot SITL at the locked tag, built under ~/ardupilot (never under /mnt/c)
    if [ -f "$ap/.sd_version" ] && [ "$(cat "$ap/.sd_version")" = "$tag" ] \
        && [ -x "$ap/build/sitl/bin/arducopter" ]; then
        say "ArduPilot $tag finns redan"
    else
        if [ -d "$ap/.git" ]; then
            [ "$(git -C "$ap" rev-parse HEAD)" = "$commit" ] \
                || { echo "$ap finns men är inte $tag. Flytta mappen och kör igen."; exit 1; }
        else
            say "Hämtar ArduPilot $tag"
            git clone --branch "$tag" --depth 1 --recurse-submodules --shallow-submodules \
                https://github.com/ArduPilot/ardupilot.git "$ap"
            [ "$(git -C "$ap" rev-parse HEAD)" = "$commit" ] || { echo "Fel ArduPilot-commit"; exit 1; }
        fi
        say "ArduPilots verktyg (tar några minuter)"
        # DO_PYTHON_VENV_ENV=0: ArduPilot's venv is not made the default Python, so uv
        # keeps using its own Python for Searchdrone.
        (cd "$ap" && DO_AP_STM_ENV=0 DO_PYTHON_VENV_ENV=0 SKIP_AP_GIT_CHECK=1 \
            Tools/environment_install/install-prereqs-ubuntu.sh -y)
        say "Bygger ArduCopter SITL"
        (cd "$ap" && PATH="$HOME/venv-ardupilot/bin:$PATH" ./waf configure --board sitl \
            && PATH="$HOME/venv-ardupilot/bin:$PATH" ./waf copter)
        # The tag is written as is: several tags point at the same commit, so
        # `git describe` could name another one.
        echo "$tag" > "$ap/.sd_version"
    fi

    # 4. Gazebo Harmonic from the OSRF apt repository
    if ! dpkg -s gz-harmonic >/dev/null 2>&1; then
        say "Gazebo Harmonic"
        tmpkey="$(mktemp)"
        curl -fsSL https://packages.osrfoundation.org/gazebo.gpg -o "$tmpkey"
        # OSRF's published signing key fingerprint (gazebosim.org install docs)
        gpg --show-keys --with-colons "$tmpkey" | grep -q '^fpr:::::::::D2486D2DD83DB69272AFE98867170598AF249743:' \
            || { echo "Fel fingeravtryck på Gazebos nyckel"; exit 1; }
        sudo install -m 0644 "$tmpkey" /usr/share/keyrings/pkgs-osrf-archive-keyring.gpg
        rm -f "$tmpkey"
        echo "deb [arch=$(dpkg --print-architecture) signed-by=/usr/share/keyrings/pkgs-osrf-archive-keyring.gpg] http://packages.osrfoundation.org/gazebo/ubuntu-stable $(lsb_release -cs) main" \
            | sudo tee /etc/apt/sources.list.d/gazebo-stable.list > /dev/null
        sudo apt-get update
    fi
    sudo apt-get install -y gz-harmonic libgz-sim8-dev rapidjson-dev \
        libopencv-dev libgstreamer1.0-dev libgstreamer-plugins-base1.0-dev

    # 5. ArduPilot's Gazebo plugin at the locked commit
    if [ ! -d "$agz/.git" ]; then
        mkdir -p "$(dirname "$agz")"
        git clone https://github.com/ArduPilot/ardupilot_gazebo.git "$agz"
    fi
    if [ "$(git -C "$agz" rev-parse HEAD)" != "$gz_commit" ] || [ ! -f "$agz/build/libArduPilotPlugin.so" ]; then
        say "ArduPilots Gazebo-plugin"
        git -C "$agz" fetch --quiet origin
        git -C "$agz" checkout --quiet "$gz_commit"
        (cd "$agz" && GZ_VERSION=harmonic cmake -S . -B build -DCMAKE_BUILD_TYPE=RelWithDebInfo \
            && cmake --build build -j"$(nproc)")
    fi

    # 6. Environment: ONE file that ~/.bashrc reads
    cat > "$HOME/.searchdrone_env" <<EOF
# Written by tools/setup/wsl-setup.sh sim. Read by ~/.bashrc.
export GZ_VERSION=harmonic
export GZ_IP=127.0.0.1                     # Gazebo Transport on loopback only
export ARDUPILOT_HOME="\$HOME/ardupilot"
export SD_ARDUCOPTER_BIN="\$HOME/ardupilot/build/sitl/bin/arducopter"
export SD_ARDUPILOT_GAZEBO_HOME="$agz"
export GZ_SIM_SYSTEM_PLUGIN_PATH="$agz/build:\${GZ_SIM_SYSTEM_PLUGIN_PATH:-}"
export GZ_SIM_RESOURCE_PATH="$REPO/simulation/gazebo/models:$REPO/build/sim/models:$agz/models:$agz/worlds:\${GZ_SIM_RESOURCE_PATH:-}"
EOF
    grep -q searchdrone_env "$HOME/.bashrc" || echo '. "$HOME/.searchdrone_env"' >> "$HOME/.bashrc"
    # shellcheck source=/dev/null
    . "$HOME/.searchdrone_env"

    # 7. Check the result
    if ! id -nG | grep -qw docker; then
        echo "Kör wsl --shutdown i PowerShell, öppna Ubuntu och kör just sim-doctor."
        exit 0
    fi
    say "Kontroll"
    just sim-doctor || warn "sim-doctor hittade fel, se tabellen ovan"
    echo "Klart. Öppna en ny terminal (eller kör: . ~/.searchdrone_env) och prova: just sim-smoke"
}

case "$MODE" in
    base) base_setup ;;
    sim) sim_setup ;;
    *) echo "Okänt läge '$MODE'. Använd: bash tools/setup/wsl-setup.sh [sim]"; exit 2 ;;
esac
