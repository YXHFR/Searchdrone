#!/usr/bin/env bash
# One-time setup of WSL2 Ubuntu 24.04 for Searchdrone. Run inside Ubuntu, in ~/Searchdrone.
# Skeleton: Docker Engine, uv and just. The simulator part (ArduPilot SITL, Gazebo)
# is added in a later step; `just sim-setup` installs it.
set -euo pipefail

cd "$(dirname "${BASH_SOURCE[0]}")/../.."
export PATH="$HOME/.local/bin:$PATH"
ver() { python3 tools/versions.py "$1"; }

# 0. Checks
grep -q 'VERSION_ID="24.04"' /etc/os-release || { echo "Kräver Ubuntu 24.04"; exit 1; }
[ "$(pwd)" = "${HOME}/Searchdrone" ] || echo "Varning: repot bör ligga i ~/Searchdrone (inte under /mnt/c)"
if [ -f /etc/wsl.conf ] && ! grep -q "systemd=true" /etc/wsl.conf; then
    echo "Varning: systemd är av i WSL (behövs för Docker)"
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
        echo "deb [arch=$(dpkg --print-architecture) signed-by=/etc/apt/keyrings/docker.asc] https://download.docker.com/linux/ubuntu $(. /etc/os-release && echo "$VERSION_CODENAME") stable" \
            | sudo tee /etc/apt/sources.list.d/docker.list > /dev/null
        sudo apt-get update
        sudo apt-get install -y docker-ce docker-ce-cli containerd.io docker-buildx-plugin docker-compose-plugin
    fi
    if ! id -nG "$USER" | grep -qw docker; then
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
