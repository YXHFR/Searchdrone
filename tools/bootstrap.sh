#!/usr/bin/env bash
# Fresh WSL2 Ubuntu 24.04 (or CI / cloud container) -> ready development environment.
# Idempotent and non-interactive: `bash tools/bootstrap.sh </dev/null` must work.
# Uses sudo only for apt. Expects git, curl and python3 (Ubuntu 24.04 has 3.12).
# The simulator, AI model and maps are NOT downloaded here (just sim-setup,
# just models-fetch, just maps-fetch).
set -euo pipefail

export PATH="$HOME/.local/bin:$HOME/.local/share/searchdrone/node/bin:$PATH"
export COREPACK_ENABLE_DOWNLOAD_PROMPT=0
export DEBIAN_FRONTEND=noninteractive

cd "$(dirname "${BASH_SOURCE[0]}")/.."
ver() { python3 tools/versions.py "$1"; }
say() { printf '\033[1m==> %s\033[0m\n' "$*"; }
warn() { printf '\033[33mVarning: %s\033[0m\n' "$*"; }

# 1. System packages (only what is missing)
missing=()
for pkg in build-essential openssl jq ffmpeg; do
    dpkg -s "$pkg" >/dev/null 2>&1 || missing+=("$pkg")
done
if [ "${#missing[@]}" -gt 0 ]; then
    say "apt: ${missing[*]}"
    if [ "$(id -u)" -eq 0 ]; then SUDO=(); else SUDO=(sudo -n); fi
    if "${SUDO[@]}" apt-get update -qq && "${SUDO[@]}" apt-get install -y -qq "${missing[@]}"; then
        :
    else
        warn "kunde inte installera ${missing[*]} (kör: sudo apt-get install ${missing[*]})"
    fi
fi

# 2. uv, exact version with checksum
UV_VERSION="$(ver uv_version)"
if ! uv --version 2>/dev/null | grep -q "^uv ${UV_VERSION}\b"; then
    say "uv ${UV_VERSION}"
    tmp="$(mktemp -d)"
    tgz="uv-x86_64-unknown-linux-gnu.tar.gz"
    curl -fsSL -o "$tmp/$tgz" "https://github.com/astral-sh/uv/releases/download/${UV_VERSION}/${tgz}"
    echo "$(ver uv_sha256)  $tmp/$tgz" | sha256sum -c --quiet - || { echo "Fel kontrollsumma för uv"; exit 1; }
    tar -xzf "$tmp/$tgz" -C "$tmp"
    mkdir -p "$HOME/.local/bin"
    install -m 0755 "$tmp/uv-x86_64-unknown-linux-gnu/uv" "$tmp/uv-x86_64-unknown-linux-gnu/uvx" "$HOME/.local/bin/"
    rm -rf "$tmp"
fi

# 3. just and pre-commit as uv tools
JUST_VERSION="$(ver just_version)"
just --version 2>/dev/null | grep -q "^just ${JUST_VERSION}$" || uv tool install --force "rust-just==${JUST_VERSION}"
PRE_COMMIT_VERSION="$(ver pre_commit_version)"
pre-commit --version 2>/dev/null | grep -q " ${PRE_COMMIT_VERSION}$" || uv tool install --force "pre-commit==${PRE_COMMIT_VERSION}"

# 4. Node, exact version from nodejs.org with checksum
NODE_VERSION="$(ver node_version)"
if ! node --version 2>/dev/null | grep -q "^v${NODE_VERSION%%.*}\."; then
    say "Node ${NODE_VERSION}"
    tmp="$(mktemp -d)"
    txz="node-v${NODE_VERSION}-linux-x64.tar.xz"
    curl -fsSL -o "$tmp/$txz" "https://nodejs.org/dist/v${NODE_VERSION}/${txz}"
    curl -fsSL -o "$tmp/SHASUMS256.txt" "https://nodejs.org/dist/v${NODE_VERSION}/SHASUMS256.txt"
    (cd "$tmp" && grep " ${txz}\$" SHASUMS256.txt | sha256sum -c --quiet -) || { echo "Fel kontrollsumma för Node"; exit 1; }
    dest="$HOME/.local/share/searchdrone"
    mkdir -p "$dest"
    rm -rf "$dest/node"
    tar -xJf "$tmp/$txz" -C "$dest"
    mv "$dest/node-v${NODE_VERSION}-linux-x64" "$dest/node"
    rm -rf "$tmp"
fi

# 5. pnpm through Corepack (version from package.json "packageManager"), no sudo
corepack enable --install-directory "$HOME/.local/bin"

# 6. Cloud sessions get an empty private pattern file; the owner's real patterns
#    exist only in their own clone.
if [ "${CLAUDE_CODE_REMOTE:-}" = "true" ] && [ ! -f config/local/private-patterns.txt ]; then
    mkdir -p config/local
    : > config/local/private-patterns.txt
fi

# 7. Dependencies, git hooks and local config
just setup

if [ ! -f config/local/private-patterns.txt ]; then
    warn "config/local/private-patterns.txt saknas. Skapa den (får vara tom) och skriv dina privata"
    warn "sökmönster där, en per rad. Utan filen stoppar git-krokarna varje commit och push."
fi

cat <<'MSG'

Klart. Lägg de här raderna i ~/.bashrc så att verktygen hittas i nya terminaler:

    export PATH="$HOME/.local/bin:$HOME/.local/share/searchdrone/node/bin:$PATH"
    export COREPACK_ENABLE_DOWNLOAD_PROMPT=0

Kör sedan: just check
MSG
