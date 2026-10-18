#!/usr/bin/env bash
# Cut the simulator stack off from the internet for `just e2e offline=1` (CI and locally).
#
#   sudo bash tools/offline/isolate.sh prepare   # test user, files, images (before the cut)
#   sudo bash tools/offline/isolate.sh seal      # firewall rules for the test user and subnet
#   sudo bash tools/offline/isolate.sh check     # proves the cut: the probes must fail
#   sudo bash tools/offline/isolate.sh down      # removes the rules, gives the files back
#
# Only the stack's own traffic is refused: everything the user `sdtest` sends outside
# loopback (the SITL container runs as that user in the host network, and so does
# MediaMTX), and everything the database's network `sd-offline` (172.31.250.0/24, from
# compose.offline.yaml) sends outside its subnet. The machine's other traffic, such as a
# CI runner talking to GitHub, is untouched. Run from the repository root.
set -euo pipefail

USER_NAME=sdtest
SUBNET=172.31.250.0/24
CURL_IMAGE="curlimages/curl:8.17.0@sha256:935d9100e9ba842cdb060de42472c7ca90cfe9a7c96e4dacb55e79e560b3ff40"
PROBE_URL="https://example.com"
V4_RULE=(OUTPUT -m owner --uid-owner "$USER_NAME" ! -d 127.0.0.0/8 -j REJECT)
V6_RULE=(OUTPUT -m owner --uid-owner "$USER_NAME" ! -d ::1 -j REJECT)
NET_RULE=(DOCKER-USER -s "$SUBNET" ! -d "$SUBNET" -j REJECT)

say() { printf '\033[1m==> %s\033[0m\n' "$*"; }
die() { echo "isolate: $*" >&2; exit 1; }

[ "$(id -u)" -eq 0 ] || die "kör med sudo"
[ -f compose.offline.yaml ] || die "kör från repots rot"
[[ "${SUDO_UID:-}" =~ ^[0-9]+$ && "${SUDO_GID:-}" =~ ^[0-9]+$ ]] \
    || die "kör med sudo från din vanliga användare (filerna lämnas tillbaka till den)"
owner="$SUDO_UID:$SUDO_GID"
data="$(sed -n 's/^SD_DATA_DIR=//p' .env 2>/dev/null | tail -n 1)"
[ -n "$data" ] || die "SD_DATA_DIR saknas i .env (kör just setup-config)"
# The folder is re-owned recursively: only an absolute folder of its own, never / or a
# home or system folder.
home="$(getent passwd "$SUDO_UID" | cut -d: -f6)"
case "$data" in
    /*) ;;
    *) die "SD_DATA_DIR ska vara en absolut sökväg" ;;
esac
data="$(realpath -m "$data")"
case "$data" in
    /home/*/*) ;;  # a folder inside a home folder
    / | /root | /home | /home/* | /usr | /usr/* | /etc | /etc/* | /var | /tmp | /opt | /srv | \
    /bin | /bin/* | /sbin | /sbin/* | /lib* | /boot* | /proc* | /sys* | /dev* | /run*)
        die "SD_DATA_DIR $data är för brett för att byta ägare på" ;;
esac
[ "$data" != "$home" ] || die "SD_DATA_DIR får inte vara hemmappen"

# The folders the stack writes. The test user gets its own 0600 copy of .env (the ground
# station refuses a .env that others can read); `just ci-sitl` mounts it over .env.
writable=(runs build "$data")
env_copy=build/offline/env
# The local settings (config/<part>.toml from `just setup-config`, 0600) are lent to
# the test user for the run and given back by `down`.
shopt -s nullglob
local_config=()
for f in config/*.toml; do
    [[ "$f" == *.example.toml ]] || local_config+=("$f")
done
shopt -u nullglob

case "${1:-}" in
    prepare)
        if ! id "$USER_NAME" >/dev/null 2>&1; then
            say "skapar användaren $USER_NAME"
            useradd --system --no-create-home --shell /usr/sbin/nologin "$USER_NAME"
        fi
        mkdir -p "${writable[@]}" "$data/video"
        install -d build/offline/venv  # the virtualenv, filled before the cut
        install -m 600 -o "$USER_NAME" -g "$USER_NAME" .env "$env_copy"
        chown -R "$USER_NAME:$USER_NAME" "${writable[@]}"
        [ "${#local_config[@]}" -eq 0 ] || chown "$USER_NAME:$USER_NAME" "${local_config[@]}"
        say "hämtar avbilder före spärren"
        docker pull -q "$CURL_IMAGE" >/dev/null
        echo "SD_OFFLINE_UID=$(id -u "$USER_NAME")"
        echo "SD_OFFLINE_GID=$(id -g "$USER_NAME")"
        ;;
    seal)
        docker network inspect sd-offline >/dev/null 2>&1 \
            || die "nätet sd-offline saknas (starta databasen med compose.offline.yaml först)"
        iptables -C "${V4_RULE[@]}" 2>/dev/null || iptables -A "${V4_RULE[@]}"
        ip6tables -C "${V6_RULE[@]}" 2>/dev/null || ip6tables -A "${V6_RULE[@]}"
        iptables -C "${NET_RULE[@]}" 2>/dev/null || iptables -I "${NET_RULE[@]}"
        say "spärrat: $USER_NAME utanför loopback och $SUBNET utanför nätet"
        ;;
    check)
        if sudo -u "$USER_NAME" env -i PATH=/usr/bin:/bin curl -sS -m 5 -o /dev/null "$PROBE_URL"; then
            die "$USER_NAME når $PROBE_URL: spärren fungerar inte"
        fi
        if docker run --rm --network sd-offline "$CURL_IMAGE" -sS -m 5 -o /dev/null "$PROBE_URL"; then
            die "nätet sd-offline når $PROBE_URL: spärren fungerar inte"
        fi
        say "spärren håller: varken $USER_NAME eller sd-offline når internet"
        ;;
    down)
        while iptables -C "${V4_RULE[@]}" 2>/dev/null; do iptables -D "${V4_RULE[@]}"; done
        while ip6tables -C "${V6_RULE[@]}" 2>/dev/null; do ip6tables -D "${V6_RULE[@]}"; done
        while iptables -C "${NET_RULE[@]}" 2>/dev/null; do iptables -D "${NET_RULE[@]}"; done
        rm -f "$env_copy"
        for dir in "${writable[@]}"; do
            [ -e "$dir" ] && chown -R "$owner" "$dir"
        done
        [ "${#local_config[@]}" -eq 0 ] || chown "$owner" "${local_config[@]}"
        say "spärren borttagen"
        ;;
    *)
        die "ange prepare, seal, check eller down"
        ;;
esac
