#!/usr/bin/env bash
# Block the end-to-end stack's traffic to the internet, and only that (F-07, CI only).
#
#   bash tools/offline/isolate.sh user      # create the test user sdtest (prints SD_E2E_USER=uid:gid)
#   bash tools/offline/isolate.sh own       # give it the checkout and the data folder
#   bash tools/offline/isolate.sh block     # the firewall rules, then show they hold
#   bash tools/offline/isolate.sh unblock   # remove the rules, give the files back
#
# Everything the stack runs (the simulator container, MediaMTX) runs as sdtest, and its
# containers (the database) on the network sd-offline (compose.offline.yaml). The rules:
#   OUTPUT:      sdtest may only reach loopback (IPv4 and IPv6);
#   DOCKER-USER: 172.31.250.0/24 may only reach itself.
# The runner's own traffic (to GitHub, the image registry) is not touched. Before the
# test the block is shown to hold: curl as sdtest and from the network sd-offline both
# fail. Needs sudo; run it on a throwaway CI runner, never on your own computer.
set -euo pipefail

USER_NAME=sdtest
SUBNET=172.31.250.0/24
NETWORK=sd-offline
PROBE=https://example.com
# curlimages/curl:8.16.0, published 2025-09-12 (multi-arch index digest)
CURL_IMAGE=curlimages/curl@sha256:463eaf6072688fe96ac64fa623fe73e1dbe25d8ad6c34404a669ad3ce1f104b6
WORKSPACE="${GITHUB_WORKSPACE:-$PWD}"
DATA="${SD_DATA_DIR:-}"  # needed by own and unblock

rules() {
    local op="$1" ok=false  # adding: a rule that fails stops the script
    [ "$op" = -D ] && ok=:  # removing: each rule on its own, whatever is still there
    sudo iptables "$op" OUTPUT -m owner --uid-owner "$USER_NAME" ! -d 127.0.0.0/8 -j REJECT || $ok
    sudo ip6tables "$op" OUTPUT -m owner --uid-owner "$USER_NAME" ! -d ::1 -j REJECT || $ok
    if [ "$op" = -A ]; then
        sudo iptables -I DOCKER-USER -s "$SUBNET" ! -d "$SUBNET" -j REJECT
    else
        sudo iptables -D DOCKER-USER -s "$SUBNET" ! -d "$SUBNET" -j REJECT || :
    fi
}

case "${1:-}" in
    user)
        id "$USER_NAME" >/dev/null 2>&1 || sudo useradd --system --create-home "$USER_NAME"
        echo "SD_E2E_USER=$(id -u "$USER_NAME"):$(id -g "$USER_NAME")"
        ;;
    own)
        [ -n "$DATA" ] || { echo "SD_DATA_DIR saknas"; exit 1; }
        mkdir -p "$DATA"
        sudo chown -R "$USER_NAME:$USER_NAME" "$WORKSPACE" "$DATA"
        ;;
    block)
        docker pull -q "$CURL_IMAGE" >/dev/null  # before the block
        rules -A
        if sudo -u "$USER_NAME" curl -sS -m 5 -o /dev/null "$PROBE"; then
            echo "Spärren håller inte: $USER_NAME når $PROBE."; exit 1
        fi
        if docker run --rm --network "$NETWORK" "$CURL_IMAGE" -sS -m 5 -o /dev/null "$PROBE"; then
            echo "Spärren håller inte: nätet $NETWORK når $PROBE."; exit 1
        fi
        echo "Spärren håller: varken $USER_NAME eller nätet $NETWORK når $PROBE."
        ;;
    unblock)
        # Also after a step that failed early: remove what is there, give back what exists.
        if id "$USER_NAME" >/dev/null 2>&1; then
            rules -D 2>/dev/null
        fi
        for path in "$WORKSPACE" "$DATA"; do
            if [ -n "$path" ] && [ -e "$path" ]; then sudo chown -R "$(id -u):$(id -g)" "$path"; fi
        done
        ;;
    *)
        echo "användning: isolate.sh user|own|block|unblock" >&2; exit 2
        ;;
esac
