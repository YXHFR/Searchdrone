#!/usr/bin/env bash
# Print the tag of the SITL image: <ardupilot_tag>-<first 12 hex of a sha256>.
#
# The hash covers the content and path of every file under simulation/docker/ plus
# simulation/versions.lock, sorted by path, so the tag changes whenever anything that
# can change the image changes (the Dockerfile may copy only those files).
#
#   bash tools/sitl_image_tag.sh          -> Copter-4.7.1-0123456789ab
#   bash tools/sitl_image_tag.sh --image  -> ghcr.io/yxhfr/searchdrone-sitl:Copter-4.7.1-0123456789ab
#
# Run from anywhere; paths are resolved from the repository root.
set -euo pipefail

root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$root"
lock=simulation/versions.lock

value() {
    # key = "value"  (the lock file is TOML; only simple string keys are read here)
    local v
    v="$(sed -nE "s/^$1[[:space:]]*=[[:space:]]*\"([^\"]*)\".*/\\1/p" "$lock" | head -n 1)"
    [ -n "$v" ] || { echo "sitl_image_tag: $1 saknas i $lock" >&2; exit 1; }
    printf '%s' "$v"
}

tag="$(value ardupilot_tag)"
[[ "$tag" =~ ^[A-Za-z0-9._-]+$ ]] || { echo "sitl_image_tag: ogiltig ardupilot_tag" >&2; exit 1; }
hash="$( { find simulation/docker -type f -not -path '*/__pycache__/*'; echo "$lock"; } | LC_ALL=C sort \
    | xargs -d '\n' sha256sum | sha256sum | cut -c1-12)"

case "${1:-}" in
    "") echo "${tag}-${hash}" ;;
    --image) echo "$(value sitl_image):${tag}-${hash}" ;;
    *) echo "sitl_image_tag: okänt argument '$1' (bara --image)" >&2; exit 2 ;;
esac
