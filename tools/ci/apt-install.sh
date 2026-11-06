#!/usr/bin/env bash
# Install apt packages on a GitHub-hosted Ubuntu runner without letting a slow mirror eat
# the job. The runner's default mirror (azure.archive.ubuntu.com) has served packages at
# 20-60 kB/s, which ran CI steps into their timeouts before any test ran.
#
#   bash tools/ci/apt-install.sh PACKAGE...        update, download, install
#   bash tools/ci/apt-install.sh --update-only     update only (same mirror fallback)
#   bash tools/ci/apt-install.sh --fresh PACKAGE.. the lists were just updated: the first
#                                                  try skips its update
#   bash tools/ci/apt-install.sh --archive-mirror  only cut azure out of the mirror list,
#                                                  for a script that runs apt itself
#
# The first try uses the runner's default mirror with the download capped at 150 s. The
# two tries after it first cut azure out of the runner's mirror list, so they go to
# archive.ubuntu.com, with 240 s each. The tries only download (--download-only); what a
# try got stays in apt's archive folder for the next one. The install then runs once from
# those .debs with no time limit of its own (the step's timeout covers it): a timeout
# inside dpkg would leave it half done and fail everything after it. apt gives up on a
# dead connection after 30 s. The man-db index is not rebuilt.
set -euo pipefail

usage="usage: $0 [--fresh] PACKAGE... | --update-only | --archive-mirror"
# The tests point these at a scratch folder; on a runner they are the system's.
apt_dir="${APT_INSTALL_ETC_APT:-/etc/apt}"
man_db="${APT_INSTALL_MAN_DB:-/var/lib/man-db}"
mirrors="$apt_dir/apt-mirrors.txt"

mode=install
fresh=false
case "${1:-}" in
  --update-only | --archive-mirror)
    mode="${1#--}"
    shift
    if [ $# -ne 0 ]; then
      echo "$usage" >&2
      exit 2
    fi
    ;;
  --fresh)
    fresh=true
    shift
    ;;
esac
if [ "$mode" = install ] && [ $# -eq 0 ]; then
  echo "$usage" >&2
  exit 2
fi
for package in "$@"; do
  case "$package" in
    -*)
      echo "$usage" >&2
      exit 2
      ;;
  esac
done

apt_get() {
  sudo DEBIAN_FRONTEND=noninteractive "$@"
}
opts=(-o Acquire::Retries=3 -o Acquire::http::Timeout=30 -o Acquire::https::Timeout=30)

use_archive_mirror() {
  # The runner's sources read their mirrors from apt-mirrors.txt, azure's first.
  if [ -f "$mirrors" ]; then
    sudo sed -i '/azure\.archive\.ubuntu\.com/d' "$mirrors"
    if ! grep -q . "$mirrors"; then
      echo 'http://archive.ubuntu.com/ubuntu/' | sudo tee "$mirrors" >/dev/null
    fi
  fi
  for f in "$apt_dir"/sources.list.d/*.sources "$apt_dir/sources.list"; do
    if [ -f "$f" ]; then
      sudo sed -i 's#azure\.archive\.ubuntu\.com#archive.ubuntu.com#g' "$f"
    fi
  done
}

if [ "$mode" = archive-mirror ]; then
  # A script that installs with apt itself gets archive.ubuntu.com from the start; it
  # updates the lists on its own.
  echo "apt: dropping azure.archive.ubuntu.com, using archive.ubuntu.com"
  use_archive_mirror
  exit 0
fi

sudo rm -f "$man_db/auto-update"
for try in 1 2 3; do
  update=true
  if [ "$try" = 1 ]; then
    limit=150
    if $fresh; then update=false; fi
  else
    limit=240
    echo "apt: dropping azure.archive.ubuntu.com, trying archive.ubuntu.com"
    use_archive_mirror
  fi
  if { ! $update || apt_get timeout 90 apt-get "${opts[@]}" update -q; } && {
    [ "$mode" != install ] ||
      apt_get timeout "$limit" apt-get "${opts[@]}" install -y -q --download-only \
        --no-install-recommends "$@"
  }; then
    if [ "$mode" = install ]; then
      apt_get apt-get "${opts[@]}" install -y -q --no-install-recommends "$@"
    fi
    exit 0
  fi
  echo "apt try $try failed or timed out"
  if [ "$try" != 3 ]; then sleep 10; fi
done
exit 1
