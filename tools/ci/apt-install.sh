#!/usr/bin/env bash
# Install apt packages on a GitHub-hosted Ubuntu runner without letting a slow mirror eat
# the job. The runner's default mirror (azure.archive.ubuntu.com) has served packages at
# 20-60 kB/s, which ran CI steps into their timeouts before any test ran.
#
#   bash tools/ci/apt-install.sh PACKAGE...    update, download, install
#   bash tools/ci/apt-install.sh --update-only update only (same mirror fallback)
#
# The first try uses the runner's default mirror with the download capped at 150 s. The
# two tries after it first cut azure out of the runner's mirror list, so they go to
# archive.ubuntu.com, with 240 s each. The tries only download (--download-only); what a
# try got stays in apt's archive folder for the next one. The install then runs once from
# those .debs with no time limit of its own (the step's timeout covers it): a timeout
# inside dpkg would leave it half done and fail everything after it. apt gives up on a
# dead connection after 30 s. The man-db index is not rebuilt.
set -euo pipefail

update_only=false
if [ "${1:-}" = --update-only ]; then
  update_only=true
  shift
fi
if ! $update_only && [ $# -eq 0 ]; then
  echo "usage: $0 PACKAGE... | --update-only" >&2
  exit 2
fi

apt_get() {
  sudo DEBIAN_FRONTEND=noninteractive "$@"
}
opts=(-o Acquire::Retries=3 -o Acquire::http::Timeout=30 -o Acquire::https::Timeout=30)

use_archive_mirror() {
  # The runner's sources read their mirrors from apt-mirrors.txt, azure's first.
  if [ -f /etc/apt/apt-mirrors.txt ]; then
    sudo sed -i '/azure\.archive\.ubuntu\.com/d' /etc/apt/apt-mirrors.txt
    if ! grep -q . /etc/apt/apt-mirrors.txt; then
      echo 'http://archive.ubuntu.com/ubuntu/' | sudo tee /etc/apt/apt-mirrors.txt >/dev/null
    fi
  fi
  for f in /etc/apt/sources.list.d/*.sources /etc/apt/sources.list; do
    if [ -f "$f" ]; then
      sudo sed -i 's#azure\.archive\.ubuntu\.com#archive.ubuntu.com#g' "$f"
    fi
  done
}

sudo rm -f /var/lib/man-db/auto-update
for try in 1 2 3; do
  if [ "$try" = 1 ]; then
    limit=150
  else
    limit=240
    echo "apt: dropping azure.archive.ubuntu.com, trying archive.ubuntu.com"
    use_archive_mirror
  fi
  if apt_get timeout 90 apt-get "${opts[@]}" update -q && {
    $update_only ||
      apt_get timeout "$limit" apt-get "${opts[@]}" install -y -q --download-only \
        --no-install-recommends "$@"
  }; then
    if ! $update_only; then
      apt_get apt-get "${opts[@]}" install -y -q --no-install-recommends "$@"
    fi
    exit 0
  fi
  echo "apt try $try failed or timed out"
  if [ "$try" != 3 ]; then sleep 10; fi
done
exit 1
