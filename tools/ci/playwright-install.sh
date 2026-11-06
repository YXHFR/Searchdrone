#!/usr/bin/env bash
# Playwright's browsers and the system packages they need, on a GitHub-hosted runner.
# Same as `playwright install --with-deps BROWSER...`, but the packages go through
# tools/ci/apt-install.sh, so a slow apt mirror falls back to archive.ubuntu.com instead
# of eating the job. Playwright names the packages (`install-deps --dry-run` lists the
# ones missing); this script only changes how they are fetched.
#
#   bash tools/ci/playwright-install.sh chromium firefox webkit
set -euo pipefail

if [ $# -eq 0 ]; then
  echo "usage: $0 BROWSER..." >&2
  exit 2
fi
here="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
pw=(pnpm --filter @searchdrone/panel exec playwright)

bash "$here/apt-install.sh" --update-only
# Exits 1 when packages are missing; anything but its two known answers is an error.
report="$("${pw[@]}" install-deps --dry-run "$@" 2>&1)" || true
if ! grep -qE '^(Missing system dependencies|All system dependencies are installed)' <<<"$report"; then
  echo "$report" >&2
  echo "playwright install-deps --dry-run gave no package list" >&2
  exit 1
fi
mapfile -t packages < <(sed -n 's/^  \([^ ]\+\)$/\1/p' <<<"$report")
if [ "${#packages[@]}" -gt 0 ]; then
  bash "$here/apt-install.sh" "${packages[@]}"
fi
"${pw[@]}" install "$@"
