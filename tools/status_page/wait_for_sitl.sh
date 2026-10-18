#!/usr/bin/env bash
# Find the run of the sitl workflow whose results the status page shows, and wait for it.
#
# Environment: GH_TOKEN, REPO (owner/name), EVENT (the status page's own event), SHA.
#   push, pull_request: the sitl run of the same commit and event, waited for until it
#                       is completed (at most WAIT_MIN minutes)
#   schedule, workflow_dispatch: the newest completed sitl run on main
# Writes run_id, run_url, sha and date (UTC) to $GITHUB_OUTPUT.
set -euo pipefail

WAIT_MIN="${WAIT_MIN:-110}"
api="repos/$REPO/actions/workflows/sitl.yml/runs"
case "$EVENT" in
    push | pull_request) query="$api?head_sha=$SHA&event=$EVENT&per_page=1" ;;
    schedule | workflow_dispatch) query="$api?branch=main&status=completed&per_page=1" ;;
    *) echo "wait_for_sitl.sh: okänd händelse $EVENT"; exit 2 ;;
esac

deadline=$(($(date +%s) + WAIT_MIN * 60))
while :; do
    run="$(gh api "$query" --jq '.workflow_runs[0] // empty')"
    if [ -n "$run" ] && [ "$(jq -r .status <<<"$run")" = completed ]; then
        break
    fi
    if [ "$(date +%s)" -ge "$deadline" ]; then
        echo "sitl-körningen för $SHA blev inte klar inom $WAIT_MIN min"
        exit 1
    fi
    echo "väntar på sitl ($(jq -r '.status // "inte startad"' <<<"${run:-null}"))"
    sleep 60
done

{
    echo "run_id=$(jq -r .id <<<"$run")"
    echo "run_url=$(jq -r .html_url <<<"$run")"
    echo "sha=$(jq -r .head_sha <<<"$run")"
    echo "date=$(jq -r '.updated_at | sub("T"; " ") | sub(":[0-9]{2}Z$"; " UTC")' <<<"$run")"
} | tee -a "$GITHUB_OUTPUT"
echo "sitl: $(jq -r .conclusion <<<"$run")"
