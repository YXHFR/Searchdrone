#!/usr/bin/env bash
# Find the run of the sitl workflow whose results the status page shows, and wait for it.
#
# Environment: GH_TOKEN, REPO (owner/name), EVENT (the status page's own event), SHA.
#   push, pull_request: the sitl run of the same commit and event, waited for until it
#                       is completed (at most WAIT_MIN minutes)
#   schedule, workflow_dispatch: the newest finished (success or failure) push or
#                       scheduled sitl run of this repository's own main; never a pull
#                       request's run, whose results its author controls
# Writes run_id, run_url, sha and date (UTC) to $GITHUB_OUTPUT.
set -euo pipefail

WAIT_MIN="${WAIT_MIN:-110}"
api="repos/$REPO/actions/workflows/sitl.yml/runs"
case "$EVENT" in
    push | pull_request)
        query="$api?head_sha=$SHA&event=$EVENT&per_page=1"
        pick='.workflow_runs[0] // empty'
        ;;
    schedule | workflow_dispatch)
        query="$api?branch=main&status=completed&per_page=50"
        pick='[.workflow_runs[] | select((.event == "push" or .event == "schedule")
            and .head_branch == "main"
            and (.head_repository.full_name | ascii_downcase) == ($repo | ascii_downcase)
            and (.conclusion == "success" or .conclusion == "failure"))][0] // empty'
        ;;
    *) echo "wait_for_sitl.sh: okänd händelse $EVENT"; exit 2 ;;
esac

deadline=$(($(date +%s) + WAIT_MIN * 60))
while :; do
    run="$(gh api "$query" | jq -c --arg repo "$REPO" "$pick")"
    if [ -n "$run" ] && [ "$(jq -r .status <<<"$run")" = completed ]; then
        case "$(jq -r .conclusion <<<"$run")" in
            success | failure) break ;;
            *) echo "sitl-körningen blev $(jq -r .conclusion <<<"$run"): inga resultat att visa"; exit 1 ;;
        esac
    fi
    if [ "$(date +%s)" -ge "$deadline" ]; then
        echo "sitl-körningen för $SHA blev inte klar inom $WAIT_MIN min"
        exit 1
    fi
    echo "väntar på sitl ($(jq -r '.status // "inte startad"' <<<"${run:-null}"))"
    sleep "${POLL_S:-60}"
done

{
    echo "run_id=$(jq -r .id <<<"$run")"
    echo "run_url=$(jq -r .html_url <<<"$run")"
    echo "sha=$(jq -r .head_sha <<<"$run")"
    echo "date=$(jq -r '.updated_at | sub("T"; " ") | sub(":[0-9]{2}Z$"; " UTC")' <<<"$run")"
} | tee -a "$GITHUB_OUTPUT"
echo "sitl: $(jq -r .conclusion <<<"$run")"
