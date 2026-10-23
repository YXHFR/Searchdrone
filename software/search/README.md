# sd-search (`sd_search`)

The search service `sd-search`. It turns the target detections the ground station stores
into **candidates**: places where someone may be, for a person to look at. A detection is
only a hint, and a candidate is only a place in a queue: nothing here decides that the
person is found, and nothing here can fly the drone.

Part of the [Searchdrone](../../README.md) monorepo. Licensed under Apache-2.0.

Run it with `just search` (needs `just search-token` once, and a running `sd-ground`).
Its checks run from the repository root: `just test` (all checks: `just check`).

## What it does

1. **Reads** the open missions and their target detections from the ground station's
   search API (`/search/v1` on the loopback listener, with its own token). It has no
   route that gives orders, approves, presses STOP or decides on a candidate.
2. **Fuses** detections of the same class into candidates (`fusion.py`): a detection joins
   the nearest open candidate within 4 m plus the larger 95 % radius (at most 12 m),
   otherwise it starts a new one. A single sighting is never dropped. Confirmed and
   rejected candidates take no new sightings. Two candidates that drift together are
   merged (the older stays) unless a person has decided on either.
3. **Scores the colours** in the crops against the approved search profile (`color.py`,
   `score.py`): `match`, `conflict` or `not_visible` per field, from the share of each
   colour among the pixels that differ from the background. Only an approved profile
   version is used; without one, only the colour shares are recorded.
4. **Ranks** each candidate for the review queue (`priority.py`). The priority is an
   order, never a filter: every candidate stays in the queue.
5. **Writes** new candidates and revisions in one batch per round. The ground station
   checks every write: the search service can never set `confirmed` or `rejected`, nor
   change a candidate a person or the pilot holds.

## If it stops

Nothing else stops with it: it has no part in flying, the gate or the emergency path. The
panel shows `parts.search` as `down` after 15 s without a heartbeat. On a restart it reads
its candidates back and goes on from the last detection it fused: the same detection never
gives a second observation, and detections that arrived meanwhile are fused then.

## Simulator

`--profile-file` (or `SD_SEARCH__PROFILE_FILE`) reads an approved profile from a JSON file
for every mission. It only works with `SD_SIM=1`: on a real search the profile always
comes from the ground station.
