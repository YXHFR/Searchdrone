# sd-ground (`sd_ground`)

The ground station `sd-ground`: API for the panel, database, safety gate, emergency path (STOP / RETURN HOME) and video.

Part of the [Searchdrone](../../README.md) monorepo. Licensed under Apache-2.0.

Start it from the repository root with `just ground` (all checks: `just check`).

## Running the server

1. `just setup-config` (or `sd-ground init`) creates `config/ground.toml`, `.env` with random
   passwords and `secrets/brain_token` for the AI commander.
2. `just db-up`, then `just migrate` (`sd-ground db setup`: creates the role `sd_app` if
   missing and migrates to the newest schema).
3. `uv run sd-ground vehicle add sim-01 --name "Sim 1" --if-missing`.
4. `just ground` starts the server and prints a one-time login link
   (`http://localhost:8080/#login=...`). A new link: `sd-ground login-link`.

`sd-ground config check` lists everything that would stop the server (non-loopback
addresses, missing or readable secrets, unknown settings, remote access switched on).

## Listeners

| Port | What | Who |
|---|---|---|
| `127.0.0.1:8080` | panel API `/api/v1`, live WebSocket `/ws/v1/live` | the pilot (login link or PIN), observers |
| `0.0.0.0:8443` | the same over TLS, only with `lan.enabled = true` (`sd-ground init --lan` makes the certificate; `sd-ground pair` prints a PIN) | the pilot's phone |
| `127.0.0.1:7700` | commander API `/commander/v1` | only `sd-brain`, with the token in `secrets/brain_token` |

None of these ports may ever be exposed to the internet.

Every panel request needs a session (also from `127.0.0.1`), changing requests need the
header `X-SD-Client: panel` and JSON, the `Host` must be on the allow-list and a foreign
`Origin` is refused. The commander token is refused on the panel ports, and the commander
port has no route for anything only the pilot may do (approvals, STOP / RETURN HOME /
Resume, clearance, limits, autonomy, the AI switch).

STOP and RETURN HOME are handled in a thread of their own (the ingress thread), so they do
not wait for the database, the AI or the rest of the server. A logged-in pilot session does
not expire while a vehicle is in the air.

## The emergency path

The ground station has its own MAVLink channel to the flight controller (as 250/190,
`[mavlink] listen`), owned by the MAVLink thread (`sd_ground.mav.thread`). It keeps
working when the main loop, the database, the AI or the onboard computer hang:

- **STOP** (every logged-in user): BRAKE, then LOITER, then LAND if a step is refused;
  armed on the ground it disarms instead. **FLY HOME** (pilot): SMART_RTL, then RTL, then
  LAND. The first command goes out at once, is resent every 200 ms until the heartbeat
  shows the mode, and for 10 s any other mode (a late order) is undone. The onboard
  computer is told over SD-link and lets go; if the ground's MAVLink is cut it sends the
  same steps itself.
- A latch per vehicle that only the pilot's **Resume** lifts (never the chat, never the
  AI); not during a flight home. `GET /api/v1/vehicles/{id}/actions/{action_id}` shows
  the result.
- The **companion watchdog**: no heartbeat from the onboard computer for 3 s in GUIDED
  gives STOP's steps, for 20 s FLY HOME's. The **battery guard** flies home at the
  mission's level. Both stand aside while the pilot's STOP stands.
- **Steering by hand** (`actions/takeover`, `actions/handback`, pilot only): LOITER, and
  the pilot's sticks go as MAVLink `MANUAL_CONTROL` from here, only in LOITER and in the
  air. Without a new stick value for 1 s nothing more is sent and the vehicle holds.
- The GCS heartbeat stops when an event loop has hung for 8 s, so the flight controller's
  own GCS failsafe takes over from a hung ground station.
- `sd-ground action stop|home` goes through the Unix socket (each client in its own
  thread with a 2 s read limit, so a stuck client never blocks a STOP).

Parts that other components fill in (SD-link, the mission flow, video) already have their
routes; until they are built those answer `501 NOT_BUILT`, and `/api/v1/health` reports
`ready = false`.

## Command line

`sd-ground init [--lan]`, `config check`, `db setup`, `serve`, `login-link [--print-code]`,
`pair`, `vehicle add`, `token create --name --role pilot|observer [--expires 24h]`,
`token revoke --name`, `link-token create --vehicle [--write-onboard PATH] [--if-missing]`,
`action stop|home --vehicle` (through the running server). Exit code 2 means the running
server is needed but not there.

## Database

Postgres 18 with PostGIS 3.6 runs locally from the repository's `compose.yaml` and listens
only on `127.0.0.1:5433`. Passwords live in `.env` (created by `just setup-config`).

- `just db-up` starts it and waits until it is healthy; `just db-down` stops it and keeps
  the data; `just db-check` prints the PostGIS version.
- Migrations are in `migrations/` (Alembic). `0001_initial` is hand-written SQL. Migrations
  run as the owner role `sd_owner`; the server uses `sd_app`, which can never delete rows
  and can only add rows to the history tables (`events`, `gate_decisions`,
  `chat_messages`, `brain_decisions`).
- Database tests (`tests/contract/`, markers `contract` and `db`) run in `just test-contract`
  against throwaway `searchdrone_test_*` databases and never touch `searchdrone`.
  The restart test (`db-down` then `db-up`) and the test that briefly gives `sd_app` extra
  rights run in CI, or locally with `SD_TEST_DB_INVASIVE=1`, because they touch the shared
  local database.
- Besides the grants, the schema itself guards human authority: only a named
  `operator:<alias>` can approve a charter or an approval, a charter's proposal (area, home,
  pilot position) and its approval cannot be rewritten, and rejected or superseded
  charters stay final.
- `alembic.ini` and `migrations/` live next to the package, not inside it, so they are not
  in the built wheel yet; `sd-ground db setup` will ship them.
