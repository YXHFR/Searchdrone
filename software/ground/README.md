# sd-ground (`sd_ground`)

The ground station `sd-ground`: API for the panel, database, safety gate, emergency path (STOP / RETURN HOME) and video.

Part of the [Searchdrone](../../README.md) monorepo. Licensed under Apache-2.0.

Run its checks from the repository root: `just ground` (all checks: `just check`).

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
