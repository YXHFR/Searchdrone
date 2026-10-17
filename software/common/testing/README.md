# sd-testing (`sd_testing`)

Test-only helpers shared by every package. Installed only in the `test` dependency
group; never imported by production packages (import-linter contract).

- `FakeClock`: simulated time for the injectable clock in `sd_obs.clock`. Time moves
  only when a test calls `advance`, `run_for` or `run_until`.

Part of the [Searchdrone](../../../README.md) monorepo. Licensed under Apache-2.0.

Run its checks from the repository root: `just test` (all checks: `just check`).
