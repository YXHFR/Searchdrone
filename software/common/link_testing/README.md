# sd-link-testing (`sd_link_testing`)

Test-only helpers for SD-link. Installed only in the `test` dependency group; never
imported by production packages (import-linter contract).

- `FakeClock`: time moves only when a test calls `advance`.
- `transport_pair`, `MemoryTransport`: an in-memory pipe.
- `LossyTransport`: drops and delays frames on a fake clock (seeded, repeatable).
- `LossyProxy`: the same for real WebSockets on 127.0.0.1; can also cut the link.
- `FakeLink`: a ground and a vehicle session joined in memory.
- `FakeGround`: a real `LinkServer` that answers `Hello`, for onboard tests without the
  ground gate.

Part of the [Searchdrone](../../../README.md) monorepo. Licensed under Apache-2.0.

Run its checks from the repository root: `just test` (all checks: `just check`).
