# sd-link (`sd_link`)

SD-link: the control channel between the ground station and the onboard computer.
One WebSocket per vehicle (`ws://127.0.0.1:7600/sdlink/v1/control`), JSON envelopes
from `sd_schemas.link`. The onboard computer dials the ground.

What the package gives both sides:

- **Envelopes** (`sd_link.envelope`): build, encode, decode. Each side may only send its
  own kinds (orders, STOP/HOME notices, manual takeover and stick values only from the
  ground). Per-kind ttl rules: `SkillCommand` needs a ttl, `ClearanceConfirmation` at
  most 5 s, `ManualControl` at most 1 s. Messages are at most 64 KiB.
- **Reliability** (`sd_link.reliability`): reliable messages are resent after 1 s
  without an `Ack`, at most 3 times, then reported as expired (`LINK_NO_ACK`). A resend
  keeps `msg_id`, gets a new `seq` and a shorter ttl. Duplicates get the same `Ack` and
  no new effect. Statuses never move backwards (`advance_stage`).
- **Sessions** (`sd_link.session.LinkSession`): heartbeat at 1 Hz, link state
  `up` → `degraded` (3 s silent) → `lost` (10 s, closed with 1001). Liveness and expiry
  run in their own task and never wait for a write; a write blocked for 2 s aborts the
  connection. When a reliable message expires, the connection is aborted at once so
  that the stale frame can never arrive later (the socket is reset, so the kernel drops
  its send queue too). As a second guard the receiver answers orders, clearances,
  releases, charters and takeovers that arrive right after a 3 s dropout with
  `Ack(expired, LINK_STALE)` (re-issue later with a new id); STOP/HOME notices always
  go through. `LINK_NO_ACK`/`LINK_RECONNECT` mean the outcome is unknown: resynchronise
  from the vehicle's state instead of assuming the order did not run. Nothing is resent
  on a new connection;
  unacknowledged messages are handed back when a session ends. Delivery order holds
  when the application waits for an Ack before the next order.
- **Stick values** (`ManualControl`) go through `LinkSession.offer`: best-effort, only
  while the link is up, never queued or resent. The receiver keeps only the newest
  value and drops values for 1 s at the start of a stream and after a 0.5 s stall, so
  an old burst is never replayed.
- **Server** (`sd_link.server.LinkServer`, ground) and **client**
  (`sd_link.client.LinkClient`, onboard, reconnects with backoff).
- **Tokens** (`sd_link.tokens`): 32 random bytes; the ground stores only the SHA-256 and
  compares in constant time.

Close codes (`sd_link.close_codes`): 4401 wrong token, 4403 unknown vehicle or a browser
`Origin`, 4400 bad `Hello`, 4409 replaced by a newer connection, 1001 link lost, 1008
too many invalid frames or events not consumed.

The `websockets` library logs through `sd_link.ws`, which is pinned to INFO and redacts
Bearer tokens.

All timing goes through an injected `Clock`; tests use `sd_link_testing.FakeClock`.

Part of the [Searchdrone](../../../README.md) monorepo. Licensed under Apache-2.0.

Run its checks from the repository root: `just test` (all checks: `just check`).
