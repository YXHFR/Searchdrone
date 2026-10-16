# sd-mav (`sd_mav`)

The vehicle link: the Python layer that talks MAVLink to the flight controller
(ArduCopter). Both the onboard computer and the ground station use it, each on its own
channel and with its own identity. It is the only package that imports pymavlink.

```python
from sd_mav import CopterMode, Role, SendOrigin, Vehicle, connect
from sd_obs.clock import MonotonicClock

conn = connect("udpin:127.0.0.1:14563", role=Role.ONBOARD, source_system=1, source_component=191)
async with Vehicle(conn, role=Role.ONBOARD, clock=MonotonicClock()) as vehicle:
    await vehicle.wait_connected(5.0)
    await vehicle.set_mode(CopterMode.GUIDED, origin=SendOrigin.SKILL)
```

**Default deny.** `Vehicle.send` is the only way out. Every message is checked first
against the deny list and then the allowlist, under every helper method. A message
that is not on a row, or whose numbers are outside the mission's limits (height, speed,
the fence radius from home), raises `ForbiddenMavlinkError` and is reported as an
event; nothing is sent. No row writes a flight-controller parameter or touches the
geofence switch. The onboard computer can never send as the ground station (250/190),
never disarms and never uploads a fence; the ground station only sends the STOP and FLY
HOME modes, disarms only on the ground and uploads only the fence. Pilot stick values
(`MANUAL_CONTROL`) pass only with the `manual` origin while the flight controller is in
LOITER.

| Module | Contents |
|---|---|
| `connection` | `connect()` / `MavConnection` (pymavlink, MAVLink 2), identity checks per role, telemetry log |
| `vehicle` | `Vehicle`: telemetry (`FcTelemetry`), 1 Hz heartbeat, stream requests, lost-link detection, command acknowledgements, `SendOrigin` |
| `allowlist` | The rows that may be sent, per profile and origin, and their number checks (`SendEnvelope`) |
| `denylist` | The most dangerous traffic, named one by one and checked first |
| `modes` | `CopterMode`, the STOP and FLY HOME steps, `override_phase` |
| `mission` | Mission and fence encoding, `mission_hash` / `fence_hash`, upload and read-back |
| `failsafe_signals` | Signs that a mode change came from a flight-controller failsafe |
| `fake` | `FakeFlightController` and `FakeVehicle`: an in-memory ArduCopter for tests |

The SITL tests are in `simulation/tests/mav/` (`just test-sitl`).

Part of the [Searchdrone](../../../README.md) monorepo. Licensed under Apache-2.0.

Run its checks from the repository root: `just test` (all checks: `just check`).
