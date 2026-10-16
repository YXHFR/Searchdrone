# sd-geo (`sd_geo`)

Geometry helpers in a local east/north plane (metres):

- `lawnmower`: the back-and-forth search pattern over a polygon (also concave ones).
- `coverage_fraction`: how much of a polygon a flown track has covered.

Projection between WGS84 and the plane, distances and containment come from the safety
gate (`sd_safety.targets`), so the gate and the executor measure with the same geometry.

Part of the [Searchdrone](../../../README.md) monorepo. Licensed under Apache-2.0.

Run its checks from the repository root: `just test` (all checks: `just check`).
