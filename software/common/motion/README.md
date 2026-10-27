# sd-motion (`sd_motion`)

Where a lost person walks while we search. The probability map gets two layers: mass that
has *stopped*, and mass *on the move* in eight headings. Each 15 s step:

1. some walkers stop, as the category's mobility curve says (faster in shrub and wetland
   for dementia);
2. some turn 45 degrees or pick any heading;
3. walkers on a track turn along it, walkers next to one turn towards it;
4. walkers step one cell at the terrain's speed. Water is a wall; off the grid is the rest
   of the world.

- `Terrain.from_classes` (ESA WorldCover and a track mask) or `Terrain.from_cover` (forest,
  track and water masks) builds the ground once per map.
- `start_state` walks the model out from the last seen point and blends it 50/50 with the
  distance rings; `advance` runs it forward; `negative` applies a sweep that found nothing.
- `for_category` reads the parameters per kind of missing person from `motion.toml`, with
  a source or a "guess" on each row.

numpy only and no project imports (`.importlinter`), so the flight-side map (`sd_search`)
and the policy simulator (`sd_searchsim`) share it.

Part of the [Searchdrone](../../../README.md) monorepo. Licensed under Apache-2.0.

Run its checks from the repository root: `just test` (all checks: `just check`).
