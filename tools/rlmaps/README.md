# rl-maps: real places for the search policy

`just rl-maps` picks 2 x 2 km places of Swedish forest and open land for the search policy
simulator (`simulation/searchsim`, `world=real`). Each place gets its land cover from
[ESA WorldCover 2021](https://esa-worldcover.org/) (10 m) and every road and track from
[OpenStreetMap](https://www.openstreetmap.org/). The places are written to `build/rl/sites/`,
which git ignores. No map data goes into the repository.

## Steps

```bash
just rl-maps yes=1      # the first time: downloads OpenStreetMap's Sweden file (about 700 MB)
just rl-maps            # 600 places, about 10-20 minutes; ESA's map is read over the internet
just rl-train world=real
just rl-eval world=real
```

| Key | Default | Meaning |
|---|---|---|
| `osm=<file>[,<file>]` | `build/rl/osm/sweden-latest.osm.pbf` | OpenStreetMap extracts to use; any `.osm.pbf` works |
| `n=600` | | How many places |
| `seed=0` | | Another seed gives other places |
| `workers=8` | | Parallel reads of ESA's map |
| `out=build/rl/sites` | | Where the places go |
| `yes=1` | | Allow the Sweden file to be downloaded from Geofabrik |

Relative paths are from the repository's root.

## How a place is chosen

1. A random point in the extract's area, even over the ground.
2. ESA WorldCover over the 2 x 2 km around it. Only that part of ESA's file is read.
3. The place is kept when at least half of it is forest or open land, at most 30 % is
   water and at most 15 % is built-up.
4. One pass over the OpenStreetMap file draws every road and track (`highway=*`, except
   planned ones and ones inside buildings) onto the kept places.
5. A place with no road or track near its middle is dropped too, because a last seen
   point is nearly always on one.

Each place is an `.npz` file with `canopy`, `track`, `water` and `built` on the
simulator's grid. `sites.json` lists the places, the sources and how many were dropped
and why.

## Data and licences

- OpenStreetMap: © OpenStreetMap contributors, [ODbL 1.0](https://www.openstreetmap.org/copyright).
- ESA WorldCover 2021 v200: © ESA WorldCover project / Contains modified Copernicus
  Sentinel data (2021) processed by ESA WorldCover consortium,
  [CC BY 4.0](https://creativecommons.org/licenses/by/4.0/).

Both stay on your computer. The code is Apache-2.0. It runs in its own Python
(`tools/rlmaps/pyproject.toml`) with rasterio (BSD-3-Clause) and pyosmium (BSD-2-Clause).
