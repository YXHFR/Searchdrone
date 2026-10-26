# sd-searchsim (`sd_searchsim`)

A fast 2D search simulator and a learned policy that picks **which sector to search
next**. The policy is advisory only: no part of the flight path imports this package (an
import contract checks that), it never sends a command, and every suggestion it could
make later still has to pass the charter, the geofence and the safety gate like any other.

Part of the [Searchdrone](../../README.md) monorepo. Licensed under Apache-2.0. Needs only
numpy.

## Commands

```bash
just rl-train                 # full training on this computer's CPU cores
just rl-train size=ci         # a small run, about a minute
just rl-eval                  # compare on 1000 held-out scenarios -> build/rl/eval.json
just rl-eval size=ci          # 300 scenarios
```

Keys: `size=full|ci`, `seed=0..9` (train), `workers=<int>` (default: every core),
`out=<file>`, `policy=<file>` (eval), `world=synthetic|real`, `sites=<folder>`. The policy is written to `build/rl/policy.json`,
which git ignores; trained weights never go into the repository. Without a policy file
`rl-eval` compares only the baselines.

## The simulator

- **Map:** 2 x 2 km in 25 m cells, split into 8 x 8 sectors of 250 m. Random tracks and
  a smooth canopy field. Canopy hides people: the chance a sweep sees a person in a
  cell is `p_det * (1 - canopy)`.
- **Prior (the belief):** log-normal distance rings around the last seen point, weighted
  up near tracks. The truth is drawn with a different ring size, track pull and
  detector strength, so the map is never exactly right.
- **The missing person:** a mixture of lost-person strategies (stay put, random walk,
  route travel along tracks, direction travel), after Hashimoto et al. 2022. People
  stop moving after a random time. The belief spreads with a Gaussian over time.
- **Flying:** one decision = fly to a sector and survey it with a lawnmower pattern
  (about 4 minutes). Bayes update after a sweep with no find. A sector is *feasible*
  only when it is permitted (some scenarios have a no-fly band along one edge) and the
  transit, the survey and the flight home fit in the battery above a 15 % reserve. When
  nothing fits, the drone flies home and changes battery. 180 minutes in all.
- **Score:** given the true path the outcome of each sweep is a coin flip with the true
  detection chance, so the simulator computes the exact expectation over the coins:
  `p_find`, `expected_minutes` (time to find, counted as 180 when not found) and
  `score = 1 - expected_minutes / 180`.

## Real places (`world=real`)

`world=real` swaps the made-up terrain and the strategy mixture for real places and real
statistics. Make the places first with `just rl-maps` (see `tools/rlmaps/README.md`):

- **Places:** 2 x 2 km of Swedish forest and open land. ESA WorldCover gives the canopy
  (trees 0.75, bushes 0.35, built-up 0.3, wetland 0.05; model defaults) and the water,
  and OpenStreetMap gives the roads and tracks. Each scenario turns or mirrors its place
  at random (8 ways).
- **Split by place:** 70 % of the places train, 10 % validate and 20 % are the final
  comparison. `rl-eval world=real` is therefore on places training never saw.
- **Who is missing:** a person with dementia (35 %), a despondent person (30 %), a hiker
  (20 %) or a berry or mushroom picker (15 %). The shares are defaults.
- **Where they are found:** from the Norwegian search dog statistics (Lie & Skår, ICAR
  2016), per category:
  - the distance from the last seen point, as a log-normal fitted to the quartiles;
  - the share found on a road or track, and within 100 m of one.

  The last seen point is on a road or track near the middle. The person stays where
  they are found, never in water. The belief uses the published numbers. The truth's
  distance is off by about 1.35x either way.
- **What is left out:** children (their numbers are only in Koester's book) and movement
  during the search.

`rl-eval world=real` also prints the mean score per category.

## The baselines and the policy

| Name | Rule |
|---|---|
| `lawnmower` | Row by row over the whole grid from the corner nearest home |
| `spiral` | Square spiral outwards from the last seen point |
| `greedy_poa` | The sector with the most probability |
| `greedy_psr` | The sector with the most expected detection per minute of transit and survey |
| `policy` | A learned scorer over each sector's features (below) |

The policy scores every sector from ten features (expected detection, the same per
minute, probability, the neighbours' expected detection, mean detection chance,
transit and home distance, times surveyed, battery and time left) with a linear part
and one hidden layer of 16 units, about 200 weights. It picks the feasible sector with
the highest score. It starts equal to `greedy_psr` and is trained with evolution
strategies (antithetic pairs, centred ranks, Adam), every candidate on the same
scenarios. The best checkpoint on a separate validation set is kept.

Why so small and why not PPO on a GPU: the time goes into the simulator, not the
network, and a scorer per sector works on any number of sectors. ES needs no value
function and spreads over CPU cores. The owner's GPU would sit idle, so training uses
the CPU.

## Reading the result

`build/rl/eval.json` has, per chooser, the mean and a 95 % interval of each measure,
and for the policy the paired difference to each baseline on the same scenarios, with
wins, ties and losses. Training, validation and evaluation use separate seed ranges.

First results (a CI-size run and an 80-iteration run, each compared on 300 held-out
scenarios): the policy clearly beats lawnmower (+0.17 score), spiral (+0.05) and
`greedy_poa` (+0.03), but it is level with `greedy_psr` (difference about 0.00, 95 %
interval about ±0.01). Search theory points the same way: when the travel cost is
counted, the greedy rule is hard to beat. The policy has not yet earned a place ahead of
`greedy_psr`. The full run on more scenarios is how to find out whether it ever does.
A validation gain on its own does not show it; only the paired comparison does.

What the numbers do not show: the map, the motion model and the detector are simple
stand-ins. A policy that wins here has only earned a place in the SITL tests, not in a
real search.
