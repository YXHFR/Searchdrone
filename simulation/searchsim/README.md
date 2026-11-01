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
`out=<file>`, `policy=<file>` (eval), `world=synthetic|real`, `sites=<folder>`,
`belief_motion=diffuse|lpb` (real only, see below), `miss_correlation=0..1` (default 0.8,
see below). The policy is written to `build/rl/policy.json`,
which git ignores; trained weights never go into the repository. Without a policy file
`rl-eval` compares only the baselines.

## The simulator

- **Map:** 2 x 2 km in 25 m cells, split into 8 x 8 sectors of 250 m. Random tracks and
  a smooth canopy field. Canopy hides people: the chance the first sweep sees a person
  in a cell is `p_det * (1 - canopy)`. A repeat sweep sees less (next point).
- **Repeat sweeps (`miss_correlation`):** someone hidden under trees on one pass is
  likely still hidden on the next pass over the same spot. So the first sweep's chance is
  split into two parts: is the person *visible* from the air in that cell at all
  (`1 - miss_correlation * canopy`, the same on every pass) and, if so, does this sweep
  see them (the rest, fresh each pass). After `n` empty sweeps with the person in the
  cell, the next sweep sees them with `per_pass * P(visible | n misses)`, which falls
  towards zero under trees and stays the same on open ground. The default is 0.8 (a
  guess, not a measured value); 0 makes every sweep an independent chance, as before.
  The truth counts the empty sweeps per cell while the person was in it; the belief
  counts the sweeps of each sector, as if the person had stayed put, and uses that for
  its Bayes update and for every sector's expected detection, so the planners see the
  smaller gain too. Both worlds work the same way.
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

Why a visible/hidden split and not a fixed factor per repeat (`pod * rho^(n-1)`): the
split keeps the score an exact expectation. Each cell's "visible" coin is drawn once and
the cells' coins are independent, so the simulator only has to count the misses per cell.
The first sweep keeps its old chance, open ground keeps independent passes (there only the
camera misses), and a person who walks into a new cell gets a fresh chance there. A fixed
factor would need a made-up `rho` with no meaning of its own and would cut the chance on
open ground as much as under trees.

**Numbers from before `miss_correlation` (when every sweep was an independent chance)
are not comparable with numbers made with the default.** To make them again, pass
`miss_correlation=0` to `rl-train` and `rl-eval`; it gives the old results bit for bit.

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

  The last seen point is on a road or track near the middle. The belief uses the
  published numbers. The truth's distance is off by about 1.35x either way.
- **Moving during the search:** the search starts 1-6 h after the person was last seen
  (a guess). The person starts somewhere on the truth's map and, if still on the move
  (the category's mobility curve), keeps walking by the motion model (`sd_motion`): on
  and towards tracks, turning back at water, until they stop or leave the map. Their
  speed and walking time are off the category's by about 1.35x either way.
- **The belief's movement** (`belief_motion`): `diffuse` spreads the map with a Gaussian
  as in the made-up world; `lpb` runs the same motion model as the truth (with the
  category's numbers), blended 50/50 with the rings at the start. Cells with a canopy of
  0.5 or more walk like forest; the sites have no shrub or wetland classes, so the
  dementia "stuck in the bushes" rule does not apply here.
- **What is left out:** children (their numbers are only in Koester's book).

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

Those first results were made before `miss_correlation`, with every sweep an independent
chance, and are not comparable with runs made now. What the change does, on the
baselines only (`just rl-eval size=ci`, 300 made-up scenarios, mean score with 95 %
interval):

| Name | Independent sweeps (`miss_correlation=0`) | Default (0.8) |
|---|---|---|
| `greedy_psr` | 0.256 (0.222-0.289) | 0.241 (0.211-0.270) |
| `greedy_poa` | 0.228 (0.196-0.259) | 0.184 (0.155-0.214) |
| `spiral` | 0.205 (0.183-0.228) | 0.205 (0.183-0.228) |
| `lawnmower` | 0.089 (0.074-0.105) | 0.089 (0.074-0.105) |

`spiral` and `lawnmower` do not change: in 180 minutes they almost never sweep a sector
twice. `greedy_psr` now spreads its about 30 surveys over 21 sectors on average instead
of 15. `greedy_poa` loses the most: it looks only at the probability, which a sweep under
trees now lowers less, so it keeps going back to the same wooded sector. The policy has to
be trained again under the new model before it can be compared.

What the numbers do not show: the map, the motion model and the detector are simple
stand-ins. A policy that wins here has only earned a place in the SITL tests, not in a
real search.
