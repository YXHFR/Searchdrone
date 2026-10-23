# The image bank (`tools/bench`)

A fixed, repeatable measuring stick for the person detector: real pictures taken from
the air, with a box drawn around every person, split once and for all into a part to
train on, a part to tune on and a locked part to measure on.

No picture is ever committed, uploaded or cached in CI. The repository holds only the
lists (which picture is in which part), the checksums and the scripts. The pictures
live on your computer under `<data_dir>/datasets/` (the data folder from `just
setup-config`, outside the repository).

## The datasets

| Name | What | Licence | Used for |
|---|---|---|---|
| `heridal` | HERIDAL: 1,684 pictures (4000×3000) of Croatian hillsides with hikers, from the University of Split | CC BY 3.0 | training, tuning, measuring |
| `wisard` | WiSARD v1, visible-light part: 20,504 pictures with labels from wilderness search flights in Washington, from the University of Washington | MIT (see below) | measuring only, for now |
| `simfarm` | Our own simulator pictures (`just sim-farm`) | Apache-2.0 | pre-training only, never measured on |

The register is `datasets.toml`: address, size, SHA-256, licence and credit of each
archive. `tools/tests/test_bench_datasets.py` holds the licence rule: only open licences
(CC0, CC BY, MIT, Apache) may be trained on; a non-commercial set may only ever be
measured on, on your own computer; nothing without a licence is allowed.

**WiSARD's licence.** The project page carries the MIT licence text for "the Software"
and says nothing else about the pictures. Until the lab confirms the data licence in
writing, WiSARD is only measured on, never trained on (`use = ["measure"]`); once it
is confirmed, add `"train"` to its `use` and set `licence_confirmed = true` in
`datasets.toml`. Weights trained on it
would still stay on your computer (`publish_weights = false`).

**Credit.** Anything published from these measurements names HERIDAL (Božić-Štulić,
Marušić and Gotovac, IJCV 2019, CC BY 3.0) and WiSARD (Broyles, Hayner and Leung,
IROS 2022).

## The split (`splits/v1.json`)

Pictures from the same flight look nearly alike, so the split is by **sequence**, never
by picture: all of a sequence's pictures are in the same part. A sequence is the place
for HERIDAL (its own train and test pictures share places, so the place is the safe
group) and the date and place for WiSARD (`200402_Carnation`). Sequences go, largest
first, to whichever part is furthest below its share (70 % train, 10 % val, 20 % test);
`make_splits.py` shows how. `test_split_no_leakage` checks that no sequence is in two
parts.

The test part is **locked**. Changing it means a new file, `splits/v2.json`, and every
number measured again.

`splits/mini_v1.json` is the mini bank: 64 pictures of the HERIDAL test part (about 330
MB) with the SHA-256 of each picture and label. The nightly CI job `bench-mini` reads
just those out of the archive on Zenodo (byte ranges, no full download), checks each
one and keeps nothing.

## The measure (`eval_detector.py`)

Each picture goes to the detector whole, as the drone's camera gives it. Detections are
matched to the person boxes (highest score first, overlap IoU ≥ 0.5). The score
threshold is set so the false alarms come to **0.5 per 12-megapixel picture** (other
sizes count by area). At that threshold the result gives:

- **recall per size class** of the person box's longest side: under 16 px, 16–32 px and
  over 32 px, with a 95 % interval, and per dataset;
- AP50, precision at score 0.25 (an assumed on-board score floor), time per picture
  (median and 95th percentile);
- the score distributions of hits and false alarms (for the simulator's stand-in).

Targets (to be tightened after the first measurement): recall ≥ 0.85 over 32 px and
≥ 0.60 at 16–32 px.

`profiles.py` turns a result into `profiles/det_<model>_v<N>.json`, the file the
simulator's detector stand-in reads. `profiles/det_oracle_v0.json` is the start profile
until a real measurement exists: anything built on it is modelled, not measured.

The result files hold numbers only: no picture, no file name, no box.

## Your steps

Run these from the repository folder in WSL (Ubuntu).

1. **Check the measure works** (5–15 minutes, about 330 MB download):

   ```bash
   just bench-fetch yes=1 mini=1
   just bench-detector model=selfcheck mini=1
   ```

   It should say `null: recall 0.0` and `facit: recall 1.0`.

2. **See what the full download is** (nothing is downloaded):

   ```bash
   just bench-fetch
   ```

3. **Download the bank.** Today that is HERIDAL (about 8 GB; 15–45 minutes depending on
   your line). WiSARD (about 44 GB more, 1–4 hours) is skipped until its archive's
   SHA-256 is pinned in `datasets.toml` (Google Drive stopped serving it before the
   whole file could be checked); a maintainer pins it by downloading it once in full
   and recording `sha256sum` of the file:

   ```bash
   just bench-fetch yes=1
   ```

   If it stops (a dropped line, a closed window), run the same command again: it picks
   up where it was. If it says the checksum does not match, the file is set aside as
   `*.bad` and nothing is unpacked; run it again, and if it happens twice, say so.
   Google Drive (WiSARD) sometimes refuses with "quota exceeded" for a day; then fetch
   HERIDAL alone with `sets=heridal` and try WiSARD the next day.

4. **Optional: the sim farm's pictures** for pre-training (after `just sim-farm`):

   ```bash
   just bench-fetch yes=1 sets=simfarm
   ```

5. **Measure a detector** (training it: `tools/train/README.md`):

   ```bash
   just bench-detector model=person_yolox_s part=val
   ```

   `model=` is `selfcheck`, `null`, `facit`, or a model in
   `<data_dir>/models/detector/` by name: `coco_yolox_s` (from `just detector-fetch`)
   or `person_<model>` (from `just export-detector`). The model runs over 640 px tiles of
   each whole picture, as on the drone. The result is in
   `build/bench/det_<name>_<part>.json`.

Keys of `just bench-fetch`: `yes=1` (required to download), `sets=heridal,wisard` (which
sets; default every set whose SHA-256 is pinned), `mini=1` (the mini bank), `keep=1`
(keep the archive after unpacking), `data=<folder>` (another data folder).

Keys of `just bench-detector`: `model=`, `part=test|val`, `sets=`, `mini=1`, `limit=N`
(the first N pictures, for a quick look), `data=`, `out=`, `compare=<earlier result>`
(fails if recall over 32 px drops more than 2 points).

## For maintainers

`python -m bench.make_splits list <set>` lists an archive's members straight from the
server; `split` draws a split from such listings and `mini` draws the mini bank. Both
refuse to overwrite a published file. Archive checksums in `datasets.toml` are filled
in by downloading the archive once and running `sha256sum` on it.
