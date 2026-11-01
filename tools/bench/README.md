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
| `kios` | KIOS Small Object Aerial Person Detection: 3,136 pictures (1920×1080 to 3840×2160) with 61,708 people from drone flights in Cyprus, from the University of Cyprus | CC BY 4.0 | training, tuning, measuring |
| `wisard` | WiSARD v1, visible-light part: 20,504 pictures with labels from wilderness search flights in Washington, from the University of Washington | MIT (see below) | measuring only, for now |
| `simfarm` | Our own simulator pictures (`just sim-farm`) | Apache-2.0 | pre-training only, never measured on |

The register is `datasets.toml`: address, size, SHA-256, licence and credit of each
archive. `tools/tests/test_bench_datasets.py` holds the licence rule: only open licences
(CC0, CC BY, MIT, Apache) may be trained on; a non-commercial set may only ever be
measured on, on your own computer.

**Measure-only sets (GPL or no data licence).** A set under the GPL, or with no data
licence at all (`licence = "NONE"`), gives no right to build on it. Since the owner's
decision of 2026-11-01 such a set may still be measured on, on your own computer, and
nothing more. Its row must say `use = ["measure"]`, `local_only = true`,
`publish_weights = false` and `publish_results = false`, or the register is refused.
`just train-detector` refuses it in `sets=` (it is not quietly left out), and
`profiles.py` refuses a result measured on it, because a profile is committed here.
Its numbers stay on your computer: they go in no README, profile, issue or paper. Any
other licence (one not listed above) is not allowed at all.

None is in the register yet. The two candidates could not be pinned (checked
2026-11-01):

- **LADD** (Lacmus Drone Dataset, lacmus-foundation, GPL-3.0 per
  [ladd-utils](https://github.com/lacmus-foundation/ladd-utils)): the download link in
  that README (Mail.ru Cloud) answers "404", and the foundation's DVC store
  ([ladd-and-weights](https://github.com/lacmus-foundation/ladd-and-weights)) needs
  credentials from the Lacmus team. Copies elsewhere (Dataset Ninja, Kaggle, Roboflow)
  are not the primary source and are not used.
- **NOMAD** (Notre Dame, WACV 2024, [ArtRuss/NOMAD](https://github.com/ArtRuss/NOMAD),
  no licence): a public Google Drive folder with the pictures as ten zip files of ten
  actors each (about 38 GB each) and the labels as loose files and one
  `annotations.json`. Google Drive answered "quota exceeded" for every file, so no size
  or SHA-256 could be read; the register also takes one picture archive per set today.

**WiSARD's licence.** The project page carries the MIT licence text for "the Software"
and says nothing else about the pictures. Until the lab confirms the data licence in
writing, WiSARD is only measured on, never trained on (`use = ["measure"]`); once it
is confirmed, add `"train"` to its `use` and set `licence_confirmed = true` in
`datasets.toml`. Weights trained on it
would still stay on your computer (`publish_weights = false`).

**KIOS.** Zenodo record [7740081](https://zenodo.org/records/7740081), CC BY 4.0 on the
record. Two archives: `Images.zip` (3.7 GB) and `Annotations.zip` (4 MB); both are
pinned and checked. The YOLO labels are read; the VOC files are not, because some of
them name another picture. The archive's own Train/Valid/Test folders are ignored: they
hold frames of the same videos, so a model would be measured on near-copies of what it
learned.

**AOS (JKU Linz) is not in the bank.** The Airborne Optical Sectioning data
([Zenodo 3894774](https://zenodo.org/records/3894774), CC BY 4.0) has person boxes only
for 512×512 thermal frames and for integral images (many frames combined into one).
Its colour single frames (`RAW_lowresRGB.zip`, `RAW_highresRGB.zip`) carry no person
boxes at all. The RGB integral images in [Zenodo 6382373](https://zenodo.org/records/6382373)
have boxes, but they are combined pictures, not what the drone's camera takes. The bank
is colour single pictures only, so AOS stays out until colour boxes exist; none are made
up here.

**Credit.** Anything published from these measurements names HERIDAL (Božić-Štulić,
Marušić and Gotovac, IJCV 2019, CC BY 3.0), KIOS (Makrigiorgis, Kyrkou and Kolios,
"Small Object Aerial Person Detection Dataset", KIOS CoE, University of Cyprus, Zenodo
2023, doi:10.5281/zenodo.7740081, CC BY 4.0) and WiSARD (Broyles, Hayner and Leung,
IROS 2022).

## The split (`splits/v2.json`)

Pictures from the same flight look nearly alike, so the split is by **sequence**, never
by picture: all of a sequence's pictures are in the same part. A sequence is the place
for HERIDAL (its own train and test pictures share places, so the place is the safe
group), the date and place for WiSARD (`200402_Carnation`) and the site for KIOS.
Sequences go, largest first, to whichever part is furthest below its share (70 % train,
10 % val, 20 % test); `make_splits.py` shows how. `test_split_no_leakage` checks that no
sequence is in two parts.

KIOS's file names carry the recording (`<n>_<recording>_<frame>.jpg`), not the site. The
sites were read off the pictures (`KIOS_SITES` in `layouts.py`): the University of
Cyprus campus and a civil-defence exercise ground (one sequence, because the recording
`people_0042` flies over both), the coast at Pegia, a shore (`ppl`, `ppl_2`), a yard
and a meadow. That gives KIOS 2,131 pictures to train on (campus and exercise), 608 to
tune on (Pegia) and 397 to measure on (shore, yard, meadow): whole sites, never single
pictures. A recording that is not in the list is refused, so a changed archive cannot
slip a new site into a part unseen.

The test part is **locked**. `v2.json` is `v1.json` with KIOS added: HERIDAL's and
WiSARD's parts are copied from v1 unchanged (`test_v2_keeps_every_v1_dataset_as_it_was`),
so a HERIDAL number measured on v1 still compares with one measured on v2. Changing a
locked part means a new file and every number measured again.

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
simulator's detector stand-in reads. It takes a result measured on one dataset only
(`sets=heridal`): a result pooled over several is refused, because its recall would be
mostly that of the set with the most people. `profiles/det_oracle_v0.json` is the start profile
until a real measurement exists: anything built on it is modelled, not measured.
Run `just fmt` before committing a new profile or split file: the repository formats
JSON with Biome, and `just lint` checks it.

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
   your line) and KIOS (about 3.7 GB; 5–20 minutes). WiSARD (about 44 GB more, 1–4 hours) is skipped until its archive's
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
   HERIDAL and KIOS alone with `sets=heridal,kios` and try WiSARD the next day.

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
   `build/bench/det_<name>_<part>.json`. It prints the recall over all the pictures and
   per dataset (`heridal`, `kios`, ...); `sets=heridal` measures one set alone.

Keys of `just bench-fetch`: `yes=1` (required to download), `sets=heridal,kios` (which
sets; default every set whose SHA-256 is pinned), `mini=1` (the mini bank), `keep=1`
(keep the archive after unpacking), `data=<folder>` (another data folder).

Keys of `just bench-detector`: `model=`, `part=test|val`, `sets=`, `mini=1`, `limit=N`
(the first N pictures, for a quick look), `data=`, `out=`, `compare=<earlier result>`
(fails if recall over 32 px drops more than 2 points for the weakest dataset or for any
one dataset, each at its own threshold, or if a dataset of the earlier result was not
measured; an earlier result without per-dataset rows is compared pooled against pooled).

## For maintainers

`python -m bench.make_splits list <set>` lists an archive's members straight from the
server (`list <set> labels=1`: its labels archive); `split` draws a split from such
listings, `extend` makes a new version that keeps an earlier one's datasets as they
were and draws only the new ones, and `mini` draws the mini bank. All refuse to
overwrite a published file. `v2.json` was made with:

```bash
m="env PYTHONPATH=tools uv run --no-sync python -m bench.make_splits"
$m list kios > /tmp/kios.txt
$m list kios labels=1 > /tmp/kios_labels.txt
$m extend version=v2 base=v1 kios=/tmp/kios.txt kios_labels=/tmp/kios_labels.txt
```

Archive checksums in `datasets.toml` are filled in by downloading the archive once and running `sha256sum` on it.
