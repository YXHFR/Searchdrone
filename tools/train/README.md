# Training the person detector (`tools/train`)

This turns two published detectors into person detectors for pictures taken from the
air. Training runs on your computer's graphics card. You choose the better of the two
with numbers measured on the image bank (`tools/bench`). Nothing here goes into git or
CI: the pictures, the weights and the trained models stay in your data folder
(`<data_dir>`, set by `just setup-config`).

The two candidates, both Apache-2.0 and both started from their authors' COCO weights:

| Name | What | Size |
|---|---|---|
| `yolox_s` | YOLOX-s, Megvii (2021) | 9 M parameters |
| `damoyolo_s` | DAMO-YOLO-S, Alibaba (2022) | 16 M parameters |

Each is trained on the same 640 px tiles the drone looks at, made from the image bank's
`train` part. The `val` part picks the best epoch and the better model. The locked
`test` part is measured once, on the chosen model only, and is never trained on.

## What you need

- Windows with WSL2 (Ubuntu) and the repository set up (`just bootstrap`, `just setup`,
  `just setup-config`).
- An NVIDIA graphics card with at least 8 GB of memory, for example an RTX 3070, and the
  ordinary NVIDIA driver for Windows. Do not install a driver inside WSL. To check it,
  run `nvidia-smi` in WSL; it should list the card.
- About 30 GB free on the disk that holds `<data_dir>`.
- 16 GB of RAM is enough.

No graphics card? Put `SD_TORCH=cpu` before every `just train-detector` and
`just export-detector`. It works, but a training run then takes days instead of hours.

## Your steps

Run these in WSL from the repository folder. The times are rough, for an RTX 3070 and a
home internet line.

1. **Download the image bank** (15–45 minutes; see `tools/bench/README.md`):

   ```bash
   just bench-fetch yes=1
   ```

2. **Download the starting weights** (2–5 minutes, about 250 MB). Without `yes=1` it
   only shows what it would download.

   ```bash
   just detector-fetch yes=1
   ```

   This also fetches `coco_yolox_s`, YOLOX-s as published, not trained for the air. It
   is a baseline to compare with.

3. **Train YOLOX-s** (about 1.5–3 hours). The first run also installs PyTorch, about
   3 GB and 5–10 minutes, and cuts the pictures into tiles, about 10–20 minutes.

   ```bash
   just train-detector model=yolox_s
   ```

   It prints the epoch, the loss and the minutes left. You can stop it at any time with
   Ctrl+C, or by closing the window. Run the same command again and it carries on from
   the last save. It saves every 10 minutes and after every epoch.

4. **Make its ONNX file** (1 minute):

   ```bash
   just export-detector model=yolox_s
   ```

   This writes `person_yolox_s.onnx` and `person_yolox_s.json` to
   `<data_dir>/models/detector/`. Before writing, it checks that the ONNX file gives the
   same answers as the trained network.

5. **Train and export DAMO-YOLO-S** (about 2–4 hours):

   ```bash
   just train-detector model=damoyolo_s
   just export-detector model=damoyolo_s
   ```

6. **Measure all three on the val part** (10–20 minutes each; the measuring runs on
   the processor):

   ```bash
   just bench-detector model=person_yolox_s part=val
   just bench-detector model=person_damoyolo_s part=val
   just bench-detector model=coco_yolox_s part=val
   ```

7. **Pick one** (seconds):

   ```bash
   just pick-detector
   ```

   The rule:
   - The model with the highest recall on people over 32 px wins.
   - Models within 2 percentage points of the best count as equal. Among them, the one
     with the higher recall at 16–32 px wins.
   - If they are still within 2 points of each other, the faster one wins.

   It prints the chosen model and the line `SD_DETECTOR_MODEL=<path to its .json>`.

8. **Measure the chosen model once on the test part** (20–40 minutes), and keep the
   result:

   ```bash
   just bench-detector model=person_yolox_s
   ```

   Use the name `pick-detector` chose. This is the number to report. The result is
   written to `build/bench/det_<model>_test.json`. To make the simulator's profile from
   it, see `tools/bench/README.md`.

9. **Use it.** The on-board computer (`sd_onboard`) loads the model when
   `SD_DETECTOR_MODEL` in its environment points at the `.json` file that
   `pick-detector` printed. It checks the manifest and the ONNX file's SHA-256 before
   using it. If the file is missing or changed, only the detector is turned off, and the
   ground station gets the reason (`executor.perception_off`) so you can see that person
   detection is off. Flying goes on. When the variable is not set, no model is
   loaded. The simulator keeps its stand-in detector.

## If something goes wrong

| What you see | What to do |
|---|---|
| "Grafikkortets minne räckte inte" | Run it again with `batch=4 fresh=1`. |
| "COCO-vikterna saknas" | Run step 2. |
| "har en träning med andra inställningar" | You changed a setting (`batch`, `epochs`, …) of a run that was already started. Run it with the old settings, or start over with `fresh=1`. |
| "Inga bilder att träna på" | Run step 1. |
| "ONNX-filen räknar inte som PyTorch" | Nothing was written. Run step 4 again. If it happens twice, report it. |
| "Tränar på processorn (cpu)" at the start, or "Inget grafikkort syns" | PyTorch does not see the card. Check `nvidia-smi` in WSL and update the Windows driver; stop with Ctrl+C before it starts on the processor. |

Useful keys for `just train-detector`:

| Key | What it does |
|---|---|
| `epochs=20` | Passes over the training tiles. |
| `batch=8` | Tiles per step; 8 fits in 8 GB. |
| `simfarm=1` | First pre-trains for `simfarm_epochs=3` epochs on the simulator's pictures (`just sim-farm`, then `just bench-fetch yes=1 sets=simfarm`). Those pictures are never measured on. |
| `limit=20` | Only the first 20 pictures, to try it out quickly. |
| `device=cpu` | Train on the processor. |
| `workers=4` | How many picture loaders run at once. |

## Which data is used, and what may be published

- **HERIDAL** (CC BY 3.0) is trained on. Weights trained on it may be published if
  HERIDAL is credited.
- **WiSARD** is only measured on for now, never trained on, because its data licence is
  not confirmed. It is not fetched at all until its archive's SHA-256 is pinned in
  `tools/bench/datasets.toml`.
- **COCO** weights come from YOLOX and DAMO-YOLO (Apache-2.0).
- **Non-commercial sets** may only ever be measured on, on your computer.

The manifest (`.json`) next to each model lists what it was trained on, under which
licence and with which credit. `publish` is `true` only when every training set allows
publishing the weights. `software/detector/MODEL_CARD.md` has the details.

## For developers

- `families.py`: builds each network and its 1-class person head. The head starts from
  COCO's person channel.
- `tiles.py`: cuts the pictures into tiles.
- `train_detector.py`: the training loop, with SGD, warmup, a cosine schedule, mixed
  precision on CUDA, EMA and resume.
- `export_onnx.py`: writes the ONNX file and its manifest.
- `fetch_weights.py`: downloads the starting weights, checking size and SHA-256
  (`models.toml`).
- `vendor/`: the authors' model code. `vendor/NOTICE.md` lists what was changed.

The folder has its own Python environment with PyTorch (`pyproject.toml`, `uv.lock`), so
the rest of the repository never installs it. The tests run on the CPU with a small
synthetic bank. `just check` does not run them, because the first run installs PyTorch
(about 1 GB for the CPU build); run them by hand after changing anything here. CI runs
them in the `train` job:

```bash
just test-train
```

The on-board side is `sd_detect.onnx` (`software/detector`). It needs only ONNX Runtime,
never PyTorch.
