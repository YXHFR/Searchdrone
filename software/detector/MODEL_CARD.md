# Model card: the person detector

What the trained person detector is, what it was trained on, how it is chosen, and what
it must not be used for. The numbers themselves are not in this file: they come from
`just bench-detector` on the owner's computer, and the chosen model's result is kept as
a detector profile (`tools/bench/profiles/det_<model>_v<N>.json`).

## What it does

It finds people in still pictures taken from the air. Each person gets a box and a
score. The drone's on-board computer runs the detector over 640 px tiles of each 12 MP
still (`sd_detect.TiledDetector`) and sends the boxes to the ground station as hints.
**A detection is only a hint.** A person looks at every hint and confirms it. A
detection never steers the drone, and it never reaches the safety gate.

It does not recognise faces or identify anyone. It does not use thermal pictures. It has
one class: person.

## The candidates

| Name | Network | From | Licence |
|---|---|---|---|
| `person_yolox_s` | YOLOX-s | Megvii's COCO weights (YOLOX 0.1.1rc0) | Apache-2.0 |
| `person_damoyolo_s` | DAMO-YOLO-S (TinyNAS L25) | Alibaba's COCO weights | Apache-2.0 |
| `coco_yolox_s` | YOLOX-s, not retrained | Megvii's published ONNX file, COCO class "person" | Apache-2.0 |

`coco_yolox_s` is the baseline. A trained model has to beat it to be worth using.

## Training

- **Data.** The `train` part of the image bank's locked split (`tools/bench/splits/v2.json`),
  from every set whose licence allows training and is confirmed (today: HERIDAL,
  CC BY 3.0, and KIOS, CC BY 4.0), mixed. The split is by site: no site has pictures in
  two parts.
  - Each picture is cut into 640 px tiles with 15 % overlap.
  - A person is kept in a tile when at least half of their box is in it.
  - Every tile with a person is kept, plus a quarter of the empty tiles.
  - Optional first stage: the simulator's own pictures (`simfarm=1`), for pre-training
    only.
- **Start.** The authors' COCO weights. The one-class head starts from COCO's person
  channel. Or (`init=`) an earlier run's best weights, for example the HERIDAL-only
  model; the manifest then credits every set those weights were trained on too.
- **Recipe.**
  - Optimiser: SGD with momentum 0.9 (Nesterov) and weight decay 5e-4.
  - Learning rate: 0.01 per 64 tiles, with a warm-up (a tenth of the run, at most three epochs) and a cosine decay.
  - Mixed precision on the graphics card, and an EMA of the weights.
  - Flips, 90° turns and brightness changes (seen from above, every way up is a way a
    person lies).
  - Defaults: 20 epochs, batch 8, which fits in 8 GB.
- **Kept epoch.** The highest mean, over the datasets, of the AP50 on each dataset's
  `val` tiles (each set's AP50 is printed and kept in the manifest).

## Choosing one (`just pick-detector`)

All candidates are measured with `just bench-detector ... part=val` on whole pictures,
the same way the drone sees them. The threshold is set so that false alarms stay at
0.5 per 12 MP picture. Every recall below is taken per dataset, and a model's recall is
its **lowest** dataset's recall, never the recall pooled over all people: KIOS's val
site has far more people than HERIDAL's val part, so a pooled number would be almost
all KIOS and could hide a model getting worse on HERIDAL's wilderness. Then, in order:

1. The highest recall on people over 32 px wins.
2. Every model within 2 percentage points of that recall is as good on it. Among those,
   the highest recall on people of 16–32 px wins, again within 2 points.
3. Among the models still level, the fastest (median time per picture) wins.

The choice is made on the `val` part only. The `test` part is measured once, on the
chosen model, and the numbers reported are **per dataset** (`just bench-detector
sets=heridal`, `sets=kios`), HERIDAL first; a pooled test number is not reported. The
targets, to be revisited after the first measurement, are recall ≥ 0.85 over 32 px and
≥ 0.60 at 16–32 px, and each dataset must meet them on its own.

## Known limits

- **Few places and seasons.** HERIDAL is Croatian hillsides in summer; KIOS adds five
  Cypriot sites (campus and an exercise ground, coast, shore, yard, meadow), mostly
  open ground. Snow and dense forest are not covered: the one open dense-forest set we
  know (JKU's Airborne Optical Sectioning data) has person boxes only for thermal and
  for combined (integral) pictures, not for single colour pictures. WiSARD (Washington wilderness) is
  measured on only once its archive's checksum is pinned, and trained on only once its
  data licence is confirmed.
- **Small people are hard.** People under 16 px are reported but not targeted. Expect
  low recall there.
- **Measured only on the CPU so far.** The latency in a result is for the computer that
  measured it, not for the drone's on-board computer.
- **Persons only.** Every other class, such as a backpack, a tent or clothing colour,
  comes from other sources in `sd_detect`.

## Licences and credit

| What | Licence | Credit |
|---|---|---|
| YOLOX code and weights | Apache-2.0 | Ge, Liu, Wang, Li and Sun, "YOLOX: Exceeding YOLO Series in 2021", Megvii |
| DAMO-YOLO code and weights | Apache-2.0 | Xu, Jiang, Chen, Huang, Zhang and Sun, "DAMO-YOLO: A Report on Real-Time Object Detection Design", Alibaba Group |
| HERIDAL | CC BY 3.0 | Božić-Štulić, Marušić and Gotovac, IJCV 2019, University of Split (FESB) |
| KIOS Small Object Aerial Person Detection | CC BY 4.0 | Makrigiorgis, Kyrkou and Kolios, KIOS CoE, University of Cyprus, Zenodo record 7740081 (2023) |
| WiSARD (measuring only) | MIT for the software; the data licence is not confirmed | Broyles, Hayner and Leung, IROS 2022, University of Washington |
| COCO, which the starting weights were trained on | CC BY 4.0 (annotations) | Lin et al., "Microsoft COCO", ECCV 2014 |

The model code is in `tools/train/vendor/` (see its `NOTICE.md`). Training and choosing
are described in `tools/train/README.md`.

Each model's manifest (`<name>.json`) lists the sets it was trained on, with their
licences and credit. Its `publish` field is `true` only when every one of those sets
allows publishing trained weights. Weights and models are never committed or uploaded
by CI. Publishing a model is the owner's decision, with the credit above.
