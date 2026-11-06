# Training the person detector (`tools/train`)

This turns published detectors into person detectors for pictures taken from the
air. Training runs on your computer's graphics card, or on a rented one at Modal (see
"Träna i molnet (Modal)"). You choose the best of them
with numbers measured on the image bank (`tools/bench`). The pictures, the weights and
the trained models never go into git or CI: they stay in your data folder
(`<data_dir>`, set by `just setup-config`).

The candidates, all Apache-2.0 and all started from their authors' COCO weights:

| Name | What | Size |
|---|---|---|
| `yolox_s` | YOLOX-s, Megvii (2021) | 9 M parameters |
| `yolox_m` | YOLOX-m, Megvii (2021) | 25 M parameters, about 2.8 times slower than `yolox_s` |
| `damoyolo_s` | DAMO-YOLO-S, Alibaba (2022) | 16 M parameters |

Each is trained on 640 px tiles of the same size and overlap as the drone looks at, made from the image bank's
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

1. **Download the image bank** (HERIDAL and KIOS, about 12 GB, 20–60 minutes; see
   `tools/bench/README.md`):

   ```bash
   just bench-fetch yes=1
   ```

2. **Download the starting weights** (2–5 minutes, about 450 MB; `only=yolox_m` fetches
   just YOLOX-m's 200 MB). Without `yes=1` it
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
   - Each recall is the model's lowest over the datasets in the val part, not the
     recall pooled over all people, so a model that gets worse on HERIDAL cannot win on
     KIOS's many people. The per-dataset numbers are printed next to each model.

   It prints the chosen model and the line `SD_DETECTOR_MODEL=<path to its .json>`.

8. **Measure the chosen model once on the test part** (20–40 minutes), and keep the
   result:

   ```bash
   just bench-detector model=person_yolox_s
   ```

   Use the name `pick-detector` chose. The numbers to report are the per-dataset
   lines it prints, HERIDAL first, not the pooled one. Each dataset is measured at its
   own threshold, so `sets=heridal` and `sets=kios` runs give the same numbers. The
   result is written to `build/bench/det_<model>_test.json`. The simulator's profile is
   made from a result on one dataset only: run `just bench-detector
   model=person_yolox_s sets=heridal` for it, and see `tools/bench/README.md`.

9. **Use it.** The on-board computer (`sd_onboard`) loads the model when
   `SD_DETECTOR_MODEL` in its environment points at the `.json` file that
   `pick-detector` printed. It checks the manifest and the ONNX file's SHA-256 before
   using it. If the file is missing or changed, only the detector is turned off, and the
   ground station gets the reason (`executor.perception_off`) so you can see that person
   detection is off. Flying goes on. When the variable is not set, no model is
   loaded. The simulator keeps its stand-in detector unless you ask for yours (next
   section).

## Träna i molnet (Modal)

Har du inget grafikkort, eller vill du träna snabbare, kan samma träning köras på ett
hyrt grafikkort hos Modal (modal.com). Du betalar per sekund som kortet används. Det är
samma kod och samma nycklar som `just train-detector`; bara datorn är en annan.

**Kör från din egen dator**

Modal startas från din egen dator (i WSL), inte från en Claude-session i molnet:
Modals anslutning (gRPC) går inte igenom den sessionens proxy.

**En gång: konto, verktyg och nyckel**

1. Skapa ett konto på modal.com och lägg in ett betalkort (Settings, Billing). Sätt
   gärna en gräns för hur mycket det får kosta per månad på samma sida.
2. Installera uv och just, om du inte redan har dem, och sedan Modals program (från
   repots mapp):

   ```bash
   bash tools/setup/wsl-setup.sh
   cd tools/train && uv sync --frozen --inexact --group cloud; cd ../..
   ```

3. Lägg in nyckeln. Enklast är att logga in i webbläsaren:

   ```bash
   cd tools/train && uv run --no-sync modal token new; cd ../..
   ```

   Nyckeln sparas i `~/.modal.toml` i din hemmapp.

   Har du redan en nyckel (modal.com, Settings, API Tokens), eller vill du inte spara
   den på disken, kan du i stället sätta den bara för terminalen du är i. Den frågar
   efter nyckeln utan att visa den, så den hamnar inte i kommandohistoriken. Skriv den
   aldrig direkt på kommandoraden:

   ```bash
   read -rp "Token id: " MODAL_TOKEN_ID; read -rsp "Token secret: " MODAL_TOKEN_SECRET; echo
   export MODAL_TOKEN_ID MODAL_TOKEN_SECRET
   ```

   Nyckeln hamnar aldrig i repot, i CI eller i utskrifter. Lägg aldrig in den i `.env`
   eller någon fil i repot.
4. Kontrollera att det fungerar. Det här kostar ingenting:

   ```bash
   cd tools/train && uv run --no-sync modal token info && uv run --no-sync modal app list; cd ../..
   ```

5. Prova med en liten körning först. Den kostar högst ungefär 3,15 $ (`max_hours=1`
   på A100-80GB, med processor och minne) och visar att hela vägen
   fungerar, ända till ONNX-filen hemma. Egen mapp och eget namn, så att den inte rör
   en riktig modell:

   ```bash
   just train-detector-cloud model=yolox_s epochs=1 limit=50 max_hours=1 out="$(just data-dir)/train/smoke" name=person_smoke
   ```

   Bilderna och startvikterna måste vara hämtade först (steg 1 och 2 ovan).

**Varje gång: träna**

Hämta bilderna och startvikterna först, som i steg 1 och 2 ovan. Sedan:

```bash
just train-detector-cloud gpu=A100-80GB model=yolox_s
```

Det här händer:

1. Den skriver ut vad det ungefär kostar: vad grafikkortet, processorn och minnet
   kostar per timme gånger timmarna, och det högsta det kan kosta (`max_hours=`, 24 timmar om du inte anger
   något). Sedan frågar den om du vill starta. Skriv `ja`. Med `yes=1` frågar den inte.
2. Den laddar upp bilderna till en privat lagringsplats hos Modal (en *volym* som heter
   `searchdrone-train`). Bara det som träningen läser skickas: `train`- och
   `val`-delarna av de set som får tränas på, aldrig den låsta `test`-delen. Nästa gång
   skickas bara nya eller ändrade filer. Första gången tar det en stund: några GB för
   HERIDAL och KIOS, mycket mer med SeaDronesSee eller UIT-Drone4.
3. Träningen körs, med samma utskrifter som hemma. Den sparar var tionde minut och
   efter varje epok. Ett anrop hos Modal får vara högst 24 timmar, så den stoppas lite
   före och startar om från senaste sparningen, tills den är klar. Det som tränats
   sedan senaste sparningen görs om, högst ungefär en kvart. Den ber inte Modal att
   försöka igen. Flyttar Modal ändå anropet till en annan dator slutar den vänta när
   tiden är slut och stoppar anropet; varje sekund räknas mot `max_hours=`, också
   den tid anropet körde innan det flyttades. Innan varje anrop skrivs den längsta tid
   anropet kan ta upp i räkningen. Den rättas när det är känt hur länge anropet körde:
   anropet skriver det själv på volymen medan det kör. Tiden mäts med vanlig klocka,
   från det att anropet skickas, så står din dator i viloläge medan anropet kör räknas
   den tiden också, och vaknar datorn efter att tiden är slut stoppar den anropet
   direkt. Efter Ctrl+C, tappat nät eller att datorn stängts av rättar nästa körning
   det, när Modal säger att anropet har slutat. Då räknas den tid anropet senast skrev,
   plus högst 36 minuter för start och det som det kan ha kört efter det, men aldrig
   mindre än den tid kommandot väntade på det. Skrev anropet aldrig någon tid räknas 16
   minuter, för start och en första sparning. Går en sparning på volymen fel försöker
   anropet igen vid nästa; går den också fel stoppar anropet sig självt, och likaså om
   en sparning inte blir klar på tio minuter. Avbröts kommandot medan anropet skickades,
   så att Modal inte hann ge det något id, väntar nästa körning en kvart: då har Modal
   stoppat det. Kan Modal inte säga att anropet har slutat, stannar nästa körning och
   säger när det senast slutar. Kör anropet ändå efter den tiden (det flyttades medan
   datorn sov, till exempel), stoppar nästa körning det. Säger Modal fortfarande att det
   kör, stannar körningen och ber dig stoppa `searchdrone-train` på modal.com/apps;
   anropet står kvar med sin längsta tid tills Modal säger att det har slutat, och
   räknas då med sin egen tid. Visar modal.com/apps att inget kör men Modal svarar ändå
   inte, säger körningen hur du stänger anropet med `close_call=`; det räknas då med sin
   längsta tid, eller mer om anropet skrev mer. Ingen körning startar ett nytt anrop
   medan ett gammalt kanske fortfarande kör, eftersom det annars kunde spara över den
   nya träningen. Går räkningen inte att läsa stannar den också: titta då på
   modal.com/apps att inget kör, och kör samma kommando med `reset_hours=` och de
   timmar körningen redan har använt. Går räkningen att läsa vägras `reset_hours=`,
   så att inga räknade timmar försvinner.
   Får den inget grafikkort från Modal innan tiden är slut, säger den det: då
   tränades inget, och du kan försöka igen senare eller med en annan `gpu=`.
4. Den gör ONNX-filen och hämtar hem `best.pt`, `person_yolox_s.onnx` och
   `person_yolox_s.json` till samma ställen som `just train-detector` och
   `just export-detector` använder. Den skriver aldrig över något du gjort på din egen
   dator: finns `last.pt`, eller en `best.pt` som inte kommer från Modal, i mappen,
   eller en ONNX-fil med samma namn som du exporterat själv, stannar den innan något
   laddas upp. Välj då en annan mapp med `out=` eller ett annat namn med `name=`.
   Sedan går du vidare med steg 6 ovan (`just bench-detector`, `just pick-detector`)
   precis som vanligt.

Avbryter du med Ctrl+C stannar grafikkortet och du slutar betala. Tappar du nätet
stannar det inom några minuter.
Den sparar inte just då. Kör samma kommando igen så fortsätter den från senaste
sparningen (var tionde minut och efter varje epok).

`fresh=1` (börja om från noll) tar först bort den gamla träningen i Modal (`last.pt`,
`best.pt` och `settings.json`), innan något laddas upp. Sedan fortsätter varje anrop
från den nya träningens egen sparning, eller börjar från noll om det inte finns någon.
Kör därför utan `fresh=1` när du fortsätter, annars tas det som tränats bort igen.
Stoppas kommandot innan den gamla träningen är borttagen, gör nästa körning det, med
eller utan `fresh=1`. Kommandot säger vilket när det stannar. Går något fel hos Modal
eller med nätet skriver den vad, på svenska, och vad du kan göra.

**Vad det kostar**

Priser från modal.com/pricing, lästa 2026-11-04. Utöver grafikkortet betalar du för
processorn och minnet bredvid det, ungefär 0,63 $ i timmen. Anropet får exakt 8 kärnor
och 32 GiB, aldrig mer, så priset per timme håller:

| `gpu=` | Grafikkortet | Allt per timme | Ungefär för `yolox_s`, 20 epoker |
|---|---|---|---|
| `A100-80GB` (standard) | 2,50 $ | 3,13 $ | 1 timme, ungefär 3,15 $ |
| `H100` | 3,95 $ | 4,58 $ | 40 minuter, ungefär 3,05 $ |
| `L40S` | 1,95 $ | 2,58 $ | 1,2 timmar, ungefär 3,10 $ |

Timmarna är en grov gissning, inte uppmätta. Fler epoker, fler `scales=` och
`upscale=` gör det längre. Vet du bättre, ange `hours=` så räknar den med det.
`max_hours=` är taket för hela körningen, alla anrop och alla gånger du kört
kommandot för samma mapp tillsammans: när de timmarna är slut stannar den, och samma
kommando med ett större `max_hours=` fortsätter. Den visar hur mycket som redan är
använt innan den frågar. `fresh=1` börjar om och räknar timmarna från 0.
Lagringen i volymen kostar lite eller inget; se modal.com/pricing.

**Vad som aldrig laddas upp**

- Set som bara får mätas på: WiSARD, GPL-set, set utan licens och icke-kommersiella
  set. Koden stoppar dem innan något skickas; det går inte att stänga av.
- Simulatorns bilder (`simfarm=1` och `simstills`): de stannar på din dator.
- Ortofoton från Lantmäteriet, och allt annat utanför bildbanken och startvikterna.

**Nycklar** utöver dem för `just train-detector`:

| Nyckel | Vad den gör |
|---|---|
| `gpu=A100-80GB` | Grafikkortet: `A100-80GB`, `H100` eller `L40S`. |
| `yes=1` | Starta utan att fråga. |
| `hours=2` | Din egen gissning av timmarna, för kostnaden som visas. |
| `max_hours=24` | Högst så många timmar på grafikkortet för körningen, allt sammanräknat. |
| `name=person_yolox_s` | Namnet på ONNX-filen och manifestet. |
| `init=<best.pt>` | Som för `just train-detector`. Måste vara en `.pt`-fil på högst 1 GB. |
| `close_call=<id>` | Bara när körningen ber om det: ett gammalt anrop som modal.com/apps visar inte kör, fast Modal inte säger att det har slutat, räknas med sin längsta tid och stoppar inte längre körningen. Går först efter den tiden. Ge den bara en gång. |
| `reset_hours=<h>` | Bara när körningen ber om det: timräkningen går inte att läsa. Den börjar om på `h` timmar, utan öppna anrop; ge de timmar körningen redan har använt, hellre för många. En timräkning som går att läsa ändras aldrig: då vägras `reset_hours=`. Går inte ihop med `fresh=1` eller `close_call=`. Ge den bara en gång. |

`device=` och `simfarm=` går inte att använda i molnet.

**Ta bort det som ligger hos Modal**

När du är klar kan du ta bort volymen med bilderna och körningarna:

```bash
cd tools/train && uv run --no-sync modal volume delete searchdrone-train; cd ../..
rm "$(just data-dir)"/train/.modal-searchdrone-train.json "$(just data-dir)"/train/.modal-hours-*.json
```

Den andra raden glömmer vad som laddats upp och hur många timmar körningarna använt,
så att allt skickas igen nästa gång.
Det du hämtat hem ligger kvar på din dator.

## The larger YOLOX-m

`yolox_m` is the same network as `yolox_s`, deeper and wider. It is trained, exported
and measured the same way:

```bash
just detector-fetch yes=1 only=yolox_m
just train-detector model=yolox_m epochs=20
just export-detector model=yolox_m
just bench-detector model=person_yolox_m part=val
```

- **Batch 4 by default**, not 8. In a short test on an RTX 3070, batch 4 took about
  2.1 GB of the card and batch 8 about 3.6 GB, on top of what Windows already used
  (2.9 GB). Batch 8 fits there, but leaves little room for tiles crowded with people,
  and running out of memory hours into a run means starting over with `fresh=1`.
  `batch=8` is yours to try.
- **It starts from Megvii's COCO YOLOX-m weights.** `init=` with a `yolox_s` run is
  refused: the layers have other widths, so the weights cannot be moved over.
- **Slower on the drone.** Megvii's README gives 73.8 GFLOPs per 640 px tile against
  26.8 for `yolox_s`, so each tile takes roughly 2.8 times as long on the Raspberry Pi 5.
  Measure it with `just bench-detector` before choosing it; `pick-detector` prefers the
  faster model when two find people equally well.
- The ONNX file is about 100 MB (`yolox_s`: 36 MB), under the 128 MB the on-board
  computer accepts.

## Go on from a model you already trained

The training mixes every fetched set that may be trained on (today HERIDAL and KIOS).
If you already have a HERIDAL-only model, you can start from its weights instead of
COCO's with `init=`. Give the new run its own folder with `out=`: `init=` may not lie in
the folder the run writes to.

```bash
just bench-fetch yes=1 sets=kios
just train-detector model=yolox_s init="$(just data-dir)/train/yolox_s/best.pt" out="$(just data-dir)/train/yolox_s_mix" epochs=12
just export-detector model=yolox_s train="$(just data-dir)/train/yolox_s_mix" name=person_yolox_s_mix
```

After each epoch the val AP50 is printed per dataset (`heridal 0.61; kios 0.55`). The
epoch kept is the one whose lowest dataset AP50 is highest, so a gain on KIOS, which has
many more people, never makes up for a loss on HERIDAL. The manifest lists every set
the weights were trained on, including those of the `init=` run, with their credit, and
the `init=` file's name and SHA-256 (not its folder).

Measure it on the val part next to the old model. Both must be measured on the same
split: re-measure the old model too, because a result from before KIOS was added was
measured on HERIDAL's val part only, and `just pick-detector` refuses to compare
results measured on different pictures.

```bash
just bench-detector model=person_yolox_s part=val
just bench-detector model=person_yolox_s_mix part=val
just pick-detector
```

## Train with the two extra sets

Two more sets of real pictures can be trained on (the owner's decision of 2026-11-01):
KIOS car/person (people on a parking lot in Cyprus, 21–31 px, every tenth video frame,
about 1,400 pictures) and SeaDronesSee (swimmers in open water, about 10,500 pictures).
They are never measured on: the locked test part stays HERIDAL's and KIOS's land
pictures, so the new model is kept only if it beats the current one there.

```bash
just bench-fetch yes=1 sets=kios_carperson,seadronessee   # about 19 GB, 1-3 hours
just train-detector model=yolox_s sets=heridal,kios,kios_carperson,seadronessee out="$(just data-dir)/train/yolox_s_balanced"
just export-detector model=yolox_s train="$(just data-dir)/train/yolox_s_balanced" name=person_yolox_s_more
just bench-detector model=person_yolox_s part=test
just bench-detector model=person_yolox_s_more part=test compare=build/bench/det_person_yolox_s_test.json
```

Each epoch draws as many tiles from every set (`balance=sets`, the default). Without
that, SeaDronesSee, with about 70 % of the tiles, drowned the land sets: after 4 epochs
the val AP50 on land fell (HERIDAL 0.42, KIOS 0.31, against 0.64 and 0.38 for the model
before) while the sea and parking-lot sets rose (0.76 and 0.78). An epoch still has as
many steps as there are tiles; the small sets are gone through more than once, the big
one only in part (a new part each epoch). `balance=tiles` takes every tile once, as
before; `balance=heridal:2,kios:2,kios_carperson:1,seadronessee:1` gives each set its
own share.

The val AP50 is printed for all four sets after each epoch, but the epoch kept is the
one whose weakest *land* set (HERIDAL, KIOS: the sets that are measured on, `pick=`) is
best, since the land is where the model is used. The manifest names those sets
(`val_ap50_pick`) and the balance. SeaDronesSee's pictures are large (mostly
3840×2160), so its tiles take far more room and time than the other sets: expect the
tile cache under `out=` to grow by several GB and each epoch to take several times
longer (a rough guess, not measured). The training does not know whether swimmers help on land; only the test part
says so.

## Train at our cameras' scales

The real pictures were taken with other cameras at other heights, so their people are
not the size ours will see. Half of HERIDAL's hikers are 56-87 px long (median 71),
half of KIOS's people 12-23 px (median 15), SeaDronesSee's swimmers 26-57 px. Our wide camera (Camera Module 3 at full resolution, 4608 px over 66°) sees a
standing person at about 33 px from 50 m, and the tele (Pi HQ with a lens of about 20°)
at about 100 px. `scales=` makes the training see people at those sizes (the owner's
decision of 2026-11-03):

```bash
just train-detector model=yolox_s sets=heridal,kios,kios_carperson,uit_drone4 scales=native,wide,tele out="$(just data-dir)/train/yolox_s_scales"
just export-detector model=yolox_s train="$(just data-dir)/train/yolox_s_scales" name=person_yolox_s_scales
just bench-detector model=person_yolox_s_scales part=test compare=build/bench/det_person_yolox_s_test.json
```

- **What happens.** Each training picture is cut as it is (`native`) and once more for
  each camera, resampled so its people are as many pixels as that camera sees them.
  The height is drawn for each picture between 30 and 60 m, the search heights: `wide`
  gives about 59-118 px per metre (a person about 28-56 px), `tele` about 192-383 px
  per metre (about 90-180 px). A number such as `scales=native,150` is a fixed scale in
  pixels per metre.
- **How a picture's scale is known.** HERIDAL's source gives about 2 cm per pixel
  (50 px per metre, `px_per_m` in `tools/bench/datasets.toml`). The other sets state no
  scale, so it is read off their person boxes: the median longest side over 0.47 m (a
  standing person seen from above, from the owner's figures). A picture with at least
  three people uses its own, one with fewer its sequence's (the same flight or site),
  and an empty picture its set's. HERIDAL shows why the register's figure wins when
  there is one: many of its people lie or sit, so their boxes are about 1.4 m long and
  would put the scale three times too high.
- **Never enlarged more than twice** (`max_up=2`). An enlarged person is blurred, not
  sharper, so a KIOS person of 15 px does not become a tele person of 100 px; it stops
  at 30 px. Shrinking has no limit: it is what a coarser camera does. Two scales that
  end up within 15 % of each other are cut once. So for `wide`, HERIDAL is enlarged
  1.2-2 times and KIOS twice (to about 30 px); for `tele` both stop at twice, which
  `wide` mostly gave already, so it is cut once. UIT-Drone4's pedestrians (median
  59 px) are shrunk for `wide` and enlarged up to twice for `tele`. Real pictures with
  people as large as the tele will see them are rare; flights with our own tele camera
  will fill that gap.
- **The val and test parts are not resampled.** The best epoch is picked on the val
  pictures as they are, the same way the locked test part is measured.
- **More tiles.** Each extra scale adds roughly (factor)² as many tiles of that
  picture: a picture enlarged twice gives about four times the tiles. The tile cache
  under `out=` and each epoch grow accordingly.
- **See the effect.** After cutting, the run prints for each set and scale how many
  people it gave and their size (quartiles of the longest side).
  `just bench-detector` prints the recall per person size (0-16, 16-24, 24-32, 32-48,
  48-64, 64-96, 96-128, 128-192 and 192+ px); compare the new model's table with the
  old one's.

**UIT-Drone4** (CC BY 4.0, Zenodo 14862652) adds real people at the sizes between our
two cameras: city traffic in Vietnam at 3840×2160, every tenth frame, 1,523 pictures
with 7,193 pedestrians. Its motorcycles are boxed with their riders, people who are
not boxed as persons; a tile that touches a motorcycle is left out, so the model is
never taught that a rider is not a person. It is trained on only, never measured on.
Fetch it once (37 GB, several hours on a home line):

```bash
just bench-fetch yes=1 sets=uit_drone4
```

## Try your trained detector in the simulator

The simulator's default person detector is a stand-in (the oracle, which knows where
everyone is). `detector=model` swaps it for your model. It needs Gazebo
(`just sim-setup`) and, once, the 3D person models from Gazebo Fuel (CC0 1.0 and CC BY
4.0, credits in `simulation/farm/people/ATTRIBUTION.md`; fetched into `build/`, never
committed):

```bash
just sim-farm-setup
export SD_DETECTOR_MODEL=/path/to/person_yolox_s.json   # what pick-detector printed
```

Instead of the variable you can add `model=/path/to/person_yolox_s.json` to each command
below. Use the full path.

| Command | What happens |
|---|---|
| `just demo detector=model` | The demo in Gazebo. The persons of the search scenario SK01 (seed 1) lie and sit in the world, the drone carries the stills camera, and `sd-onboard` runs your model on its pictures. Finds show up in the panel. |
| `just e2e-search detector=model` | The search demo (SK01) headless in Gazebo on this computer, with your model instead of the stand-in. The report is `runs/<run>/search_report.md`; its "Motorer" line names your model. |
| `just demo-search detector=model` | The same search demo. |
| `just sim-headless detector=model` | Builtin physics has no camera: `sd-onboard` gets your model and checks it, but sees no pictures. Useful to check that the model loads. |

Before anything starts, the manifest and the ONNX file's SHA-256 are checked. If the
file is missing or changed, the command stops with the reason (exit code 2); it never
falls back to the stand-in. The persons stand exactly where the scenario (and the
stand-in) has them, so the report's truth and the camera agree. Each model has the colour
of the person's upper garment; the faces and clothes are not real people.

CI always uses the stand-in: there are no weights in CI. The run asserts what the stand-in
run asserts: the person in the red jacket is found and confirmed. A real detector can miss
the person; then `just e2e-search detector=model` fails without a confirmed find. That
says something about the model (more training pictures, a lower threshold), not the
simulator.

## If something goes wrong

| What you see | What to do |
|---|---|
| "Grafikkortets minne räckte inte" | Run it again with `batch=4 fresh=1`. |
| "COCO-vikterna saknas" | Run step 2. |
| "har en träning med andra inställningar" | You changed a setting (`batch`, `epochs`, …) of a run that was already started. Run it with the old settings, or start over with `fresh=1`. A run started before `balance=` existed carries on with `balance=tiles` and `pick=` naming all its sets. |
| "Inga bilder att träna på" | Run step 1. |
| "ONNX-filen räknar inte som PyTorch" | Nothing was written. Run step 4 again. If it happens twice, report it. |
| "Tränar på processorn (cpu)" at the start, or "Inget grafikkort syns" | PyTorch does not see the card. Check `nvidia-smi` in WSL and update the Windows driver; stop with Ctrl+C before it starts on the processor. |

Useful keys for `just train-detector`:

| Key | What it does |
|---|---|
| `epochs=20` | Passes over the training tiles. |
| `batch=8` | Tiles per step; 8 fits in 8 GB. |
| `sets=heridal,kios` | Which fetched sets to train on (default: every one that may be trained on). |
| `balance=sets` | How each epoch draws its tiles: `sets` (default) as many from every set, `tiles` every tile once (a big set counts more), or weights such as `heridal:2,kios:2,kios_carperson:1,seadronessee:1` naming every set. |
| `pick=heridal,kios` | The sets whose val AP50 picks the best epoch (default: those in `sets=` that are measured on, or all of them if none is). Every set's AP50 is still printed. |
| `upscale=2` | Each tile covers 320×320 picture pixels and is enlarged to 640 px, so a person of 15 px is trained on (and, on the drone, looked for) as 30 px. The manifest carries it, so `bench-detector`, the simulator and the drone enlarge their tiles the same way. Costs about 3.8 times the tiles: 170 instead of 45 per 12 MP picture. Default 1 (as before). |
| `scales=native,wide,tele` | Also cut each training picture resampled so its people are as large as the wide camera (about 33 px) or tele camera (about 100 px) sees them at 30-60 m; a number is pixels per metre. Default `native` (as before). See "Train at our cameras' scales". |
| `max_up=2` | The most a picture is enlarged to reach a scale (1-8). |
| `init=<best.pt>` | Start from an earlier run's weights instead of COCO's; use with a new `out=`. |
| `out=<folder>` | Where the run is saved (default `<data_dir>/train/<model>`). |
| `simfarm=1` | First pre-trains for `simfarm_epochs=3` epochs on the simulator's pictures (`just sim-farm`, then `just bench-fetch yes=1 sets=simfarm`). Those pictures are never measured on. |
| `limit=20` | Only the first 20 pictures, to try it out quickly. |
| `device=cpu` | Train on the processor. |
| `workers=4` | How many picture loaders run at once. |

## Which data is used, and what may be published

- **HERIDAL** (CC BY 3.0) is trained on. Weights trained on it may be published if
  HERIDAL is credited.
- **KIOS** (CC BY 4.0, University of Cyprus) is trained on. Weights trained on it may be
  published if KIOS is credited (the manifest carries the credit).
- **KIOS car/person** (CC BY 4.0, University of Cyprus) and **SeaDronesSee** (CC0 1.0,
  University of Tübingen) are trained on only when `sets=` names them, never measured
  on. Weights trained on them may be published with their credit (the manifest carries
  it). See "Train with the two extra sets" above.
- **UIT-Drone4** (CC BY 4.0, University of Information Technology, VNU-HCM) is trained
  on only when `sets=` names it, never measured on, with the same rule for its credit.
  See "Train at our cameras' scales" above.
- **AOS** (JKU Linz, CC BY 4.0) is not used: it has no person boxes for single colour
  pictures (see `tools/bench/README.md`).
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
- `cloud.py`: `just train-detector-cloud`. Decides what is uploaded (the licence guard),
  what changed since last time, what it costs (the GPU hours per run, a hard ceiling),
  where the results go (never over a local run) and which repository files go into the
  container (`IMAGE_FILES`, an allowlist). It never imports `modal`, so its tests run
  without Modal, the network or an account.
- `gpu_call.py`: what one GPU call does: `train_detector.py`, then `export_onnx.py`,
  stopped with SIGINT when the call's own budget (`budget_s`, counted from when the
  container started) is used up, and killed if it has not stopped after a grace. No
  Modal; tested with real child processes.
- `modal_app.py`: the Modal side: the container image (this folder's `uv.lock` with
  PyTorch for CUDA, plus the allowlisted files), the volume and the function that runs
  `gpu_call` on the GPU, without retries, keeping a mark with its seconds on the
  volume (`.calls/`). Its tests use a fake `modal`. Modal is the `cloud` dependency
  group, installed only by `just train-detector-cloud`.
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

Pyright does not check `families.py`, `train_detector.py`, `export_onnx.py` or
`tests/`, because they need PyTorch, which the main environment does not have, nor
`modal_app.py`, which needs Modal. CI never calls Modal.

The on-board side is `sd_detect.onnx` (`software/detector`). It needs only ONNX Runtime,
never PyTorch.
