# fpv7: CAD model of the 7-inch quadcopter

A parametric model of the 7-inch quadcopter, written as code with
[CadQuery](https://cadquery.readthedocs.io). Every size and mass is a number in
[`params.yaml`](params.yaml); change a number and rebuild. The frame follows GEPRC's
dimensioned drawing of the Mark4, with its plates, arms, standoffs, screws and nuts; the
motors are bells with their mounting screws, the propellers have blades, and the cables,
antennas, XT60, battery straps and canopy screws are in place. Boards and the battery are
boxes the size of the real part. The model shows whether everything fits, what the drone
weighs, where its centre of gravity is and roughly how much drag it has, and it opens in
Autodesk Fusion for further work (see *Öppna och ändra i Fusion*).

```bash
just cad            # STEP, STL, report and pictures in build/cad/fpv7/ (not committed)
just cad-detailed   # the same, plus fpv7_detailed.step with the manufacturers' 3D models
just cad-test       # overlaps, propeller margins, camera view, drag, repeatable exports
```

| File | What it does |
|---|---|
| `params.yaml` | Every size, mass and position (millimetres, grams, degrees) |
| `parts.py` | One solid per kind of part |
| `assembly.py` | Puts the parts in place and writes the exports (`just cad`) |
| `checks.py` | Mass, centre of gravity, inertia, overlaps, propeller margins, camera view |
| `aero.py` | Drag: Cd A part by part, with shielding |
| `cfd.py` | An OpenFOAM case to check `aero.py` (run by hand, see *Drag*) |
| `render.py` | The pictures (`fpv7_views.png`, `fpv7_closeup.png`) |
| `vendor.py`, `vendor.yaml` | Download, check and place the manufacturers' 3D models (`just cad-detailed`) |
| `tests/` | The acceptance checks (`just cad-test`) |

Output in `build/cad/fpv7/` (the CI artifact `cad-fpv7`):
- `fpv7.step`: the whole drone in millimetres, z up, with a sub-assembly per group
  (`frame_group`, `fastener_group`, `motor_group`, `prop_group`, `prop_disc_group`,
  `electronics_group`, `computer_group`, `camera_group`, `battery_group`, `printed_group`,
  `cover_group`, `cable_group`) and one named, coloured solid per part;
- `step/<part>.step`: every part on its own, in place in the drone's coordinates, and
  `step/canopy_form.step`: the canopy's outer form, a closed solid of ten faces to edit;
- `fpv7.stl` and `stl/<part>.stl`: the printed parts (`canopy.stl`, `nose_fairing.stl`,
  `pi_tray.stl`, `camera_mount.stl`, `leg_*.stl`, `gps_mast.stl`) go to the slicer;
- `report.md` / `report.json`, `fpv7_views.png` (three angles) and `fpv7_closeup.png`
  (the centre of the frame without the canopy);
- `build/cad/fpv7-detailed/fpv7_detailed.step` (only with `just cad-detailed`), kept
  outside `build/cad/fpv7/` because it holds manufacturer models.

Two builds give byte-identical STEP, STL and report files: the STEP header has a fixed
date, assembly occurrences are numbered from 1 and the colour styles, which OCCT writes
in a different order every run, are written again in a fixed order.

## Axes and layout

x forward, y left, z up. The origin is the frame centre; z = 0 is the underside of the
bottom plate. Motors 1-4: front left, rear left, rear right, front right; the front-left
and rear-right propellers turn clockwise seen from above.

- **Frame:** GEPRC Mark4 7" (see *Frame*). The kit's 121 g is spread over the plates,
  arms, standoffs, screws and nuts by volume times density.
- **Between the plates:** the flight stack (MicoAir743 V2 on the 55 A AM32 4-in-1 ESC) on
  soft-mounted posts over the arm nuts, the 5 V BEC in front of it and the ELRS receiver
  behind it.
- **On the top plate:** the Raspberry Pi 5 with Active Cooler and AI HAT+ 2 on a printed
  tray with rubber dampers, the Remote ID board at the front and the GPS on a mast at the
  rear.
- **Under the bottom plate:** the 6S2P battery on two straps, the mated XT60 behind it,
  and ahead of it the camera looking straight down, with the long side of the image
  across the drone.
- **Legs:** four printed legs under the arms that keep the battery 10 mm off the ground.
- **Motors and propellers:** EMAX ECO II 2807 drawn from EMAX's drawing (stator mount,
  bell, M5 shaft and nut), held by four M3 x 8 screws from under the arm. The HQ 7x4x3 is
  drawn with three blades (`display.prop_blades`) for the pictures and the STEP file; the
  checks use the swept disc (177.8 mm, 10 mm high) that the blades stay inside.
- **Cables** (approximate routes, as tubes with their mass per metre): three motor leads
  along each arm, the 12 AWG ESC pigtail from the XT60 round the rear of the frame, the
  BEC's lead to the Pi through the top plate's front hole, the camera's ribbon up in front
  of the frame, the GPS lead down the mast, and the receiver's two antennas out of the
  rear in a V, pointing down.
- **Covers:** a printed top canopy over the companion computer and a printed nose under
  the frame ahead of the battery (see *Printed covers*).

Not modelled: connectors and components on the boards (the boards are boxes as tall as
their tallest part), the soft-mount grommets (only the gap they leave), a 4G modem and a
stand-alone Remote ID beacon. Solder, heat-shrink and zip ties are a point mass
(`misc.mass`).

## Frame

GEPRC publishes no STEP or DXF for the Mark4, only its sizes and photos. The model uses:

- from GEPRC's product page: the H layout, 193 x 223 mm between motor axes (295 mm
  diagonal), 2.5 mm plates, 5 mm arms, 2 mm protect plates, 8 M3 x 25 mm and 4
  M3 x 18 mm aluminium standoffs, the 30.5 and 20 mm stack patterns and the 121 g mass;
- from retailer listings: the 16 x 16 and 19 x 19 mm motor patterns;
- **(osäkert)** measured on GEPRC's dimensioned photo of the Mark4 plates (the 5" photo,
  scaled from its 172 mm dimension at about 5.2 pixels per mm, so about +-1 mm): the
  outlines of the bottom and top plates and their windows (`frame.*.outline`,
  `frame.*.windows`, half outlines mirrored in y), the standoff positions
  (`frame.standoff.positions`), the arm's width, tip and teardrop window, and where the
  arm screws sit. The 7" kit uses the same plates and longer arms, as GEPRC's
  [7" kit photo](https://geprc.com/wp-content/uploads/2019/06/03-5332127494.jpg) shows;
  the photo is a little asymmetric front to rear (1.7 mm), which the model evens out.

Each arm runs from its outer root screw to its motor (115 mm at 53 degrees from the x
axis) and lies on the bottom plate. Two screws hold each arm root: flat-head M3 from
below through the bottom plate, the arm and a 2 mm protect plate, the outer one into an
18 mm standoff, the inner one into a thin M3 nut that stays under the ESC. The 25 mm
standoffs carry the top plate with flat-head M3 x 8 screws from above and below.

## Printed covers

Both covers print in PETG or ASA (`material`) with a `wall` of 1.2 mm; their mass is
their volume times the material's density. Together they must stay under
`checks.cover_mass_max` (80 g).

- **Top canopy** (`canopy.stl`): a loft through the sections in `canopy.stations`
  (rectangles with rounded top corners, from the nose to the tail) minus the same form
  shrunk by the wall. Its rim stands `canopy.rim_clearance` above the propeller discs,
  and seen from above it stays `canopy.prop_keep_out` outside them. Openings: an intake
  slot in the nose, a roof vent over the cooler fan and the AI HAT heatsink, an exhaust
  slot in the tail and a hole for the GPS mast with room for a rubber grommet and the GPS
  lead. Check the Pi's and the Hailo's temperatures with the canopy on during the first
  bench test.
- **Fastening:** four M2 x 4 button-head screws through the canopy's side walls into
  brass heat-set inserts in four bosses on the Pi tray. Screws rather than printed clips
  because the canopy sits over the Pi and its cooler, where PETG clips creep and loosen in
  the warmth, and a canopy that comes off in a hard landing or a crash must not land in a
  propeller; the screws hold in both cases and come out with a 1.3 mm hex key.
- **Nose fairing** (`nose_fairing.stl`): a rounded nose hanging under the bottom plate in
  front of the battery, around the camera. The camera looks through a window cut from
  the field of view plus `nose_fairing.window_extra_deg` on every side. The fairing ends
  ahead of the battery straps, which stay free.

## Drag

`aero.py` estimates Cd A, the drag area, part by part. Every part except the propeller
discs is projected onto the plane across the oncoming air on a 0.5 mm grid, with how far
upstream its front and rear faces lie. A point of a part counts fully if nothing lies
ahead of it, at `aero.wake_factor` (0.35) if another part lies ahead of it within
`aero.wake_length` (it sits in that part's wake, where the air is slower), and not at all
if it lies inside another part (a screw in a standoff). Each part has a drag coefficient
by its shape (`aero.cd_*`: rounded covers, round parts, thin plates edge-on, box-like
parts). The report gives the frontal area, Cd A and drag at `aero.speed` with and without
the covers, the split by shape and the largest contributors, next to the range that NASA
measured in a wind tunnel for whole quadcopter bodies without rotors (0.018-0.023 m^2 for
larger drones; `aero.reference_cda_m2`). All coefficients are **(osäkert)**: textbook
values for single bodies, to be calibrated against flight logs.

CFD_SECTION

## Öppna och ändra i Fusion

`fpv7.step` är i millimeter med z uppåt, en underkomponent per grupp och en namngiven,
färgad kropp per del.

1. **Z uppåt:** Fusion har y uppåt som standard. Välj *Preferences > General > Default
   modeling orientation: Z up* innan du öppnar filen, annars ligger drönaren på sidan.
2. **Öppna:** *File > Open > Open from my computer* och välj `fpv7.step` (från
   CI-artefakten `cad-fpv7` eller `build/cad/fpv7/` efter `just cad`). Fusion laddar upp
   filen och gör om den till en design med komponenterna `frame_group`, `motor_group`,
   `prop_group` och så vidare. Dölj `prop_disc_group` (ögat i webbläsaren till vänster):
   det är skivorna som kontrollerna använder, de ligger runt propellrarna.
3. **Ändra kåpan:** öppna `step/canopy_form.step`, kåpans yttre form som en sluten kropp
   med tio ytor. Ändra formen (t.ex. *Modify > Press Pull* på taket, eller *Fillet*
   på kanterna), skala sedan ur den med *Modify > Shell*: markera undersidan och ge
   tjockleken 1.2 mm (`canopy.wall`). Skär sedan ut intaget, ventilen, avluftningen,
   hålet för GPS-masten och de fyra hålen för M2-skruvarna (2.2 mm) med *Extrude > Cut*.
   Lägg in den i drönaren (högerklicka på filen i datapanelen, *Insert into Current
   Design*) för att se att den går fri från propellrarna och Pi-brickan: båda filerna har
   samma koordinater. Det säkraste är ändå att ändra
   `canopy.stations` i `params.yaml` och köra `just cad`: då körs alla kontroller
   (propellermarginal, kameravy, vikt).
4. **Exportera STL för utskrift:** högerklicka på kroppen i webbläsaren och välj *Save as
   Mesh*, format STL (binär), enhet millimeter, förfining *High*. Eller *File > 3D Print*
   för att skicka den direkt till en slicer.

## Manufacturer 3D models

`just cad-detailed` downloads the models below from the manufacturer, checks each archive's
SHA-256 (`vendor.yaml`) and writes `build/cad/fpv7-detailed/fpv7_detailed.step`, where they replace the
simplified envelopes. The test suite then checks that each model sits within 2 mm of its
envelope, so the checks on the simple model hold for the real parts. The files are never
committed: the Raspberry Pi 5 model alone is 78 MB, and the detailed STEP about 280 MB.
CI builds the detailed STEP but does not publish it.

| Part | Model | Licence | In the repository | Checked |
|---|---|---|---|---|
| Raspberry Pi 5 | [RaspberryPi5-step.zip](https://datasheets.raspberrypi.com/rpi5/RaspberryPi5-step.zip) | MIT, Copyright (c) 2026 Raspberry Pi Ltd (`LICENSE.txt` in the zip) | No (size); downloaded | 2026-10-18 |
| Camera Module 3 | [camera-module-3-step.zip](https://datasheets.raspberrypi.com/camera/camera-module-3-step.zip) | No licence in the file; Raspberry Pi's [design file terms](https://www.raspberrypi.com/licensing/) do not name 3D models | No; downloaded for local use only | 2026-10-18 |
| Holybro Micro M10 | [Holybro downloads](https://docs.holybro.com/gps-and-rtk-system/m8n-m9n-m10-gps/micro-m10-m9n-gps/download) (`micro-gps_asm.stp`) | No licence; Holybro's terms of service forbid copying | No; drawn from the datasheet, download it yourself | 2026-10-18 |
| Active Cooler, AI HAT+ 2 | none published (mechanical drawings only) | - | Drawn from the drawings | 2026-10-18 |
| GEPRC Mark4 7" | none published (no STEP or DXF) | - | Drawn from GEPRC's dimensioned photo (see Frame) | 2026-10-18 |
| EMAX ECO II 2807, HQ 7x4x3, MicoAir743 V2, 55 A ESC, Matek BEC12S-PRO, RadioMaster RP4TD, XT60H | none published | - | Drawn from the manufacturers' drawings and published sizes | 2026-10-18 |

## Checks

`checks.py` treats every part as a solid of uniform density with the mass from
`params.yaml`. Inertia is about the centre of gravity in the body axes, in kg m^2.

1. No two parts share volume (screws, nuts, cables and canopy screws included).
2. Every propeller disc is at least `checks.prop_margin_min` (5 mm) from every other part
   and from the other discs; its own motor is the only exception. Seen from above,
   nothing that sits above the discs hangs over them (`checks.plan_margin_min`, 3 mm).
3. Nothing but the camera itself is inside the camera's field of view (66 x 41 degrees,
   checked to 500 mm below the lens).
4. The centre of gravity is within `checks.cog_offset_max` (5 mm) of the centre in x and y,
   and the take-off mass is within `max_takeoff_mass_kg` in `hardware/limits.yaml`.
5. Mass, centre of gravity and inertia are computed from the model and compared with
   the reference values in a table (`report.md`).
6. The printed covers weigh at most 80 g, keep 5 mm from the propellers, overlap nothing
   and leave the camera's view clear.
7. With covers, Cd A and drag are lower than without; the drag estimate is tested against
   hand calculations (frontal area, wake, a part inside another, pitch) and lands within
   a factor of two of NASA's measured range.
8. The propeller blades stay inside their discs; the canopy form is one closed solid with
   few faces; `fpv7.step` has its groups, names and colours and reads back.
9. The STEP, STL and report files are the same on every build.
10. After `just cad-detailed`, every manufacturer model fits its envelope within 2 mm.
11. The download checks in `vendor.py` (https only, also after redirects, pinned SHA-256,
    size limits, only the named file extracted) are tested offline.

## Sources

Read on the date given. **(osäkert)** marks a value that is uncertain: a value no
manufacturer publishes, a measurement from a photo, an estimate, or a design default
that may change. Retailer data is used only where the manufacturer publishes none. The
lists of points (`frame.*.outline`, `frame.*.windows`, `frame.standoff.positions`,
`canopy.stations`) are described under *Frame* and *Printed covers*.

| Value | Number | Status and note | Source | Date |
|---|---|---|---|---|
| `frame.wheelbase` | 295 mm | motor to motor | [GEPRC Mark4](https://geprc.com/product/gep-mark4-frame/) | 2026-10-18 |
| `frame.motor_spacing` | 193, 223 mm | front-rear (x) and side-side (y) between motor axes; sqrt(193^2 + 223^2) = 295 | GEPRC Mark4 ('Size 193 x 223') | 2026-10-18 |
| `frame.mass` | 121 g | 7" kit; the shop field "150 g" is the shipping weight | [GEPRC Mark4](https://geprc.com/product/gep-mark4-frame/) | 2026-10-18 |
| `frame.bottom_plate.thickness` | 2.5 mm |  | [GEPRC Mark4](https://geprc.com/product/gep-mark4-frame/) | 2026-10-18 |
| `frame.bottom_plate.corner_radius` | 3 mm | (osäkert) outline corners | Measured on [GEPRC's dimensioned Mark4 photo](https://geprc.com/wp-content/uploads/2019/06/03-8505339277.jpg) | 2026-10-18 |
| `frame.top_plate.thickness` | 2.5 mm |  | [GEPRC Mark4](https://geprc.com/product/gep-mark4-frame/) | 2026-10-18 |
| `frame.top_plate.corner_radius` | 3 mm | (osäkert) outline corners | Measured on [GEPRC's dimensioned Mark4 photo](https://geprc.com/wp-content/uploads/2019/06/03-8505339277.jpg) | 2026-10-18 |
| `frame.top_plate.cam_hole` | 82, 8 mm | (osäkert) [x, diameter] of the round hole at the front | Measured on [GEPRC's dimensioned Mark4 photo](https://geprc.com/wp-content/uploads/2019/06/03-8505339277.jpg) | 2026-10-18 |
| `frame.arm.thickness` | 5 mm |  | [GEPRC Mark4](https://geprc.com/product/gep-mark4-frame/) | 2026-10-18 |
| `frame.arm.width` | 12 mm | (osäkert) consistent with the 16.5 g spare arm | Measured on [GEPRC's dimensioned Mark4 photo](https://geprc.com/wp-content/uploads/2019/06/03-8505339277.jpg) | 2026-10-18 |
| `frame.arm.root_screw` | 27.2, 19.3 mm | (osäkert) outer arm screw [x, y], made symmetric front to rear | Measured on [GEPRC's dimensioned Mark4 photo](https://geprc.com/wp-content/uploads/2019/06/03-8505339277.jpg) | 2026-10-18 |
| `frame.arm.root_length` | 14 mm | (osäkert) | Measured on [GEPRC's dimensioned Mark4 photo](https://geprc.com/wp-content/uploads/2019/06/03-8505339277.jpg) | 2026-10-18 |
| `frame.arm.root_hole_gap` | 9 mm | (osäkert) second arm screw along the arm | Measured on [GEPRC's dimensioned Mark4 photo](https://geprc.com/wp-content/uploads/2019/06/03-8505339277.jpg) | 2026-10-18 |
| `frame.arm.tip_width` | 25 mm | (osäkert) spade round the motor | Measured on [GEPRC's dimensioned Mark4 photo](https://geprc.com/wp-content/uploads/2019/06/03-8505339277.jpg) | 2026-10-18 |
| `frame.arm.tip_length` | 24 mm | (osäkert) arm past the motor axis | Measured on [GEPRC's dimensioned Mark4 photo](https://geprc.com/wp-content/uploads/2019/06/03-8505339277.jpg) | 2026-10-18 |
| `frame.arm.window` | 11, 20, 5 mm | (osäkert) teardrop in the tip: from, to, width | Measured on [GEPRC's dimensioned Mark4 photo](https://geprc.com/wp-content/uploads/2019/06/03-8505339277.jpg) | 2026-10-18 |
| `frame.arm.motor_bore` | 7 mm | (osäkert) centre hole for the motor's circlip | Estimate, no drawing published | - |
| `frame.arm.motor_holes` | 16, 19 mm | 16 x 16 and 19 x 19 patterns, M3 | Retailer listings of the Mark4 7" (Rotorama, BZBUAS) | 2026-10-18 |
| `frame.reinforce_plate.thickness` | 2 mm | 2 mm protect plates | [GEPRC Mark4](https://geprc.com/product/gep-mark4-frame/) | 2026-10-18 |
| `frame.reinforce_plate.pad` | 4.5 mm | (osäkert) pad radius round the arm screws | Estimate, no drawing published | - |
| `frame.reinforce_plate.waist` | 3.5 mm | (osäkert) half width of the bar between the pads | Estimate, no drawing published | - |
| `frame.standoff.height` | 25 mm | 8 x M3 x 25 mm, aluminium 6061 | [GEPRC Mark4](https://geprc.com/product/gep-mark4-frame/) | 2026-10-18 |
| `frame.standoff.diameter` | 5 mm | (osäkert) | Estimate, no drawing published | - |
| `frame.standoff.bore` | 3.2 mm | M3 clearance | Design default | - |
| `frame.arm_standoff.height` | 18 mm | 4 x M3 x 18 mm on the outer arm screws | [GEPRC Mark4](https://geprc.com/product/gep-mark4-frame/) | 2026-10-18 |
| `frame.arm_standoff.diameter` | 5 mm | (osäkert) | Estimate, no drawing published | - |
| `frame.arm_standoff.bore` | 3.2 mm | M3 clearance | Design default | - |
| `frame.hole` | 3.2 mm | M3 clearance holes | Design default | - |
| `frame.stack_holes` | 30.5, 20 mm | 30.5 x 30.5 (M3) and 20 x 20 (M2) flight stack patterns | [GEPRC Mark4](https://geprc.com/product/gep-mark4-frame/) | 2026-10-18 |
| `motor.diameter` | 33.9 mm | bell | [EMAX ECO II 2807 drawing](https://cdn.shopify.com/s/files/1/0469/7358/3518/files/ECOII-2807Motor_specifications_200926_-01.jpg) | 2026-10-18 |
| `motor.height` | 20 mm | base to the top of the bell: 34 mm overall minus the 14 mm shaft | [EMAX ECO II 2807 drawing](https://cdn.shopify.com/s/files/1/0469/7358/3518/files/ECOII-2807Motor_specifications_200926_-01.jpg) | 2026-10-18 |
| `motor.base_height` | 6.6 mm | stator mount below the bell | [EMAX ECO II 2807 drawing](https://cdn.shopify.com/s/files/1/0469/7358/3518/files/ECOII-2807Motor_specifications_200926_-01.jpg) | 2026-10-18 |
| `motor.base_diameter` | 27 mm | (osäkert) stator mount | Estimate, no drawing published | - |
| `motor.shaft` | 5, 14 mm | M5 shaft: diameter, length above the bell | [EMAX ECO II 2807 drawing](https://cdn.shopify.com/s/files/1/0469/7358/3518/files/ECOII-2807Motor_specifications_200926_-01.jpg) | 2026-10-18 |
| `motor.nut` | 9.2, 5 mm | (osäkert) M5 nylon lock nut: across corners, height | Estimate, no drawing published | - |
| `motor.mount_circle` | 19 mm | 4 x M3 on a 19 mm circle | [EMAX ECO II 2807 drawing](https://cdn.shopify.com/s/files/1/0469/7358/3518/files/ECOII-2807Motor_specifications_200926_-01.jpg) | 2026-10-18 |
| `motor.mass` | 47.6 g | without leads (the leads are cables, `wiring.motor`) | [EMAX ECO II 2807 drawing](https://cdn.shopify.com/s/files/1/0469/7358/3518/files/ECOII-2807Motor_specifications_200926_-01.jpg) | 2026-10-18 |
| `prop.diameter_in` | 7 in |  | [HQProp 7x4x3 Durable](https://hqprop.com/hq-durable-prop-7x4x3-light-grey-2cw2ccw-poly-carbonate-popo-p0133.html) | 2026-10-18 |
| `prop.pitch_in` | 4 in |  | [HQProp 7x4x3 Durable](https://hqprop.com/hq-durable-prop-7x4x3-light-grey-2cw2ccw-poly-carbonate-popo-p0133.html) | 2026-10-18 |
| `prop.blades` | 3 |  | [HQProp 7x4x3 Durable](https://hqprop.com/hq-durable-prop-7x4x3-light-grey-2cw2ccw-poly-carbonate-popo-p0133.html) | 2026-10-18 |
| `prop.swept_height` | 10 mm | (osäkert) the disc the checks use; hub 7 mm, blade pitch adds height | Estimate, no drawing published | - |
| `prop.gap_above_motor` | 0 mm | the hub sits on the bell | Design default | - |
| `prop.hub` | 13.3, 7 mm | hub diameter and height | Retailer listings of the HQ 7x4x3 (GetFPV, RaceDayQuads, Pyrodrone) | 2026-10-18 |
| `prop.chord_max` | 19 mm | (osäkert) widest blade chord, drawing only | Estimate, no drawing published | - |
| `prop.bore` | 10 mm | the disc's hole: room for the shaft and nut | Design default | - |
| `prop.mass` | 9.7 g | retailers list 9.1-9.3 g | [HQProp 7x4x3 Durable](https://hqprop.com/hq-durable-prop-7x4x3-light-grey-2cw2ccw-poly-carbonate-popo-p0133.html) | 2026-10-18 |
| `stack.base_height` | 9.7 mm | (osäkert) ESC underside above the bottom plate: clears the 2.4 mm arm nuts | Design default | - |
| `stack.board_gap` | 1.8 mm | (osäkert) between ESC and FC; the FC stays under the top plate | Design default | - |
| `stack.esc.length` | 44 mm |  | [Rotorama, MicoAir 55A AM32](https://www.rotorama.com/product/micoair-am32-55a-4v1) (retailer) | 2026-10-18 |
| `stack.esc.width` | 43.5 mm |  | [Rotorama, MicoAir 55A AM32](https://www.rotorama.com/product/micoair-am32-55a-4v1) (retailer) | 2026-10-18 |
| `stack.esc.height` | 5.2 mm |  | [Rotorama, MicoAir 55A AM32](https://www.rotorama.com/product/micoair-am32-55a-4v1) (retailer) | 2026-10-18 |
| `stack.esc.mass` | 14 g |  | [Rotorama, MicoAir 55A AM32](https://www.rotorama.com/product/micoair-am32-55a-4v1) (retailer) | 2026-10-18 |
| `stack.fc.length` | 36 mm |  | [MicoAir743 V2](https://micoair.com/flightcontroller_micoair743v2/) | 2026-10-18 |
| `stack.fc.width` | 36 mm |  | [MicoAir743 V2](https://micoair.com/flightcontroller_micoair743v2/) | 2026-10-18 |
| `stack.fc.height` | 8 mm |  | [MicoAir743 V2](https://micoair.com/flightcontroller_micoair743v2/) | 2026-10-18 |
| `stack.fc.mass` | 10 g |  | [MicoAir743 V2](https://micoair.com/flightcontroller_micoair743v2/) | 2026-10-18 |
| `bec.length` | 35 mm |  | [Matek BEC12S-PRO](https://www.mateksys.com/?portfolio=bec12s-pro) | 2026-10-18 |
| `bec.width` | 24 mm |  | [Matek BEC12S-PRO](https://www.mateksys.com/?portfolio=bec12s-pro) | 2026-10-18 |
| `bec.height` | 5.5 mm |  | [Matek BEC12S-PRO](https://www.mateksys.com/?portfolio=bec12s-pro) | 2026-10-18 |
| `bec.mass` | 5 g |  | [Matek BEC12S-PRO](https://www.mateksys.com/?portfolio=bec12s-pro) | 2026-10-18 |
| `receiver.length` | 23.7 mm |  | [RadioMaster RP4TD](https://www.radiomasterrc.com/products/rp4td-expresslrs-2-4ghz-diversity-receiver) | 2026-10-18 |
| `receiver.width` | 16.3 mm |  | [RadioMaster RP4TD](https://www.radiomasterrc.com/products/rp4td-expresslrs-2-4ghz-diversity-receiver) | 2026-10-18 |
| `receiver.height` | 4 mm |  | [RadioMaster RP4TD](https://www.radiomasterrc.com/products/rp4td-expresslrs-2-4ghz-diversity-receiver) | 2026-10-18 |
| `receiver.mass` | 1.4 g | without the two 65 mm antennas (`antenna.mass`) | [RadioMaster RP4TD](https://www.radiomasterrc.com/products/rp4td-expresslrs-2-4ghz-diversity-receiver) | 2026-10-18 |
| `remote_id.length` | 23.2 mm |  | [ESP32-S3 Super Mini](https://circuit.rocks/products/esp32-s3-supermini-development-board) (retailer, generic board) | 2026-10-18 |
| `remote_id.width` | 18.2 mm |  | [ESP32-S3 Super Mini](https://circuit.rocks/products/esp32-s3-supermini-development-board) (retailer, generic board) | 2026-10-18 |
| `remote_id.height` | 4 mm | (osäkert) | Estimate, no drawing published | - |
| `remote_id.mass` | 2.7 g | with pin headers | [ESP32-S3 Super Mini](https://circuit.rocks/products/esp32-s3-supermini-development-board) (retailer, generic board) | 2026-10-18 |
| `gps.length` | 34 mm | with case | [Holybro Micro M10](https://holybro.com/products/micro-m10-gps) | 2026-10-18 |
| `gps.width` | 28 mm | with case | [Holybro Micro M10](https://holybro.com/products/micro-m10-gps) | 2026-10-18 |
| `gps.height` | 11 mm | with case | [Holybro Micro M10](https://holybro.com/products/micro-m10-gps) | 2026-10-18 |
| `gps.mass` | 16 g | with case | [Holybro Micro M10](https://holybro.com/products/micro-m10-gps) | 2026-10-18 |
| `gps.mast.diameter` | 10 mm | (osäkert) printed mast | Design default | - |
| `gps.mast.height` | 60 mm | (osäkert) GPS away from the electronics | Design default | - |
| `gps.mast.mass` | 6 g | (osäkert) | Design default | - |
| `gps.mast.foot` | 22, 3 mm | (osäkert) square foot: side, thickness | Design default | - |
| `battery.length` | 145 mm | another retailer lists 150 x 65 x 45 mm | [Rotorama, Dogcom 10 000 mAh](https://www.rotorama.com/product/dogcom-liion-10000mah-6s2p-25c-21700) (retailer) | 2026-10-18 |
| `battery.width` | 64 mm |  | [Rotorama, Dogcom 10 000 mAh](https://www.rotorama.com/product/dogcom-liion-10000mah-6s2p-25c-21700) (retailer) | 2026-10-18 |
| `battery.height` | 43 mm |  | [Rotorama, Dogcom 10 000 mAh](https://www.rotorama.com/product/dogcom-liion-10000mah-6s2p-25c-21700) (retailer) | 2026-10-18 |
| `battery.mass` | 913 g |  | [Rotorama, Dogcom 10 000 mAh](https://www.rotorama.com/product/dogcom-liion-10000mah-6s2p-25c-21700) (retailer) | 2026-10-18 |
| `pi5.length` | 85 mm |  | [Raspberry Pi 5 mechanical drawing](https://pip-assets.raspberrypi.com/categories/892-raspberry-pi-5/documents/RP-008347-DS-1-raspberry-pi-5-mechanical-drawing.pdf) | 2026-10-18 |
| `pi5.width` | 56 mm |  | [Raspberry Pi 5 mechanical drawing](https://pip-assets.raspberrypi.com/categories/892-raspberry-pi-5/documents/RP-008347-DS-1-raspberry-pi-5-mechanical-drawing.pdf) | 2026-10-18 |
| `pi5.height` | 1.6 mm | (osäkert) board thickness | Estimate, no drawing published | - |
| `pi5.underside_clearance` | 3 mm | (osäkert) parts under the board | Design default | - |
| `pi5.mass` | 46 g | (osäkert) not in the drawing or product brief | Estimate, no drawing published | - |
| `pi5.ports.length` | 21 mm | (osäkert) USB and Ethernet block | Estimate, no drawing published | - |
| `pi5.ports.width` | 53 mm | (osäkert) | Estimate, no drawing published | - |
| `pi5.ports.height` | 16 mm | (osäkert) | Estimate, no drawing published | - |
| `pi5.ports.overhang` | 3 mm | past the board edge | [Raspberry Pi 5 mechanical drawing](https://pip-assets.raspberrypi.com/categories/892-raspberry-pi-5/documents/RP-008347-DS-1-raspberry-pi-5-mechanical-drawing.pdf) | 2026-10-18 |
| `pi5.ports.mass` | 0 g | included in pi5.mass | Design default | - |
| `cooler.length` | 63.5 mm |  | [Active Cooler product brief](https://datasheets.raspberrypi.com/cooling/raspberry-pi-active-cooler-product-brief.pdf) | 2026-10-18 |
| `cooler.width` | 42.5 mm |  | [Active Cooler product brief](https://datasheets.raspberrypi.com/cooling/raspberry-pi-active-cooler-product-brief.pdf) | 2026-10-18 |
| `cooler.height` | 13.7 mm |  | [Active Cooler product brief](https://datasheets.raspberrypi.com/cooling/raspberry-pi-active-cooler-product-brief.pdf) | 2026-10-18 |
| `cooler.offset_from_gpio_end` | 33 mm | (osäkert) cooler centre from the GPIO end | Estimate, no drawing published | - |
| `cooler.mass` | 29 g | (osäkert) | [Think Robotics, Active Cooler](https://thinkrobotics.com/products/raspberry-pi-active-cooler-for-raspberry-pi-5) (retailer) | 2026-10-18 |
| `ai_hat.length` | 65 mm |  | [AI HAT+ 2 product brief](https://pip.raspberrypi.com/documents/RP-009655-MM-raspberry-pi-ai-hat-plus-2-product-brief.pdf) | 2026-10-18 |
| `ai_hat.width` | 56.5 mm |  | [AI HAT+ 2 product brief](https://pip.raspberrypi.com/documents/RP-009655-MM-raspberry-pi-ai-hat-plus-2-product-brief.pdf) | 2026-10-18 |
| `ai_hat.height` | 15.6 mm | (osäkert) 1.6 mm board (estimate) + 14 mm heatsink | [AI HAT+ 2 product brief](https://pip.raspberrypi.com/documents/RP-009655-MM-raspberry-pi-ai-hat-plus-2-product-brief.pdf) | 2026-10-18 |
| `ai_hat.standoff_height` | 16 mm | 16 mm stacking header | [AI HAT+ 2 product brief](https://pip.raspberrypi.com/documents/RP-009655-MM-raspberry-pi-ai-hat-plus-2-product-brief.pdf) | 2026-10-18 |
| `ai_hat.mass` | 45 g | (osäkert) not published; board with heatsink | Estimate, no drawing published | - |
| `pi_tray.length` | 86 mm | (osäkert) printed tray, the size of the board | Design default | - |
| `pi_tray.width` | 63 mm | (osäkert) printed tray, room for the canopy screw bosses | Design default | - |
| `pi_tray.height` | 3 mm | (osäkert) | Design default | - |
| `pi_tray.damper_height` | 8 mm | (osäkert) rubber dampers | Design default | - |
| `pi_tray.mass` | 20 g | (osäkert) | Design default | - |
| `pi_tray.boss` | 7, 3, 9 mm | (osäkert) canopy screw boss: length, thickness, height; 3 mm hole for an M2 heat-set insert | Design default | - |
| `camera.length` | 25 mm |  | [Camera Module 3 product brief](https://datasheets.raspberrypi.com/camera/camera-module-3-product-brief.pdf) | 2026-10-18 |
| `camera.width` | 24 mm |  | [Camera Module 3 product brief](https://datasheets.raspberrypi.com/camera/camera-module-3-product-brief.pdf) | 2026-10-18 |
| `camera.height` | 11.5 mm | with the lens | [Camera Module 3 product brief](https://datasheets.raspberrypi.com/camera/camera-module-3-product-brief.pdf) | 2026-10-18 |
| `camera.mass` | 4 g | (osäkert) | [Core Electronics, Camera Module 3](https://core-electronics.com.au/raspberry-pi-camera-3.html) (retailer) | 2026-10-18 |
| `camera.hfov_deg` | 66 deg | standard lens | [Camera Module 3 product brief](https://datasheets.raspberrypi.com/camera/camera-module-3-product-brief.pdf) | 2026-10-18 |
| `camera.vfov_deg` | 41 deg |  | [Camera Module 3 product brief](https://datasheets.raspberrypi.com/camera/camera-module-3-product-brief.pdf) | 2026-10-18 |
| `camera_mount.length` | 30 mm | (osäkert) printed | Design default | - |
| `camera_mount.width` | 30 mm | (osäkert) | Design default | - |
| `camera_mount.height` | 6 mm | (osäkert) | Design default | - |
| `camera_mount.mass` | 4 g | (osäkert) | Design default | - |
| `legs.length` | 20 mm | (osäkert) printed (TPU) | Design default | - |
| `legs.width` | 10 mm | (osäkert) | Design default | - |
| `legs.from_motor` | 40 mm | (osäkert) leg centre this far from the motor along the arm | Design default | - |
| `legs.ground_clearance` | 10 mm | (osäkert) below the battery | Design default | - |
| `legs.mass` | 6 g | (osäkert) each | Design default | - |
| `canopy.wall` | 1.2 mm | printed wall | Design default | - |
| `canopy.edge_radius` | 12 mm |  | Design default | - |
| `canopy.x` | 0 mm | centre in x | Design default | - |
| `canopy.rim_clearance` | 6 mm | rim above the propeller discs | Design default | - |
| `canopy.intake` | 24, 8 mm | nose slot, width and height | Design default | - |
| `canopy.vent` | 46, 34 mm | roof opening over the cooler and AI HAT | Design default | - |
| `canopy.exhaust` | 22, 7 mm | tail slot, width and height | Design default | - |
| `canopy.mast_clearance` | 3 mm | round the GPS mast: grommet and GPS lead | Design default | - |
| `canopy.prop_keep_out` | 4 mm | seen from above, outside the propeller discs | Design default | - |
| `canopy.screws.x` | -26, 34 mm | x of the 2 x 2 M2 side screws | Design default | - |
| `canopy.screws.length` | 4 mm | M2 x 4 button head into a heat-set insert | Design default | - |
| `nose_fairing.wall` | 1.2 mm | printed wall | Design default | - |
| `nose_fairing.length` | 62 mm | tip to open rear end | Design default | - |
| `nose_fairing.width` | 76 mm |  | Design default | - |
| `nose_fairing.height` | 47 mm |  | Design default | - |
| `nose_fairing.nose_radius` | 25 mm |  | Design default | - |
| `nose_fairing.edge_radius` | 10 mm |  | Design default | - |
| `nose_fairing.rear_trim` | 20 mm | modelling aid: rounded rear end cut off | Design default | - |
| `nose_fairing.x_tip` | 117 mm |  | Design default | - |
| `nose_fairing.window_extra_deg` | 4 deg | camera window margin | Design default | - |
| `materials.PETG` | 1.27 g/cm^3 | ISO 1183 | [Prusament PETG technical data sheet](https://storage.googleapis.com/prusa3d-content-prod-14e8-wordpress-prusament-prod/2023/10/9f8d2165-tds_prusament-petg_n_en.pdf) | 2026-10-18 |
| `materials.ASA` | 1.07 g/cm^3 | ISO 1183 | [Prusament ASA technical data sheet](https://storage.googleapis.com/prusa3d-content-prod-14e8-wordpress-prusament-prod/2024/05/bc8551ef-tds_prusament-asa_2024_en.pdf) | 2026-10-18 |
| `materials.carbon` | 1.55 g/cm^3 | (osäkert) 3K carbon plate, typical | Estimate, no drawing published | - |
| `materials.aluminium` | 2.7 g/cm^3 | 6061 | Handbook value (ASM) | - |
| `materials.steel` | 7.85 g/cm^3 | screws and nuts | Handbook value | - |
| `aero.speed` | 10 m/s | search speed | Design default | - |
| `aero.air_density` | 1.225 kg/m^3 | ISA sea level, 15 C | Design default | - |
| `aero.kinematic_viscosity` | 1.5e-05 m^2/s | air at 15 C | ISA | - |
| `aero.pitch_deg` | 0 deg | nose-down attitude in the drag estimate; level unless set | Design default | - |
| `aero.cd_bluff` | 1.05 | (osäkert) box with sharp edges (cube 1.05) | Hoerner, *Fluid-Dynamic Drag* (1965), textbook value | - |
| `aero.cd_faired` | 0.5 | (osäkert) rounded nose, open rear | Hoerner, *Fluid-Dynamic Drag* (1965), textbook value | - |
| `aero.cd_cylinder` | 0.9 | (osäkert) short cylinder across the flow (long: 1.0-1.2) | Hoerner, *Fluid-Dynamic Drag* (1965), textbook value | - |
| `aero.cd_plate` | 1.6 | (osäkert) thin rectangular section edge-on to the flow | Hoerner, *Fluid-Dynamic Drag* (1965), textbook value | - |
| `aero.wake_factor` | 0.35 | (osäkert) share of drag left on a body in another's near wake (tandem bodies) | Hoerner, *Fluid-Dynamic Drag* (1965), textbook value | - |
| `aero.wake_length` | 150 mm | (osäkert) about ten body widths | Hoerner, *Fluid-Dynamic Drag* (1965), textbook value | - |
| `aero.reference_cda_m2` | 0.018, 0.023 m^2 | D/q of bare quadcopter bodies (Iris+, Phantom 3, SOLO) at 12 m/s | [Russell et al., NASA Ames 2016](https://rotorcraft.arc.nasa.gov/Publications/files/72-2016-374.pdf) | 2026-10-18 |
| `misc.mass` | 15 g | (osäkert) solder, heat-shrink, zip ties, tape, soft mounts | Estimate, no drawing published | - |
| `screws.m3_button` | 5.7, 1.65 mm | head diameter and height | ISO 7380-1 | - |
| `screws.m3_flat` | 6.72, 1.86 mm | head diameter and height | ISO 10642 | - |
| `screws.m2_button` | 3.5, 1.3 mm | head diameter and height | ISO 7380-1 | - |
| `screws.m3_nut` | 6, 2.4 mm | across corners and height; thin enough to stay under the ESC | ISO 4032 | - |
| `screws.motor_length` | 8 mm | (osäkert) through the 5 mm arm, 3 mm into the motor | Design default | - |
| `wiring.motor.diameter` | 2.3 mm | 18 AWG silicone lead | [EMAX ECO II 2807 drawing](https://cdn.shopify.com/s/files/1/0469/7358/3518/files/ECOII-2807Motor_specifications_200926_-01.jpg) | 2026-10-18 |
| `wiring.motor.count` | 3 | three per motor | [EMAX ECO II 2807 drawing](https://cdn.shopify.com/s/files/1/0469/7358/3518/files/ECOII-2807Motor_specifications_200926_-01.jpg) | 2026-10-18 |
| `wiring.motor.g_per_m` | 10.5 g/m | (osäkert) 18 AWG silicone | Estimate, no drawing published | - |
| `wiring.battery.diameter` | 4.5 mm | (osäkert) 12 AWG silicone, outer diameter | Estimate, no drawing published | - |
| `wiring.battery.count` | 2 | plus and minus | Design default | - |
| `wiring.battery.g_per_m` | 44 g/m | (osäkert) 12 AWG silicone | Estimate, no drawing published | - |
| `wiring.power.diameter` | 1.9 mm | (osäkert) 20 AWG silicone | Estimate, no drawing published | - |
| `wiring.power.count` | 2 | plus and minus | Design default | - |
| `wiring.power.g_per_m` | 7.5 g/m | (osäkert) 20 AWG silicone | Estimate, no drawing published | - |
| `wiring.gps.diameter` | 2.5 mm | (osäkert) JST-GH 6-pin lead as one bundle | Estimate, no drawing published | - |
| `wiring.gps.count` | 1 |  | Design default | - |
| `wiring.gps.g_per_m` | 8 g/m | (osäkert) | Estimate, no drawing published | - |
| `wiring.ribbon.width` | 16 mm | Camera Module 3 FFC, 15 to 22 pin | [Camera Module 3 product brief](https://datasheets.raspberrypi.com/camera/camera-module-3-product-brief.pdf) | 2026-10-18 |
| `wiring.ribbon.thickness` | 0.3 mm | (osäkert) | Estimate, no drawing published | - |
| `wiring.ribbon.g_per_m` | 6 g/m | (osäkert) | Estimate, no drawing published | - |
| `xt60.length` | 23.8 mm | XT60H-M, one half | [AMASS XT60H-M datasheet](https://cdn.eicom.ru/media/PDF/XT60H-M.pdf) (mirror) | 2026-10-18 |
| `xt60.width` | 9.2 mm |  | [AMASS XT60H-M datasheet](https://cdn.eicom.ru/media/PDF/XT60H-M.pdf) (mirror) | 2026-10-18 |
| `xt60.height` | 16.4 mm |  | [AMASS XT60H-M datasheet](https://cdn.eicom.ru/media/PDF/XT60H-M.pdf) (mirror) | 2026-10-18 |
| `xt60.mass` | 7.7 g | (osäkert) the mated pair with its shells | Pyrodrone listing (retailer) | 2026-10-18 |
| `strap.width` | 20 mm | Lumenier 20 x 250 mm strap | GetFPV listing (retailer) | 2026-10-18 |
| `strap.thickness` | 1.5 mm | (osäkert) | Estimate, no drawing published | - |
| `strap.mass` | 7.5 g | each | GetFPV listing (retailer) | 2026-10-18 |
| `strap.x` | -40, 40 mm | two straps, x of each | Design default | - |
| `antenna.length` | 65 mm | 2 x 65 mm T antennas | [RadioMaster RP4TD manual](https://cdn.shopify.com/s/files/1/0609/8324/7079/files/RP4TD.pdf) | 2026-10-18 |
| `antenna.diameter` | 1.5 mm | (osäkert) coax | Estimate, no drawing published | - |
| `antenna.element` | 30, 3 mm | (osäkert) active element: length, diameter of its sleeve | Estimate, no drawing published | - |
| `antenna.mass` | 0.3 g | (osäkert) each | Estimate, no drawing published | - |
| `antenna.spread_deg` | 30 deg | V out of the rear, each side | Design default | - |
| `antenna.droop_deg` | 30 deg | pointing down, away from the carbon | Design default | - |
| `layout.stack_x` | 0 mm |  | Design default | - |
| `layout.bec` | 64, 0 mm | [x, y], between the plates | Design default | - |
| `layout.receiver` | -62, 0 mm | [x, y], between the plates | Design default | - |
| `layout.remote_id` | 62, 0 mm | [x, y], on the top plate | Design default | - |
| `layout.pi_x` | 4 mm |  | Design default | - |
| `layout.gps_x` | -62 mm | mast at the rear | Design default | - |
| `layout.battery_x` | 0 mm | balances the centre of gravity | Design default | - |
| `layout.camera_x` | 92 mm | ahead of the battery | Design default | - |
| `layout.xt60` | -100, -10 mm | [x, z] of the mated pair behind the battery | Design default | - |
| `reference.takeoff_mass_g` | 1500 g | (osäkert) planning target | Estimate, no drawing published | - |
| `reference.inertia_g_m2` | 5.5, 6.5, 9 g m^2 | (osäkert) hand estimate with point masses, made before purchase | Estimate, no drawing published | - |
| `checks.prop_margin_min` | 5 mm | acceptance limit | Design default | - |
| `checks.plan_margin_min` | 3 mm | seen from above, parts above the discs stay outside them | Design default | - |
| `checks.cog_offset_max` | 5 mm | acceptance limit, x and y | Design default | - |
| `checks.cover_mass_max` | 80 mm | canopy and nose fairing together | Design default | - |
| `checks.fov_length` | 500 mm | how far below the camera the view is checked | Design default | - |
| `checks.stl_tolerance` | 0.05 mm | STL chord tolerance | Design default | - |
