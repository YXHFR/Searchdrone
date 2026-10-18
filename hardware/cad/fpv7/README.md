# fpv7: baseline CAD model of the 7-inch quadcopter

A parametric model of the 7-inch quadcopter, written as code with
[CadQuery](https://cadquery.readthedocs.io). Every size and mass is a number in
[`params.yaml`](params.yaml); change a number and rebuild. The parts are simplified
envelopes (boxes and cylinders), not detailed meshes. The model shows whether everything
fits, what the drone weighs and where its centre of gravity is, and it is the starting
point for the simulator model and for 3D-printed mounts.

```bash
just cad            # STEP, STL, report and picture in build/cad/fpv7/ (not committed)
just cad-detailed   # the same, plus fpv7_detailed.step with the manufacturers' 3D models
just cad-test       # overlaps, propeller margins, camera view, repeatable exports
```

| File | What it does |
|---|---|
| `params.yaml` | Every size, mass and position (millimetres, grams, degrees) |
| `parts.py` | One simplified solid per kind of part |
| `assembly.py` | Puts the parts in place and writes the exports (`just cad`) |
| `checks.py` | Mass, centre of gravity, inertia, overlaps, propeller margins, camera view |
| `render.py` | The picture from three angles (`fpv7_views.png`) |
| `vendor.py`, `vendor.yaml` | Download, check and place the manufacturers' 3D models (`just cad-detailed`) |
| `tests/` | The acceptance checks (`just cad-test`) |

Output in `build/cad/fpv7/`:
- `fpv7.step`: the whole assembly with one named solid per part, to open in FreeCAD or
  Onshape;
- `step/<part>.step`: every part on its own, in place in the drone's coordinates;
- `fpv7.stl` and `stl/<part>.stl`: the printed parts (`canopy.stl`, `nose_fairing.stl`,
  `pi_tray.stl`, `camera_mount.stl`, `leg_*.stl`, `gps_mast.stl`) go to the slicer;
- `report.md` / `report.json` and `fpv7_views.png`;
- `build/cad/fpv7-detailed/fpv7_detailed.step` (only with `just cad-detailed`), kept
  outside `build/cad/fpv7/` because it holds manufacturer models.

Two builds give byte-identical STEP, STL and report files.

## Axes and layout

x forward, y left, z up. The origin is the frame centre; z = 0 is the underside of the
bottom plate. Motors 1-4 sit at 45, 135, 225 and 315 degrees.

- **Frame:** GEPRC Mark4 7" (295 mm motor to motor, 5 mm arms). The frame's 121 g is
  spread over the plates, arms and standoffs by volume.
- **Between the plates:** the flight stack (MicoAir743 V2 on the 55 A AM32 4-in-1 ESC) on
  soft mounts, the 5 V BEC in front of it and the ELRS receiver behind it.
- **On the top plate:** the Raspberry Pi 5 with Active Cooler and AI HAT+ 2 on a printed
  tray with rubber dampers, the Remote ID board at the front and the GPS on a mast at the
  rear.
- **Under the bottom plate:** the 6S2P battery, and ahead of it the camera looking
  straight down, with the long side of the image across the drone.
- **Legs:** four printed legs under the arms that keep the battery 10 mm off the ground.
- **Propellers:** HQ 7x4x3 as discs of 177.8 mm, 10 mm high, standing on the motor bells.

- **Covers:** a printed top canopy over the companion computer and a printed nose under
  the frame ahead of the battery (see below).

Not modelled yet: receiver antennas, cables (their mass is a point mass, `misc.mass`),
screws, a 4G modem and a stand-alone Remote ID beacon.

## Printed covers

Both covers print in PETG or ASA (`material`) with a `wall` of 1.2 mm; their mass is
their volume times the material's density. Together they must stay under
`checks.cover_mass_max` (80 g).

- **Top canopy** (`canopy.stl`): a shell with a rounded nose over the Raspberry Pi, the
  Active Cooler and the AI HAT. Its rim stands `canopy.rim_clearance` above the propeller
  discs, and seen from above it stays `canopy.prop_keep_out` outside them: each corner has
  a scallop cut around the propeller disc. Openings: an intake slot in the nose, a roof
  vent over the cooler fan and the AI HAT heatsink, an exhaust slot in the tail and a hole
  for the GPS mast with room for a rubber grommet. It is meant to be fastened to the Pi
  tray with clips or M2 screws; the fastening is not modelled yet. Check the Pi's and the
  Hailo's temperatures with the canopy on during the first bench test.
- **Nose fairing** (`nose_fairing.stl`): a rounded nose hanging under the bottom plate in
  front of the battery, around the camera. The camera looks through a window cut from
  the field of view plus `nose_fairing.window_extra_deg` on every side. The fairing ends
  ahead of the battery strap and the battery's XT60 lead, which stay free.

The report gives the frontal area and a rough drag estimate (Cd A and drag at
`aero.speed`) with and without the covers. Without covers every part counts as a bluff
body (`aero.cd_bluff`); with them the area hidden behind the covers counts at
`aero.cd_faired`. Both coefficients are **(osäkert)** and are to be calibrated against
flight logs. Propeller discs are left out.

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
| GEPRC Mark4 7", EMAX ECO II 2807, MicoAir743 V2, 55 A ESC, Matek BEC12S-PRO, RadioMaster RP4TD | none published | - | Drawn from published sizes; frame sizes marked (osäkert) are to be measured | 2026-10-18 |

## Checks

`checks.py` treats every part as a solid of uniform density with the mass from
`params.yaml`. Inertia is about the centre of gravity in the body axes, in kg m^2.

1. No two parts share volume.
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
7. With covers, Cd A and drag are lower than without (rough estimate, see above); the
   frontal area and drag formulas are tested against a hand calculation.
8. The STEP, STL and report files are the same on every build.
9. After `just cad-detailed`, every manufacturer model fits its envelope within 2 mm.
10. The download checks in `vendor.py` (https only, also after redirects, pinned SHA-256,
    size limits, only the named file extracted) are tested offline.

## Sources

Read on the date given. **(osäkert)** marks a value that is uncertain: a value no
manufacturer publishes, an estimate, or a design default that may change. Retailer data
is used only where the manufacturer publishes none.

| Value | Number | Status and note | Source | Date |
|---|---|---|---|---|
| `frame.wheelbase` | 295 mm | motor to motor | [GEPRC Mark4](https://geprc.com/product/gep-mark4-frame/) | 2026-10-18 |
| `frame.mass` | 121 g | 7" kit; the shop field "150 g" is the shipping weight | [GEPRC Mark4](https://geprc.com/product/gep-mark4-frame/) | 2026-10-18 |
| `frame.bottom_plate.length` | 190 mm | (osäkert) | Estimate, no drawing published | - |
| `frame.bottom_plate.width` | 50 mm | (osäkert) | Estimate, no drawing published | - |
| `frame.bottom_plate.thickness` | 2.5 mm |  | [GEPRC Mark4](https://geprc.com/product/gep-mark4-frame/) | 2026-10-18 |
| `frame.bottom_plate.corner_radius` | 8 mm | (osäkert) outline simplified to a rounded rectangle | Estimate, no drawing published | - |
| `frame.top_plate.length` | 140 mm | (osäkert) | Estimate, no drawing published | - |
| `frame.top_plate.width` | 40 mm | (osäkert) | Estimate, no drawing published | - |
| `frame.top_plate.thickness` | 2.5 mm |  | [GEPRC Mark4](https://geprc.com/product/gep-mark4-frame/) | 2026-10-18 |
| `frame.top_plate.corner_radius` | 10 mm | (osäkert) | Estimate, no drawing published | - |
| `frame.arm.thickness` | 5 mm |  | [GEPRC Mark4](https://geprc.com/product/gep-mark4-frame/) | 2026-10-18 |
| `frame.arm.root_width` | 28 mm | (osäkert) | Estimate, no drawing published | - |
| `frame.arm.tip_width` | 30 mm | (osäkert) about 30 mm at the motor | Estimate, no drawing published | - |
| `frame.arm.root_offset` | 30 mm | (osäkert) where the arm root starts | Estimate, no drawing published | - |
| `frame.arm.tip_overhang` | 15 mm | (osäkert) arm past the motor axis | Estimate, no drawing published | - |
| `frame.standoff.height` | 25 mm | M3 x 25 mm | [GEPRC Mark4](https://geprc.com/product/gep-mark4-frame/) | 2026-10-18 |
| `frame.standoff.diameter` | 5 mm | (osäkert) | Estimate, no drawing published | - |
| `frame.standoff.pattern_x` | 120 mm | (osäkert) | Estimate, no drawing published | - |
| `frame.standoff.pattern_y` | 34 mm | (osäkert) | Estimate, no drawing published | - |
| `motor.diameter` | 33.9 mm | bell | [EMAX ECO II 2807 drawing](https://shop.emaxmodel.com/products/emax-eco-ii-series-2807-1300kv-1700kv-1500kv-brushless-motor-for-rc-drone-fpv-racing) | 2026-10-18 |
| `motor.height` | 20 mm | (osäkert) 34 mm overall minus 14 mm shaft above the bell | [EMAX ECO II 2807 drawing](https://shop.emaxmodel.com/products/emax-eco-ii-series-2807-1300kv-1700kv-1500kv-brushless-motor-for-rc-drone-fpv-racing) | 2026-10-18 |
| `motor.mass` | 47.6 g | without leads (leads are in misc.mass) | [EMAX ECO II 2807 drawing](https://shop.emaxmodel.com/products/emax-eco-ii-series-2807-1300kv-1700kv-1500kv-brushless-motor-for-rc-drone-fpv-racing) | 2026-10-18 |
| `prop.diameter_in` | 7 in |  | [HQProp 7x4x3 Durable](https://hqprop.com/hq-durable-prop-7x4x3-light-grey-2cw2ccw-poly-carbonate-popo-p0133.html) | 2026-10-18 |
| `prop.swept_height` | 10 mm | (osäkert) hub is 6.8 mm thick; blade pitch adds height | Estimate, no drawing published | - |
| `prop.gap_above_motor` | 0 mm | the hub sits on the bell | Design default | - |
| `prop.mass` | 9.7 g | retailers list 9.1-9.3 g | [HQProp 7x4x3 Durable](https://hqprop.com/hq-durable-prop-7x4x3-light-grey-2cw2ccw-poly-carbonate-popo-p0133.html) | 2026-10-18 |
| `stack.grommet_height` | 3 mm | (osäkert) soft mounts | Design default | - |
| `stack.board_gap` | 4 mm | (osäkert) between ESC and FC | Design default | - |
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
| `receiver.mass` | 1.4 g | without the two 65 mm antennas | [RadioMaster RP4TD](https://www.radiomasterrc.com/products/rp4td-expresslrs-2-4ghz-diversity-receiver) | 2026-10-18 |
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
| `pi_tray.width` | 56 mm | (osäkert) | Design default | - |
| `pi_tray.height` | 3 mm | (osäkert) | Design default | - |
| `pi_tray.damper_height` | 8 mm | (osäkert) rubber dampers | Design default | - |
| `pi_tray.mass` | 20 g | (osäkert) | Design default | - |
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
| `legs.radius` | 110 mm | (osäkert) from the centre, under each arm | Design default | - |
| `legs.ground_clearance` | 10 mm | (osäkert) below the battery | Design default | - |
| `legs.mass` | 6 g | (osäkert) each | Design default | - |
| `canopy.wall` | 1.2 mm | printed wall | Design default | - |
| `canopy.length` | 144 mm |  | Design default | - |
| `canopy.width` | 66 mm |  | Design default | - |
| `canopy.height` | 44 mm |  | Design default | - |
| `canopy.nose_radius` | 28 mm | rounded nose | Design default | - |
| `canopy.edge_radius` | 12 mm |  | Design default | - |
| `canopy.x` | 0 mm | centre in x | Design default | - |
| `canopy.rim_clearance` | 6 mm | rim above the propeller discs | Design default | - |
| `canopy.intake` | 30, 10 mm | nose slot, width and height | Design default | - |
| `canopy.vent` | 46, 34 mm | roof opening over the cooler and AI HAT | Design default | - |
| `canopy.exhaust` | 30, 10 mm | tail slot | Design default | - |
| `canopy.mast_clearance` | 2.5 mm | around the GPS mast, room for a rubber grommet | Design default | - |
| `canopy.prop_keep_out` | 4 mm | seen from above, outside the propeller discs | Design default | - |
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
| `aero.speed` | 10 m/s | search speed | Design default | - |
| `aero.air_density` | 1.225 kg/m^3 | ISA sea level, 15 C | Design default | - |
| `aero.cd_bluff` | 1.05  | (osäkert) box-like bluff body | Rough textbook value, to be calibrated against flight logs | - |
| `aero.cd_faired` | 0.5  | (osäkert) behind a rounded nose | Rough textbook value, to be calibrated against flight logs | - |
| `misc.mass` | 80 g | (osäkert) wiring, XT60, screws, battery strap | Estimate, no drawing published | - |
| `layout.stack_x` | 0 mm |  | Design default | - |
| `layout.bec` | 55, 0 mm | [x, y], between the plates | Design default | - |
| `layout.receiver` | -62, 0 mm | [x, y], between the plates | Design default | - |
| `layout.remote_id` | 62, 0 mm | [x, y], on the top plate | Design default | - |
| `layout.pi_x` | 4 mm |  | Design default | - |
| `layout.gps_x` | -62 mm | mast at the rear | Design default | - |
| `layout.battery_x` | 0 mm | balances the centre of gravity | Design default | - |
| `layout.camera_x` | 92 mm | ahead of the battery | Design default | - |
| `reference.takeoff_mass_g` | 1500 g | (osäkert) planning target | Estimate, no drawing published | - |
| `reference.inertia_g_m2` | 5.5, 6.5, 9.0 g m^2 | (osäkert) hand estimate with point masses, made before purchase | Estimate, no drawing published | - |
| `checks.prop_margin_min` | 5 mm | acceptance limit | Design default | - |
| `checks.plan_margin_min` | 3 mm | seen from above, parts above the discs stay outside them | Design default | - |
| `checks.cog_offset_max` | 5 mm | acceptance limit, x and y | Design default | - |
| `checks.cover_mass_max` | 80 g | canopy and nose fairing together | Design default | - |
| `checks.fov_length` | 500 mm | how far below the camera the view is checked | Design default | - |
| `checks.stl_tolerance` | 0.05 mm | STL chord tolerance | Design default | - |
