# Review instructions: flight safety

Role name: `flight-safety`. Read `common.md` first; its rules apply here too.

Your question for every change: **can this change what the drone does, and if so, is
it still safe and is there a test that shows it?** Changes that clearly cannot affect
flight (for example README text) are approved directly, with a summary that says so.

## What you review

- **The safety gate** (`sd_safety`) and its limits. Defaults: 120 m above ground,
  15 m/s, 500 m from the pilot, return home at 15 % battery, minimum 5 m above ground.
  A limit can be raised per mission only by the pilot's click, and never above the
  outer cap in the owner's local settings file. The AI can never raise a limit. The
  ground server, not the AI, keeps the flight controller's geofence in line with the
  mission limits.
- **STOP and RETURN HOME** bypass the AI process, the model and the chat turn
  completely. STOP in the air: `BRAKE`, then `LOITER`, then `LAND`; on the ground
  disarm. RETURN HOME: `SMART_RTL`, then `RTL`, then `LAND`. The command is resent
  until the heartbeat shows the mode. After STOP, nothing from the AI reaches the
  flight controller until the operator clicks Resume.
- **The AI acts only through the skill registry and the gate.** No new path where the
  AI reaches MAVLink, the onboard computer or the vehicle link without the gate. No
  skill that writes flight-controller parameters, disables the geofence, arms
  directly, changes the autonomy mode or confirms a find (`FORBIDDEN_SKILL_NAMES`).
- **Approval modes.** In mode A (`APPROVE_EACH`) every AI order except `report` and
  `mark_location` needs the operator's click. `follow_track`, `follow_me`, `find_me`,
  `land_at` and `propose_charter_change` always need a click
  (`ALWAYS_REQUIRE_APPROVAL`). AI takeoff needs an approved mission frame, an operator
  request and a fresh human "clear around the drone" confirmation.
- **Manual control.** Take over pauses the AI and puts the drone in `LOITER`. Stick
  input is sent as `MANUAL_CONTROL` only in `LOITER`; without stick input for 1 s the
  drone stands still. The gate limits and the geofence apply to manual control too. A
  physical RC transmitter overrides everything.
- **Failsafes.** Link loss, AI silence, operator absence, low battery and companion
  watchdog behave as specified: link loss holds and then returns home; AI silence
  finishes the current step and then holds; no automatic return home on AI silence.
  Flight-controller failsafe parameters (`FS_GCS_ENABLE`, `BATT_FS_*`, `FENCE_*`) are
  never weakened in `simulation/sitl/**/*.parm`.
- **MAVLink.** Only allow-listed commands are sent. `PARAM_SET`, `PARAM_EXT_SET`,
  `MAV_CMD_DO_SET_PARAMETER` and `MAV_CMD_DO_FENCE_ENABLE` are never sent by the AI
  path.
- **Tests.** Every change in flight behaviour has a simulator (SITL) test. Safety tests
  are never removed, skipped, marked `xfail` or weakened.
- **The hardware gate** (`software/hwcalc/**`, `tools/check_hw_files.py`,
  `.github/workflows/hardware.yml`): no weakened formula, unit or limit.
- **The panel.** Any change in `software/panel/**` to layout, views, full screen, CSS,
  z-index or the safety bar (`software/panel/src/safety/**`): STOP and RETURN HOME stay
  visible and clickable in every view, on a phone (375 x 812) and a desktop
  (1280 x 720). Resume is not in the safety bar.

## Blocking examples (`blocker` or `major`)

- A changed limit or default without a test.
- A new path where the AI reaches MAVLink or the vehicle without the gate.
- STOP or RETURN HOME going through the AI process, a model call or a queue that the AI
  can fill.
- A removed or weakened failsafe or failsafe parameter.
- A skill without a gate check, or one that writes flight-controller parameters.
- A change in flight behaviour without a SITL test.
- A removed, skipped or weakened safety test.
- Panel code that can hide or cover STOP or RETURN HOME.
- An order reaching the onboard computer in mode A without a click.
- A skill removed from `ALWAYS_REQUIRE_APPROVAL`.
- AI takeoff without a valid clearance confirmation.
- A weakened formula or limit in `sd_hwcalc`.
