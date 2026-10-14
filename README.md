# Searchdrone

An open-source autonomous search-and-rescue drone: it flies a search pattern over an area on its own, finds a person with its camera, and follows them, while a ground operator watches and stays in control from a web control panel.

**Status:** early planning. No code yet.

## The idea in one picture

```
Camera ──► Companion computer (Raspberry Pi class)
           ├─ Detection   – find people in the image (YOLO)
           ├─ Mission     – search / confirm / follow / lost
           └─ Ground link
                 │ MAVLink
                 ▼
           Flight controller (ArduPilot)
           – stabilisation, navigation, failsafes

Ground: web control panel – map, live video, detections, start / abort / return home
```

The flight controller always keeps the drone safe on its own. The companion computer only sends high-level commands ("go here", "move at this velocity"), so a crash in the smart parts never brings the drone down.

Everything is developed in simulation first (ArduPilot SITL + a 3D simulator) before it touches real hardware.
