# Panel (`@searchdrone/panel`)

The web control panel: chat with the AI commander, map, live video, candidate queue, status
and manual control. STOP and RETURN HOME are always visible at the bottom and call the ground
station directly, never through the AI. React + Vite + TypeScript + MapLibre, no internet needed.

Part of the [Searchdrone](../../README.md) monorepo. Licensed under Apache-2.0.

From the repository root:

- `just panel` starts the dev server on `http://127.0.0.1:5173` (loopback only; never use
  `--host`). It proxies `/api`, `/ws` and `/tiles` to the ground station on `127.0.0.1:8080`.
  Log in by opening the one-time link the ground station prints (`…/#login=<code>`).
- `just test` runs the unit tests (Vitest); `just test-panel-e2e` builds the panel, checks
  that the build has no external URLs and fits the size budget, and runs the Playwright
  tests against a fake ground station.
- `just test-panel-sitl` runs the panel in a browser against the whole stack on ArduPilot SITL
  (ground station, onboard computer, the AI with a scripted backend). The stack runs in the
  SITL container; Playwright runs beside it and logs in with a one-time code, like a person.
  It needs the database (`just db-up`) and Playwright's Chromium on the host.
- `just test-video` (after `just video-up`) includes the live video in the panel's test build
  in Google Chrome: the badge says LIVE · WebRTC, the picture moves and the panel counts at
  least 12 frames a second.
