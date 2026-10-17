import { defineConfig } from "@playwright/test";

// The live video against the real stack (sd-ground + MediaMTX + the simulator's test
// picture), started by simulation/tests/video (`just test-video`). Google Chrome, not
// Playwright's Chromium: only Chrome decodes H.264.
export default defineConfig({
  testDir: "e2e/video",
  fullyParallel: false,
  workers: 1,
  forbidOnly: !!process.env.CI,
  retries: 0,
  reporter: "list",
  use: {
    baseURL: process.env.SD_VIDEO_BASE_URL ?? "http://localhost:8080",
    ignoreHTTPSErrors: process.env.SD_VIDEO_IGNORE_HTTPS === "1",
    trace: "retain-on-failure",
    viewport: { width: 1280, height: 720 },
  },
  projects: [
    {
      name: "chrome",
      use: {
        channel: "chrome",
        // Everything is on this machine: never an HTTP proxy from the environment. The LAN
        // test reaches the machine's .local name without mDNS.
        launchOptions: {
          args: [
            "--no-proxy-server",
            ...(process.env.SD_VIDEO_HOST_RULES
              ? [`--host-resolver-rules=${process.env.SD_VIDEO_HOST_RULES}`]
              : []),
          ],
        },
      },
    },
  ],
});
