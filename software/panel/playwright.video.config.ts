import { defineConfig, devices } from "@playwright/test";

// Live video against a running sd-ground, MediaMTX and the synthetic camera, in Google
// Chrome (its WebRTC has H.264). Started by simulation/tests/video (`just test-video`),
// which passes the panel's address and a one-time login code in the environment.
export default defineConfig({
  testDir: "e2e/video",
  retries: 0,
  workers: 1,
  reporter: "list",
  timeout: 90_000,
  use: {
    ...devices["Desktop Chrome"],
    channel: "chrome",
    viewport: { width: 1280, height: 720 },
    baseURL: process.env.SD_VIDEO_BASE_URL ?? "http://127.0.0.1:8080",
    ignoreHTTPSErrors: true, // LAN mode: self-signed certificate
    trace: "retain-on-failure",
  },
});
