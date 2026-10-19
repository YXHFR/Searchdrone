import { defineConfig } from "@playwright/test";

// The browser part of `just perf` (sd_sim.perf), against the stack the driver started.
// Google Chrome, not Playwright's Chromium: only Chrome decodes H.264. One long test, no
// retries and no traces (the login link must not end up in a trace file).
export default defineConfig({
  testDir: "e2e/perf",
  fullyParallel: false,
  workers: 1,
  forbidOnly: !!process.env.CI,
  retries: 0,
  timeout: 15 * 60_000,
  reporter: "list",
  use: {
    baseURL: process.env.SD_PERF_BASE_URL ?? "http://127.0.0.1:8080",
    headless: true,
    trace: "off",
    screenshot: "off",
    video: "off",
    viewport: { width: 1280, height: 720 },
  },
  projects: [
    {
      name: "chrome",
      use: {
        channel: "chrome",
        // Everything is on this machine: never an HTTP proxy from the environment.
        launchOptions: { args: ["--no-proxy-server"] },
      },
    },
  ],
});
