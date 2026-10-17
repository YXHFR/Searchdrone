import { defineConfig, devices } from "@playwright/test";

// The panel against the real stack on SITL (`just test-panel-sitl`). The stack serves the
// test build itself; its URL and a one-time login code come from global-setup.ts.
const SWS = { args: ["--use-angle=swiftshader", "--enable-unsafe-swiftshader"] };
const chromium = process.env.SD_CHROMIUM_PATH
  ? { ...SWS, executablePath: process.env.SD_CHROMIUM_PATH }
  : SWS;

export default defineConfig({
  testDir: "e2e/sitl",
  fullyParallel: false,
  workers: 1,
  retries: 0,
  forbidOnly: !!process.env.CI,
  timeout: 12 * 60_000,
  reporter: process.env.CI ? [["list"], ["html", { open: "never" }]] : "list",
  globalSetup: "./e2e/sitl/global-setup.ts",
  use: {
    storageState: "e2e/sitl/.auth/pilot.json",
    trace: "retain-on-failure",
  },
  projects: [
    {
      name: "desktop",
      use: {
        ...devices["Desktop Chrome"],
        viewport: { width: 1280, height: 720 },
        launchOptions: chromium,
      },
    },
  ],
});
