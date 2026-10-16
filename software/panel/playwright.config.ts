import { defineConfig, devices } from "@playwright/test";

// Chromium has no software WebGL fallback by itself on GPU-less CI machines.
const SWS = { args: ["--use-angle=swiftshader", "--enable-unsafe-swiftshader"] };
// Local runs may point at a preinstalled Chromium (CI installs Playwright's own browsers).
const chromium = process.env.SD_CHROMIUM_PATH
  ? { ...SWS, executablePath: process.env.SD_CHROMIUM_PATH }
  : SWS;
// No WebKit available locally: run the iPhone project on Chromium instead (CI keeps WebKit).
const chromiumOnly = process.env.SD_PW_CHROMIUM_ONLY === "1";

export default defineConfig({
  testDir: "e2e/mock",
  fullyParallel: true,
  forbidOnly: !!process.env.CI,
  retries: 0,
  reporter: process.env.CI ? [["list"], ["html", { open: "never" }]] : "list",
  webServer: {
    command: "pnpm exec vite preview --outDir dist-test --port 4173 --strictPort --host 127.0.0.1",
    url: "http://127.0.0.1:4173",
    reuseExistingServer: !process.env.CI,
  },
  use: { baseURL: "http://127.0.0.1:4173", trace: "retain-on-failure" },
  projects: [
    {
      name: "desktop",
      use: {
        ...devices["Desktop Chrome"],
        viewport: { width: 1280, height: 720 },
        launchOptions: chromium,
      },
    },
    {
      name: "tablet",
      use: {
        ...devices["Desktop Chrome"],
        viewport: { width: 768, height: 1024 },
        hasTouch: true,
        launchOptions: chromium,
      },
    },
    {
      name: "phone-360",
      use: {
        ...devices["Desktop Chrome"],
        viewport: { width: 360, height: 640 },
        isMobile: true,
        hasTouch: true,
        launchOptions: chromium,
      },
    },
    // Extra sizes from the size list, for the safety-bar visibility matrix.
    {
      name: "desktop-xl",
      testMatch: /safety\.spec\.ts/,
      use: {
        ...devices["Desktop Chrome"],
        viewport: { width: 1920, height: 1080 },
        launchOptions: chromium,
      },
    },
    {
      name: "phone-390",
      testMatch: /safety\.spec\.ts/,
      use: {
        ...devices["Desktop Chrome"],
        viewport: { width: 390, height: 844 },
        isMobile: true,
        hasTouch: true,
        launchOptions: chromium,
      },
    },
    {
      name: "phone-375",
      use: chromiumOnly
        ? {
            ...devices["Desktop Chrome"],
            viewport: { width: 375, height: 812 },
            isMobile: true,
            hasTouch: true,
            launchOptions: chromium,
          }
        : { ...devices["iPhone 15"], viewport: { width: 375, height: 812 } },
    },
  ],
});
