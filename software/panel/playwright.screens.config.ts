import { defineConfig, devices } from "@playwright/test";

// Screenshots of every panel state at the sizes the panel is designed for, dark and light
// (`pnpm screens`). They are a review aid, uploaded by CI; nothing is compared.
const SWS = { args: ["--use-angle=swiftshader", "--enable-unsafe-swiftshader"] };
const chromium = process.env.SD_CHROMIUM_PATH
  ? { ...SWS, executablePath: process.env.SD_CHROMIUM_PATH }
  : SWS;

export default defineConfig({
  testDir: "e2e/screens",
  outputDir: "test-results/screens-run",
  fullyParallel: true,
  retries: 1,
  workers: 2,
  forbidOnly: !!process.env.CI,
  timeout: 90_000,
  reporter: "list",
  webServer: {
    command: "pnpm exec vite preview --outDir dist-test --port 4173 --strictPort --host 127.0.0.1",
    url: "http://127.0.0.1:4173",
    reuseExistingServer: !process.env.CI,
  },
  use: { baseURL: "http://127.0.0.1:4173", launchOptions: chromium },
  projects: [
    {
      name: "1440x900",
      use: { ...devices["Desktop Chrome"], viewport: { width: 1440, height: 900 } },
    },
    {
      name: "1920x1080",
      use: { ...devices["Desktop Chrome"], viewport: { width: 1920, height: 1080 } },
    },
    {
      name: "390x844",
      use: {
        ...devices["Desktop Chrome"],
        viewport: { width: 390, height: 844 },
        isMobile: true,
        hasTouch: true,
      },
    },
  ],
});
