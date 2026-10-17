// Logs in exactly like a person: opens the one-time login link the ground station made,
// waits for the shell and keeps the cookie for the tests. No token, no flag.
import { mkdirSync } from "node:fs";
import { chromium } from "@playwright/test";
import { waitForStack } from "./stack";

export default async function globalSetup(): Promise<void> {
  const stack = await waitForStack();
  mkdirSync("e2e/sitl/.auth", { recursive: true });
  const browser = await chromium.launch(
    process.env.SD_CHROMIUM_PATH ? { executablePath: process.env.SD_CHROMIUM_PATH } : {},
  );
  const page = await browser.newPage();
  await page.goto(`${stack.url}/#login=${stack.login_code}`);
  await page.getByTestId("topbar").waitFor({ timeout: 30_000 });
  if (new URL(page.url()).hash !== "") throw new Error("the login code stayed in the address");
  await page.context().storageState({ path: "e2e/sitl/.auth/pilot.json" });
  await browser.close();
}
