// LAN mode on a phone: the ground's HTTPS listener with its self-signed certificate, and
// the six-digit PIN from `sd-ground pair` typed on the login page.
import { devices, expect, test } from "@playwright/test";
import { readStack } from "./stack";

const { defaultBrowserType: _browser, ...pixel7 } = devices["Pixel 7"];
test.use({ ...pixel7, ignoreHTTPSErrors: true, storageState: { cookies: [], origins: [] } });

test("PIN-inloggning i LAN-läget på en mobil", async ({ page }) => {
  const stack = readStack();
  await page.goto(stack.lan_url);
  await expect(page.getByRole("heading", { name: "Logga in" })).toBeVisible({ timeout: 15_000 });
  await expect(page.getByTestId("safety-stop")).toBeVisible();
  await page.getByLabel("PIN").fill(stack.pin);
  await page.getByRole("button", { name: "Logga in" }).click();
  await expect(page.getByTestId("topbar")).toBeVisible({ timeout: 15_000 });
  await expect(page.getByTestId("mission-title")).toContainText("Uppdrag", { timeout: 15_000 });
  const stored = await page.evaluate(
    () => JSON.stringify({ ...localStorage }) + JSON.stringify({ ...sessionStorage }),
  );
  expect(stored).not.toContain(stack.pin);
});
