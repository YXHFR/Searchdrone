import { expect, test } from "../helpers/fakeGround";

test("loads with no external requests and sends the security headers", async ({ page, ground }) => {
  const res = await page.goto("/");
  expect(res?.headers()["content-security-policy"]).toContain("default-src 'self'");
  expect(res?.headers()["x-content-type-options"]).toBe("nosniff");
  await expect(page.getByTestId("topbar")).toBeVisible();
  await page.waitForLoadState("networkidle");
  // No external fonts, CDNs or tiles: the fake ground records any other host as a failure.
  expect(ground.external).toEqual([]);
});

test("login with PIN and with a one-time link", async ({ page, ground }) => {
  ground.setSession(null);
  await ground.open();
  const pin = page.getByLabel("PIN");
  await pin.fill("111111");
  await page.getByRole("button", { name: "Logga in" }).click();
  await expect(page.getByText("Fel kod, eller koden har gått ut.")).toBeVisible();
  await pin.fill("123456");
  await page.getByRole("button", { name: "Logga in" }).click();
  await expect(page.getByTestId("topbar")).toBeVisible();
  expect(
    await page.evaluate(
      () => JSON.stringify({ ...localStorage }) + JSON.stringify({ ...sessionStorage }),
    ),
  ).not.toContain("123456");

  ground.setSession(null);
  await page.goto("about:blank");
  await page.goto("/#login=abc");
  await expect(page.getByTestId("topbar")).toBeVisible();
  expect(await page.evaluate(() => location.hash)).toBe("");
  expect(
    ground
      .posts("/api/v1/session")
      .some((r) => (r.body as { login_code?: string }).login_code === "abc"),
  ).toBe(true);
});
