import AxeBuilder from "@axe-core/playwright";
import { expect, test } from "../helpers/fakeGround";
import { isMobile } from "../helpers/safety";

const VIEWS = ["chat", "map", "video", "candidates", "status", "manual"] as const;

async function scan(page: import("@playwright/test").Page, label: string): Promise<void> {
  const r = await new AxeBuilder({ page }).analyze();
  const bad = r.violations.filter((v) => v.impact === "serious" || v.impact === "critical");
  expect(
    bad.map((v) => `${label}: ${v.id} (${v.nodes.map((n) => n.target.join(" ")).join(", ")})`),
  ).toEqual([]);
}

for (const theme of ["dark", "light"] as const) {
  test(`axe: no serious or critical issues in any view (${theme})`, async ({ page, ground }) => {
    test.skip(
      page.viewportSize()?.width !== 1280 &&
        page.viewportSize()?.width !== 360 &&
        page.viewportSize()?.width !== 768,
      "desktop, tablet, phone-360",
    );
    await page.addInitScript((t) => localStorage.setItem("sd.panel.theme", t), theme);
    await ground.open();
    await expect(page.getByTestId("topbar")).toBeVisible();
    for (const v of VIEWS) {
      if (isMobile(page)) {
        await page.getByTestId(`tab-${v}`).click();
      } else {
        const first = page.locator("[data-view]").first();
        if ((await first.getAttribute("data-view")) !== v)
          await first.locator("select").first().selectOption(v);
      }
      await expect(page.getByTestId(`view-${v}`)).toBeVisible();
      await scan(page, `${theme}/${v}`);
    }
  });
}

test("axe: login page", async ({ page, ground }) => {
  ground.setSession(null);
  await ground.open();
  await expect(page.getByRole("heading", { name: "Logga in" })).toBeVisible();
  await scan(page, "login");
});
