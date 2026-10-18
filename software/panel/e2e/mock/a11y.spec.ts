import AxeBuilder from "@axe-core/playwright";
import { approval, charter, command } from "../fixtures";
import { expect, test } from "../helpers/fakeGround";
import { isMobile } from "../helpers/safety";

const VIEWS = ["chat", "map", "video", "candidates", "status", "manual"] as const;

async function scan(page: import("@playwright/test").Page, label: string): Promise<void> {
  const r = await new AxeBuilder({ page }).analyze();
  // Every violation fails, whatever its impact.
  const bad = r.violations;
  expect(
    bad.map((v) => `${label}: ${v.id} (${v.nodes.map((n) => n.target.join(" ")).join(", ")})`),
  ).toEqual([]);
}

for (const theme of ["dark", "light"] as const) {
  test(`axe: no issues in any view (${theme})`, async ({ page, ground }) => {
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

test("axe: full screen and re-login sheet", async ({ page, ground }) => {
  test.skip(page.viewportSize()?.width !== 1280, "desktop");
  await ground.open();
  await expect(page.getByTestId("mission-title")).toContainText("Uppdrag");
  await page.getByTestId("maximize-chat").click();
  await scan(page, "maximized");
  ground.setSession(null);
  ground.failNext("POST /api/v1/missions/M1/chat", 401, { code: "UNAUTHENTICATED" });
  const box = page.getByRole("textbox", { name: "Meddelande till AI-befälhavaren" });
  await box.fill("hej");
  await box.press("Enter");
  await expect(page.getByTestId("reauth-sheet")).toBeVisible();
  await scan(page, "reauth");
});

test("axe: emergency page", async ({ page, ground }) => {
  test.skip(page.viewportSize()?.width !== 1280, "desktop");
  ground.respond("GET /api/v1/panel/config", () => ({ status: 500, body: {} }), 100);
  await ground.open();
  await expect(page.getByTestId("emergency")).toBeVisible({ timeout: 3_500 });
  await scan(page, "emergency");
});

for (const theme of ["dark", "light"] as const) {
  test(`axe: approval cards and the limits form (${theme})`, async ({ page, ground }) => {
    test.skip(page.viewportSize()?.width !== 1280, "desktop");
    await page.addInitScript((t) => localStorage.setItem("sd.panel.theme", t), theme);
    await ground.open();
    await ground.liveConnected();
    await ground.resynced();
    ground.pushLive("mission", "charter.updated", charter());
    ground.pushLive("approvals", "approval.requested", approval());
    ground.pushLive("commands", "command.created", command());
    ground.pushLive(
      "approvals",
      "approval.requested",
      approval({ id: "AP2", kind: "command", subject_id: "K1" }),
    );
    await expect(page.getByTestId("approval-AP1")).toBeVisible();
    await expect(page.getByTestId("approval-AP2")).toContainText("Order: Starta");
    await scan(page, `${theme}/cards`);
    await page.getByTestId("approval-AP1").getByTestId("limits-open").click();
    await expect(page.getByTestId("limits-form")).toBeVisible();
    await scan(page, `${theme}/limits-form`);
  });
}
