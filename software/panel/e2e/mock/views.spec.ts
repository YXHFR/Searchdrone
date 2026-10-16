import { expect, test } from "../helpers/fakeGround";
import { isMobile } from "../helpers/safety";

const VIEWS = ["chat", "map", "video", "candidates", "status", "manual"] as const;

test.describe("views and layout", () => {
  test("desktop: the standard layout combines chat, map and video", async ({ page, ground }) => {
    test.skip(isMobile(page) || (page.viewportSize()?.width ?? 0) < 1200, "desktop only");
    await ground.open();
    for (const v of ["chat", "map", "video"])
      await expect(page.getByTestId(`view-${v}`)).toBeVisible();
  });

  test("every view can be shown, put in full screen and restored", async ({ page, ground }) => {
    test.skip(isMobile(page), "desktop and tablet");
    await ground.open();
    await expect(page.getByTestId("view-chat")).toBeVisible();
    for (const v of VIEWS) {
      // Show the view in the first pane, then maximize it.
      const first = page.locator("[data-view]").first();
      const current = await first.getAttribute("data-view");
      if (current !== v) await first.locator("select").first().selectOption(v);
      await page.getByTestId(`maximize-${v}`).click();
      await expect(page.locator(".frame")).toHaveCount(1);
      await expect(page.getByTestId(`view-${v}`)).toBeVisible();
      await page.keyboard.press("Escape");
      await expect(page.locator(".frame").first()).toBeVisible();
      expect(await page.locator(".frame").count()).toBeGreaterThan(1);
    }
  });

  test("presets give the right views and the layout survives a reload", async ({
    page,
    ground,
  }) => {
    test.skip((page.viewportSize()?.width ?? 0) < 1200, "desktop only");
    await ground.open();
    const presets: Record<string, string[]> = {
      standard: ["chat", "map", "video"],
      map_video: ["map", "video"],
      chat_only: ["chat"],
      quad: ["chat", "map", "video", "status"],
      flight: ["map", "video", "status"],
    };
    for (const [id, views] of Object.entries(presets)) {
      await page.getByTestId("layout-menu").click();
      await page.getByTestId(`preset-${id}`).click();
      await expect(page.locator(".frame")).toHaveCount(views.length);
      for (const v of views) await expect(page.getByTestId(`view-${v}`)).toBeVisible();
    }
    await page.reload();
    await expect(page.locator(".frame")).toHaveCount(3);
    await expect(page.getByTestId("view-status")).toBeVisible();
  });

  test("a broken stored layout falls back to standard without errors", async ({ page, ground }) => {
    test.skip(isMobile(page), "desktop and tablet");
    const errors: string[] = [];
    page.on("pageerror", (e) => errors.push(e.message));
    await page.addInitScript(() => localStorage.setItem("sd.panel.layout.v1.desktop", "{broken"));
    await ground.open();
    await expect(page.getByTestId("view-chat")).toBeVisible();
    expect(errors).toEqual([]);
  });

  test("tablet shows at most two views", async ({ page, ground }) => {
    const w = page.viewportSize()?.width ?? 0;
    test.skip(w < 768 || w >= 1200, "tablet only");
    await ground.open();
    await page.getByTestId("layout-menu").click();
    await page.getByTestId("preset-quad").click();
    await expect(page.locator(".frame")).toHaveCount(2);
  });

  test("phone: one view at a time via tabs, combined with picture-in-picture", async ({
    page,
    ground,
  }) => {
    test.skip(!isMobile(page), "phone only");
    await ground.open();
    for (const v of VIEWS) {
      await page.getByTestId(`tab-${v}`).click();
      await expect(page.getByTestId(`view-${v}`)).toBeVisible();
    }
    await page.getByTestId("tab-chat").click();
    await page.getByTestId("pip-toggle").click();
    await expect(page.getByTestId("pip")).toHaveAttribute("data-view", "map");
    await page
      .getByTestId("pip")
      .getByRole("button", { name: /Byt plats/ })
      .click();
    await expect(page.getByTestId("view-map")).toBeVisible();
  });

  test("a crashing view shows its own error and STOP still works", async ({ page, ground }) => {
    await ground.open("/?crash=chat");
    await expect(page.getByText("Vyn kunde inte visas.")).toBeVisible();
    await page.getByTestId("safety-stop").click();
    await expect.poll(() => ground.posts("/api/v1/vehicles/sim-01/actions/stop").length).toBe(1);
  });

  test("the panel never reads the device position", async ({ page, ground }) => {
    await ground.open();
    await expect(page.getByTestId("safety-stop")).toBeVisible();
    // Visit every view so each one has rendered at least once.
    if (isMobile(page)) {
      for (const v of VIEWS) await page.getByTestId(`tab-${v}`).click();
    }
    const geo = await page.evaluate(
      () => (window as unknown as { __geoAccess: number }).__geoAccess,
    );
    expect(geo).toBe(0);
  });
});
