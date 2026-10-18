import { expect, test } from "../helpers/fakeGround";
import {
  expectSafetyVisible,
  isMobile,
  layoutOf,
  type ShownView,
  showView,
  withCamera,
} from "../helpers/safety";

const VIEWS: ShownView[] = ["chat", "events", "map", "video", "candidates", "status", "manual"];

test.describe("views and layout", () => {
  test("no camera configured: a plain note instead of an empty video window", async ({
    page,
    ground,
  }) => {
    await ground.open();
    await expect(page.getByTestId("no-camera")).toBeVisible();
    await expect(page.getByTestId("view-video")).toHaveCount(0);
    await expect(page.getByTestId("view-map")).toBeVisible();
  });

  test("desktop: mission column, map with the video in a small window, and the chat", async ({
    page,
    ground,
  }) => {
    test.skip(isMobile(page) || (page.viewportSize()?.width ?? 0) < 1200, "desktop only");
    withCamera(ground);
    await ground.open();
    expect(await layoutOf(page)).toBe("desktop");
    for (const id of ["mission-panel", "finds-panel", "view-map", "view-video", "view-chat"])
      await expect(page.getByTestId(id)).toBeVisible();
    await expect(page.locator(".stage")).toHaveAttribute("data-main", "map");
  });

  test("every view can be shown and the chat stays beside it", async ({ page, ground }) => {
    test.skip(isMobile(page), "desktop and tablet");
    withCamera(ground);
    await ground.open();
    for (const v of VIEWS) {
      await showView(page, v);
      await expect(page.locator(".side-col")).toBeVisible();
    }
    await showView(page, "map");
    await expect(page.locator(".stage")).toHaveAttribute("data-main", "map");
    await showView(page, "video");
    await expect(page.locator(".stage")).toHaveAttribute("data-main", "video");
  });

  test("the theme is one press away and survives a reload", async ({ page, ground }) => {
    test.skip((page.viewportSize()?.width ?? 0) < 900, "theme buttons in the top bar");
    await ground.open();
    await page.getByTestId("theme-light").click();
    await expect(page.locator("html")).toHaveAttribute("data-theme", "light");
    await page.reload();
    await expect(page.locator("html")).toHaveAttribute("data-theme", "light");
    await expect(page.getByTestId("theme-light")).toHaveAttribute("aria-pressed", "true");
    await page.getByTestId("theme-dark").click();
    await expect(page.locator("html")).toHaveAttribute("data-theme", "dark");
  });

  test("a broken stored theme falls back to auto without errors", async ({ page, ground }) => {
    const errors: string[] = [];
    page.on("pageerror", (e) => errors.push(e.message));
    await page.addInitScript(() => localStorage.setItem("sd.panel.theme", "{broken"));
    await ground.open();
    await expect(page.getByTestId("safety-stop")).toBeVisible();
    expect(errors).toEqual([]);
  });

  test("medium width: two columns, the mission as a side tab", async ({ page, ground }) => {
    test.skip((page.viewportSize()?.width ?? 0) < 1200, "resized from desktop");
    await page.setViewportSize({ width: 1024, height: 768 });
    await ground.open();
    expect(await layoutOf(page)).toBe("medium");
    await expect(page.getByTestId("mission-panel")).toHaveCount(0);
    await page.getByTestId("side-tab-mission").click();
    await expect(page.getByTestId("mission-panel")).toBeVisible();
    await expect(page.getByTestId("view-map")).toBeVisible();
    await expectSafetyVisible(page, "medium, mission tab");
  });

  test("phone: the map fills the screen, chat and events in a sheet", async ({ page, ground }) => {
    test.skip((page.viewportSize()?.width ?? 0) >= 900, "phone layout only");
    withCamera(ground);
    await ground.open();
    expect(await layoutOf(page)).toBe("phone");
    const sheet = page.getByTestId("sheet");
    await expect(sheet).toHaveAttribute("data-size", "peek");
    await page.getByTestId("sheet-handle").click();
    await expect(sheet).toHaveAttribute("data-size", "half");
    await page.getByTestId("sheet-handle").click();
    await expect(sheet).toHaveAttribute("data-size", "full");
    await page.getByTestId("sheet-handle").click();
    await expect(sheet).toHaveAttribute("data-size", "peek");
    for (const v of VIEWS) await showView(page, v);
    await showView(page, "candidates");
    await page.getByTestId("review-back").click();
    await expect(page.getByTestId("view-map")).toBeVisible();
    const width = await page.evaluate(() => document.documentElement.scrollWidth);
    expect(width).toBeLessThanOrEqual(page.viewportSize()?.width ?? 0);
  });

  test("a crashing view shows its own error and STOP still works", async ({ page, ground }) => {
    await ground.open("/?crash=chat");
    if ((await layoutOf(page)) === "phone") await page.getByTestId("side-tab-chat").click();
    await expect(page.getByText("Vyn kunde inte visas.")).toBeVisible();
    await page.getByTestId("safety-stop").click();
    await expect.poll(() => ground.posts("/api/v1/vehicles/sim-01/actions/stop").length).toBe(1);
  });

  test("the panel never reads the device position", async ({ page, ground }) => {
    withCamera(ground);
    await ground.open();
    await expect(page.getByTestId("safety-stop")).toBeVisible();
    // Visit every view so each one has rendered at least once.
    for (const v of VIEWS) await showView(page, v);
    const geo = await page.evaluate(
      () => (window as unknown as { __geoAccess: number }).__geoAccess,
    );
    expect(geo).toBe(0);
  });
});
