import { expect, type Page } from "@playwright/test";

/**
 * STOP and RETURN HOME are entirely inside the viewport, at least 56 px high, and the
 * element at each button's centre is the button itself (nothing covers it).
 */
export async function expectSafetyVisible(page: Page, label = ""): Promise<void> {
  const vp = page.viewportSize();
  if (!vp) throw new Error("no viewport");
  for (const id of ["safety-stop", "safety-home"]) {
    const btn = page.getByTestId(id);
    await expect(btn, `${label} ${id}`).toBeVisible();
    const box = await btn.boundingBox();
    if (!box) throw new Error(`${id} has no box`);
    expect(box.x, `${label} ${id} left`).toBeGreaterThanOrEqual(0);
    expect(box.y, `${label} ${id} top`).toBeGreaterThanOrEqual(0);
    expect(box.x + box.width, `${label} ${id} right`).toBeLessThanOrEqual(vp.width + 0.5);
    expect(box.y + box.height, `${label} ${id} bottom`).toBeLessThanOrEqual(vp.height + 0.5);
    expect(box.height, `${label} ${id} height`).toBeGreaterThanOrEqual(56);
    if (vp.width < 768)
      expect(box.width, `${label} ${id} width`).toBeGreaterThanOrEqual(0.44 * vp.width - 0.5);
    else expect(box.width, `${label} ${id} width`).toBeGreaterThanOrEqual(160);
    const hit = await page.evaluate(
      ([x, y, testId]) => {
        const el = document.elementFromPoint(x as number, y as number);
        return !!el?.closest(`[data-testid="${testId}"]`);
      },
      [box.x + box.width / 2, box.y + box.height / 2, id] as const,
    );
    expect(hit, `${label} ${id} is not covered`).toBe(true);
  }
}

export const isMobile = (page: Page): boolean => (page.viewportSize()?.width ?? 1280) < 768;

/** The shell's layout: "desktop" (three columns), "medium" (two) or "phone" (map and sheet). */
export async function layoutOf(page: Page): Promise<string> {
  await expect(page.locator(".shell")).toBeVisible();
  return (await page.locator(".shell").getAttribute("data-vp")) ?? "desktop";
}

export type ShownView = "chat" | "events" | "map" | "video" | "candidates" | "status" | "manual";

/**
 * Bring a view up the way an operator would: the side tab (the sheet on a phone), the small
 * window's swap button for map and video, the menu for status and manual, Granska for finds.
 */
export async function showView(page: Page, view: ShownView): Promise<void> {
  const phone = (await layoutOf(page)) === "phone";
  // A phone's overlay is a page of its own; the sheet is under it.
  if (
    phone &&
    view !== "status" &&
    view !== "manual" &&
    (await page.getByTestId("overlay-close").count()) > 0
  )
    await page.getByTestId("overlay-close").click();
  if (view === "chat" || view === "events") {
    await page.getByTestId(`side-tab-${view}`).click();
    await expect(page.getByTestId(view === "chat" ? "view-chat" : "events")).toBeVisible();
    return;
  }
  if (view === "status" || view === "manual") {
    await page.getByTestId("main-menu").click();
    await page.getByTestId(`menu-${view}`).click();
    await expect(page.getByTestId(`view-${view}`)).toBeVisible();
    return;
  }
  if (view === "candidates") {
    if ((await page.getByTestId("mode-review").count()) > 0)
      await page.getByTestId("mode-review").click();
    else {
      await page.getByTestId("side-tab-mission").click();
      await page.getByTestId("finds-open").click();
    }
    await expect(page.getByTestId("view-candidates")).toBeVisible();
    return;
  }
  if ((await page.getByTestId("overlay-close").count()) > 0)
    await page.getByTestId("overlay-close").click();
  if ((await page.getByTestId("mode-fly").count()) > 0) await page.getByTestId("mode-fly").click();
  if ((await page.getByTestId("review-back").count()) > 0)
    await page.getByTestId("review-back").click();
  const swap = page.getByTestId(`pip-swap-${view}`);
  if ((await swap.count()) > 0) await swap.click();
  await expect(page.getByTestId(`view-${view}`)).toBeVisible();
}

/** Press and hold a hold-to-confirm button past its 0.8 s. */
export async function holdClick(page: Page, testId: string, within = page.locator("body")) {
  const btn = within.getByTestId(testId);
  await expect(btn).toBeEnabled();
  await btn.scrollIntoViewIfNeeded();
  const box = await btn.boundingBox();
  if (!box) throw new Error(`${testId} has no box`);
  await page.mouse.move(box.x + box.width / 2, box.y + box.height / 2);
  await page.mouse.down();
  await page.waitForTimeout(1_000);
  await page.mouse.up();
}

const WHEP = "/api/v1/video/sim-01/main/whep";
const HLS = "/api/v1/video/sim-01/main/hls/index.m3u8";

interface CameraGround {
  setConfig(patch: { video: { whep_url: string; hls_url: string } }): void;
  respond(match: string, handler: () => { status: number; body: unknown }, times?: number): void;
}

/** A configured camera that sends no picture: the video window is there, honestly empty. */
export function withCamera(ground: CameraGround): void {
  ground.setConfig({ video: { whep_url: WHEP, hls_url: HLS } });
  ground.respond(`POST ${WHEP}`, () => ({ status: 404, body: { code: "VIDEO_NOT_FOUND" } }), 1_000);
  ground.respond(`GET ${HLS}`, () => ({ status: 404, body: {} }), 1_000);
}
