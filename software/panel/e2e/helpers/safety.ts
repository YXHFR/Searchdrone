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
