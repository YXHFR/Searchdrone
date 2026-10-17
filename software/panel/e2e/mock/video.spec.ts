// P-22: the video view's WHEP → HLS fallback and closing the WHEP session, against the
// fake ground station. The real picture (MediaMTX, Chrome, latency) is tested in
// e2e/video/ against the synthetic camera.
import { expect, test } from "../helpers/fakeGround";
import { isMobile } from "../helpers/safety";

const WHEP = "/api/v1/video/sim-01/main/whep";
const HLS = "/api/v1/video/sim-01/main/hls/index.m3u8";

test.describe("video view (P-22)", () => {
  test.beforeEach(({ page }) => {
    test.skip(isMobile(page) || (page.viewportSize()?.width ?? 0) < 1200, "desktop layout");
  });

  test("WHEP 404 switches to HLS within 6 s", async ({ page, ground }) => {
    ground.setConfig({ video: { whep_url: WHEP, hls_url: HLS } });
    await ground.open();
    await expect(page.getByTestId("video-badge")).toContainText("HLS", { timeout: 6000 });
    expect(ground.posts(WHEP)).toHaveLength(1);
    const offer = ground.posts(WHEP)[0];
    expect(offer?.headers["content-type"]).toBe("application/sdp");
    expect(String(offer?.body)).toContain("m=video");
    expect(String(offer?.body)).toContain("a=recvonly");
  });

  test("closing the view ends the WHEP session exactly once", async ({ page, ground }) => {
    ground.respond(`POST ${WHEP}`, () => ({
      status: 201,
      body: "v=0",
      headers: { Location: `${WHEP}/s1`, "Content-Type": "application/sdp" },
    }));
    ground.setConfig({ video: { whep_url: WHEP, hls_url: null } });
    await ground.open();
    await expect.poll(() => ground.posts(WHEP).length).toBe(1);
    const pane = page.locator('[data-view="video"]');
    await pane.locator("select").first().selectOption("status");
    await expect(page.getByTestId("view-status")).toBeVisible();
    await expect
      .poll(() => ground.calls(/^DELETE \/api\/v1\/video\/sim-01\/main\/whep\/s1$/).length)
      .toBe(1);
    await page.waitForTimeout(500);
    expect(ground.calls(/^DELETE /)).toHaveLength(1);
    expect(ground.calls(/^DELETE /)[0]?.headers["x-sd-client"]).toBe("panel");
  });

  test("no video in the config shows the empty state and Try again", async ({ page, ground }) => {
    ground.setConfig({ video: null });
    await ground.open();
    const view = page.getByTestId("view-video");
    await expect(view).toContainText("Ingen video");
    await expect(page.getByTestId("video-retry")).toBeVisible();
    expect(ground.posts(WHEP)).toHaveLength(0);
  });
});
