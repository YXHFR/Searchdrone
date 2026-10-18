// P-22: the video view. WHEP first; when the WHEP mock says 404 the view switches to HLS
// within 6 s; a WHEP session that the mock accepts gets exactly one DELETE when the view
// closes; a stream that stops after LIVE ends at "Ingen video" and reconnects by itself;
// no video configured shows "Ingen video" with a retry button.
//
// The accepting WHEP mock answers with a real SDP answer, made in the page by a second
// RTCPeerConnection that sends a canvas, so the view really plays (no network involved).
import type { Page, Route } from "@playwright/test";
import { expect, test } from "../helpers/fakeGround";
import { expectSafetyVisible, isMobile, showView } from "../helpers/safety";

const WHEP = "/api/v1/video/sim-01/main/whep";
const HLS = "/api/v1/video/sim-01/main/hls/index.m3u8";
const SESSION = `${WHEP}/s1`;

async function answerInPage(page: Page, offer: string): Promise<string> {
  return page.evaluate(async (sdp) => {
    const canvas = document.createElement("canvas");
    canvas.width = 640;
    canvas.height = 480;
    const ctx = canvas.getContext("2d");
    let n = 0;
    setInterval(() => {
      if (!ctx) return;
      n += 1;
      ctx.fillStyle = n % 2 ? "#204020" : "#402020";
      ctx.fillRect(0, 0, 640, 480);
    }, 50);
    const sender = new RTCPeerConnection({ iceServers: [] });
    (window as unknown as { __sender: RTCPeerConnection }).__sender = sender;
    for (const track of canvas.captureStream(15).getTracks()) sender.addTrack(track);
    await sender.setRemoteDescription({ type: "offer", sdp });
    await sender.setLocalDescription(await sender.createAnswer());
    await new Promise((r) => setTimeout(r, 300)); // host candidates into the answer
    return sender.localDescription?.sdp ?? "";
  }, offer);
}

/** A WHEP mock that accepts the offer with a real answer; records every POST and DELETE. */
async function acceptWhep(page: Page): Promise<{ deletes: string[]; posts: () => number }> {
  const deletes: string[] = [];
  let count = 0;
  await page.route(`**${WHEP}**`, async (route: Route) => {
    const req = route.request();
    const path = new URL(req.url()).pathname;
    if (req.method() === "POST" && path === WHEP) {
      count += 1;
      const sdp = await answerInPage(page, req.postData() ?? "");
      await route.fulfill({
        status: 201,
        body: sdp,
        headers: { "Content-Type": "application/sdp", Location: SESSION },
      });
      return;
    }
    if (req.method() === "DELETE") {
      deletes.push(path);
      expect(req.headers()["x-sd-client"]).toBe("panel");
      await route.fulfill({ status: 204 });
      return;
    }
    await route.fulfill({ status: 404 });
  });
  return { deletes, posts: () => count };
}

test.describe("video view (P-22)", () => {
  test.beforeEach(({ page }) => {
    test.skip(isMobile(page) || (page.viewportSize()?.width ?? 0) < 1200, "desktop layout");
  });

  test("WHEP 404 switches to HLS within 6 s", async ({ page, ground }) => {
    ground.setConfig({ video: { whep_url: WHEP, hls_url: HLS } });
    ground.respond(`POST ${WHEP}`, () => ({ status: 404, body: { code: "VIDEO_NOT_FOUND" } }), 5);
    ground.respond(`GET ${HLS}`, () => ({ status: 404, body: {} }), 50);
    await ground.open();
    const start = Date.now();
    await expect(page.getByTestId("video-badge")).toContainText("HLS", { timeout: 6000 });
    expect(Date.now() - start).toBeLessThan(6000);
    expect(ground.posts(WHEP).length).toBeGreaterThanOrEqual(1);
    expect(ground.posts(WHEP)[0]?.headers["content-type"]).toContain("application/sdp");
  });

  test("an accepted session is deleted exactly once when the view closes", async ({
    page,
    ground,
  }) => {
    ground.setConfig({ video: { whep_url: WHEP, hls_url: null } });
    const { deletes, posts } = await acceptWhep(page);
    await ground.open();
    await expect(page.getByTestId("video-badge")).toContainText("WebRTC", { timeout: 8000 });
    expect(posts()).toBe(1);
    expect(deletes).toEqual([]);
    await expectSafetyVisible(page, "video playing");
    // Granska takes the stage: the video unmounts and its session goes.
    await showView(page, "candidates");
    await expect(page.getByTestId("view-video")).toHaveCount(0);
    await expect.poll(() => deletes.length).toBe(1);
    await page.waitForTimeout(500);
    expect(deletes).toEqual([SESSION]);
  });

  // Frames stop with the connection still up (camera gone) and the whole peer gone
  // (link or MediaMTX gone): the stall check and the connection-state check each end it.
  const stops = {
    "frames stop, connection stays": () => {
      const pc = (window as unknown as { __sender: RTCPeerConnection }).__sender;
      return Promise.all(pc.getSenders().map((s) => s.replaceTrack(null))).then(() => undefined);
    },
    "the peer connection closes": () => {
      (window as unknown as { __sender: RTCPeerConnection }).__sender.close();
    },
  };
  for (const [name, stop] of Object.entries(stops)) {
    test(`stream stops after LIVE (${name}): badge goes, then a new WHEP POST`, async ({
      page,
      ground,
    }) => {
      test.setTimeout(45_000);
      ground.setConfig({ video: { whep_url: WHEP, hls_url: null } });
      const { deletes, posts } = await acceptWhep(page);
      await ground.open();
      await showView(page, "video");
      await expect(page.getByTestId("video-badge")).toContainText("WebRTC", { timeout: 8000 });
      expect(posts()).toBe(1);
      await page.waitForTimeout(1_500); // the last frame is kept once a second

      await page.evaluate(stop);
      await expect(page.getByTestId("video-badge")).toHaveCount(0, { timeout: 6000 });
      // The last picture stays, grey and dated, never passed off as live.
      await expect(page.getByTestId("video-frozen")).toBeVisible();
      await expect(page.getByTestId("video-stale")).toContainText("Gammal bild");
      await expect(page.getByTestId("hud")).toHaveCount(0);
      await expect(page.getByTestId("video-none")).toContainText("Ingen video");
      await expect(page.getByTestId("video-retry")).toBeVisible();
      await expectSafetyVisible(page, "video lost");
      await expect.poll(() => deletes).toEqual([SESSION]);

      // The automatic retry (every 10 s) posts a new offer and the picture comes back.
      await expect.poll(posts, { timeout: 15_000 }).toBe(2);
      await expect(page.getByTestId("video-badge")).toContainText("WebRTC", { timeout: 8000 });
    });
  }

  test("no video configured: a plain 'no camera' note and no video requests", async ({
    page,
    ground,
  }) => {
    ground.setConfig({ video: null });
    await ground.open();
    await expect(page.getByTestId("no-camera")).toContainText("Ingen kamera");
    await expect(page.getByTestId("view-video")).toHaveCount(0);
    await expect(page.getByTestId("pip-swap-video")).toHaveCount(0);
    expect(ground.posts(WHEP)).toEqual([]);
  });
});

test("phone: STOP and RETURN HOME stay visible while the video plays", async ({ page, ground }) => {
  test.skip(!isMobile(page), "phone only");
  ground.setConfig({ video: { whep_url: WHEP, hls_url: null } });
  await acceptWhep(page);
  await ground.open();
  await showView(page, "video");
  await expect(page.getByTestId("video-badge")).toContainText("WebRTC", { timeout: 8000 });
  await expectSafetyVisible(page, "video playing on a phone");
});
