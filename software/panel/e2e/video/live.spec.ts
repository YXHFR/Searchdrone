// B1-09 points 1 and 2 in a real browser: the panel logs in like a person, the video view
// gets the picture through /api/v1/video/sim-01/main/whep within 5 s, and the delay read
// from the time code strip has a median below 1 000 ms over 50 valid samples (more than
// 5 unreadable samples fail the test). The numbers go to SD_VIDEO_RESULT for the Python
// side (simulation/tests/video).
import { writeFileSync } from "node:fs";
import { expect, test } from "@playwright/test";
import { collectSamples, decodeStrip, summarize } from "../helpers/frameClock";

const WHEP = "/api/v1/video/sim-01/main/whep";
const VALID = 50;
const MAX_INVALID = 5;

test("live video through the proxy, delay below 1 s", async ({ page }) => {
  const code = process.env.SD_VIDEO_LOGIN_CODE;
  expect(code, "SD_VIDEO_LOGIN_CODE").toBeTruthy();
  let offerAt = 0;
  page.on("request", (req) => {
    if (req.method() === "POST" && new URL(req.url()).pathname === WHEP && !offerAt)
      offerAt = Date.now();
  });
  await page.goto(`/#login=${code}`);
  const badge = page.getByTestId("video-badge");
  await expect(badge).toContainText("WebRTC", { timeout: 15_000 });
  await page.waitForFunction(() => {
    const v = document.querySelector("video");
    return !!v && v.readyState >= 2 && v.videoWidth > 0;
  });
  const firstFrameMs = Date.now() - offerAt;
  expect(offerAt).toBeGreaterThan(0);

  await page.evaluate(`window.__sdDecodeStrip = ${decodeStrip.toString()}`);
  const samples = await page.evaluate(collectSamples, {
    count: VALID + MAX_INVALID,
    timeoutMs: 30_000,
  });
  const summary = summarize(samples);
  const result = { first_frame_ms: firstFrameMs, ...summary };
  if (process.env.SD_VIDEO_RESULT)
    writeFileSync(process.env.SD_VIDEO_RESULT, JSON.stringify(result, null, 2));
  expect(firstFrameMs).toBeLessThan(5000);
  expect(summary.invalid).toBeLessThanOrEqual(MAX_INVALID);
  expect(summary.valid).toBeGreaterThanOrEqual(VALID);
  expect(summary.median ?? Number.POSITIVE_INFINITY).toBeLessThan(1000);
});
