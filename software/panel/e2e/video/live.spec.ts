// B1-09 points 1, 2 and 5: live video RTP → MediaMTX → WHEP through sd-ground's proxy,
// shown within 5 s, with a median latency under 1 000 ms over 50 valid samples of the
// time strip; and the proxy answers 401 without a session.
import { writeFileSync } from "node:fs";
import { expect, test } from "@playwright/test";
import { median, type Sample, sampleVideo } from "../helpers/frameClock";

const VEHICLE = "sim-01";
const WHEP = `/api/v1/video/${VEHICLE}/main/whep`;
const SAMPLES = 50;
const MAX_INVALID = 5;
const SHOWN_WITHIN_MS = 5000;
const MEDIAN_LIMIT_MS = 1000;

test("live video through the proxy: shown within 5 s, median latency < 1 s", async ({
  page,
  request,
}) => {
  const code = process.env.SD_VIDEO_LOGIN_CODE;
  expect(code, "SD_VIDEO_LOGIN_CODE").toBeTruthy();

  // Without a session the proxy refuses (the browser context has no cookie yet).
  const anonymous = await request.post(WHEP, {
    headers: { "X-SD-Client": "panel", "Content-Type": "application/sdp" },
    data: "v=0\r\n",
  });
  expect(anonymous.status()).toBe(401);

  await page.goto(`/#login=${code}`);
  const view = page.getByTestId("video-view");
  await expect(view).toBeVisible({ timeout: 20_000 });
  const shownFrom = Date.now();
  await expect(page.getByTestId("video-badge")).toContainText("LIVE", {
    timeout: SHOWN_WITHIN_MS,
  });
  const shownMs = Date.now() - shownFrom;

  const valid: Sample[] = [];
  let invalid = 0;
  while (valid.length < SAMPLES) {
    const s = await sampleVideo(page, '[data-testid="video-el"]');
    // A wrong bit gives a time far off; such a sample is as invalid as an unsure one.
    if (s.valid && s.latencyMs > -200 && s.latencyMs < 60_000) valid.push(s);
    else invalid += 1;
    expect(invalid, "invalid samples").toBeLessThanOrEqual(MAX_INVALID);
    await page.waitForTimeout(100);
  }
  const latencies = valid.map((s) => s.latencyMs);
  const result = {
    shown_ms: shownMs,
    samples: valid.length,
    invalid,
    median_ms: median(latencies),
    min_ms: Math.min(...latencies),
    max_ms: Math.max(...latencies),
  };
  console.log(`video: ${JSON.stringify(result)}`);
  if (process.env.SD_VIDEO_RESULT)
    writeFileSync(process.env.SD_VIDEO_RESULT, JSON.stringify(result));
  expect(result.median_ms).toBeLessThan(MEDIAN_LIMIT_MS);
});
