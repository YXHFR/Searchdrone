// PS-1 in Google Chrome against the real stack: the panel's test build (window.__sd)
// served by sd-ground, MediaMTX and the simulator's test picture. "LIVE · WebRTC" within
// 10 s, the picture's time moves at least 2 s in 3 s, and the panel counts at least 12
// frames a second. Started by simulation/tests/video (`just test-video`).
import { expect, test } from "@playwright/test";

interface Hooks {
  metrics: { videoFps: number | null };
}

test("livevideo i panelen (PS-1)", async ({ page }) => {
  const code = process.env.SD_VIDEO_LOGIN_CODE;
  expect(code, "SD_VIDEO_LOGIN_CODE").toBeTruthy();
  await page.goto(`/#login=${code}`);
  await expect(page.locator('meta[name="sd-build"]')).toHaveAttribute("content", "test");
  await expect(page.getByTestId("video-badge")).toContainText("LIVE · WebRTC", {
    timeout: 10_000,
  });

  const video = page.getByTestId("video-element");
  const before = await video.evaluate((v: HTMLVideoElement) => v.currentTime);
  await page.waitForTimeout(3_000);
  const after = await video.evaluate((v: HTMLVideoElement) => v.currentTime);
  expect(after - before).toBeGreaterThanOrEqual(2);

  await expect
    .poll(
      () => page.evaluate(() => (window as unknown as { __sd: Hooks }).__sd.metrics.videoFps ?? 0),
      { timeout: 5_000 },
    )
    .toBeGreaterThanOrEqual(12);
});
