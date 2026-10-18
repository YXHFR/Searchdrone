// The browser part of `just perf` (B1-17), started by sd_sim.perf while the simulated
// drone flies: logs in like a person, counts the frames of the live video for
// SD_PERF_VIDEO_S seconds (P2) and measures telemetry latency and rate over a second live
// WebSocket for SD_PERF_TELEMETRY_S seconds, with timed pings before and after for the
// clock offset (P7). The raw numbers go to SD_PERF_RESULT; the Python side evaluates them.
// The result file is always written, also on failure. The login code is never printed or
// written anywhere.
import { writeFileSync } from "node:fs";
import { expect, test } from "@playwright/test";
import { showView } from "../helpers/safety";
import { countVideoFrames, type LiveProbe, liveProbe } from "./probes";

const VEHICLE_ID = "sim-01";
const PINGS = 10;

interface Result {
  video: { frames: number; seconds: number; fps: number; mode: string } | null;
  telemetry: Omit<LiveProbe, "errors">;
  errors: string[];
}

function seconds(name: string, fallback: number): number {
  const raw = process.env[name];
  const n = raw ? Number(raw) : fallback;
  return Number.isFinite(n) && n > 0 ? n : fallback;
}

test("perf: video frames and telemetry latency in the panel", async ({ page }) => {
  const base = (process.env.SD_PERF_BASE_URL ?? "").replace(/\/+$/, "");
  const code = process.env.SD_PERF_LOGIN_CODE ?? "";
  const resultPath = process.env.SD_PERF_RESULT;
  const videoS = seconds("SD_PERF_VIDEO_S", 30);
  const telemetryS = seconds("SD_PERF_TELEMETRY_S", 120);
  const result: Result = {
    video: null,
    telemetry: { arrivals: [], pings_start: [], pings_end: [] },
    errors: [],
  };
  // Error texts can quote the URL; the login code must never reach the output.
  const clean = (err: unknown) => {
    const text = err instanceof Error ? err.message : String(err);
    return code ? text.split(code).join("***") : text;
  };

  try {
    if (!base) throw new Error("SD_PERF_BASE_URL missing");
    if (!code) throw new Error("SD_PERF_LOGIN_CODE missing");
    // The code goes in as an evaluate argument, not in a goto URL: Playwright quotes goto
    // URLs in step titles and timeout reports.
    await page.goto(`${base}/`);
    await page
      .evaluate((c) => {
        window.location.hash = `login=${c}`;
        window.location.reload();
      }, code)
      .catch(() => undefined); // the reload may end the evaluate's context
    await page.getByTestId("topbar").waitFor({ timeout: 30_000 });
    if (new URL(page.url()).hash !== "") throw new Error("the login code stayed in the address");

    try {
      await showView(page, "video");
      const view = page.getByTestId("video-view");
      await expect(view).toHaveAttribute("data-mode", /^(webrtc|hls)$/, { timeout: 60_000 });
      await page.waitForFunction(
        () => {
          const v = document.querySelector('[data-testid="video-element"]');
          return v instanceof HTMLVideoElement && v.readyState >= 2 && v.videoWidth > 0;
        },
        undefined,
        { timeout: 30_000 },
      );
      const count = await page.evaluate(countVideoFrames, { seconds: videoS });
      const mode = (await view.getAttribute("data-mode")) ?? "";
      result.video = {
        frames: count.frames,
        seconds: count.seconds,
        fps: count.seconds > 0 ? count.frames / count.seconds : 0,
        mode,
      };
    } catch (err) {
      result.errors.push(`video: ${clean(err)}`);
    }

    const live = await page.evaluate(liveProbe, {
      vehicleId: VEHICLE_ID,
      pings: PINGS,
      telemetryMs: telemetryS * 1000,
      keepaliveMs: 10_000,
      pongTimeoutMs: 5_000,
    });
    result.telemetry = {
      arrivals: live.arrivals,
      pings_start: live.pings_start,
      pings_end: live.pings_end,
    };
    result.errors.push(...live.errors.map((e) => `telemetry: ${clean(e)}`));
  } catch (err) {
    result.errors.push(clean(err));
  } finally {
    if (resultPath) writeFileSync(resultPath, JSON.stringify(result));
  }

  expect(result.errors).toEqual([]);
});
