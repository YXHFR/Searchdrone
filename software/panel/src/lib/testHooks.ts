// Test build only (vite --mode test). In the production build these branches are removed,
// and scripts/check-offline.mjs fails if the hook name is found in dist/.
import type { ViewId } from "../layout/layoutModel";
import { useLiveStore } from "../state/liveStore";
import { useMissionStore } from "../state/missionStore";

export const TEST_HOOKS = import.meta.env.VITE_SD_TEST_HOOKS === "1";

/** What the simulator tests measure in the panel: telemetry delay and video frame rate. */
interface Metrics {
  telemetry: { lastServerT: string | null; lastRecvMs: number | null };
  wsRecvTs: number | null;
  videoFps: number | null;
}

const metrics: Metrics = {
  telemetry: { lastServerT: null, lastRecvMs: null },
  wsRecvTs: null,
  videoFps: null,
};

export function installTestHooks(): void {
  if (!TEST_HOOKS) return;
  (window as unknown as { __sd?: unknown }).__sd = {
    liveStore: useLiveStore,
    missionStore: useMissionStore,
    map: null,
    metrics,
  };
}

/** A live message arrived (`t` = the telemetry's own server time, when it is telemetry). */
export function noteLive(t?: string): void {
  if (!TEST_HOOKS) return;
  const now = Date.now();
  metrics.wsRecvTs = now;
  if (t !== undefined) metrics.telemetry = { lastServerT: t, lastRecvMs: now };
}

/** The video view's frames in the last second (null without a picture). */
export function noteVideoFps(fps: number | null): void {
  if (TEST_HOOKS) metrics.videoFps = fps;
}

/** `?crash=<view>` makes that view throw, to test its error boundary. */
export function shouldCrash(view: ViewId): boolean {
  if (!TEST_HOOKS) return false;
  return new URLSearchParams(window.location.search).get("crash") === view;
}

export function exposeMap(map: unknown): void {
  if (!TEST_HOOKS) return;
  const hooks = (window as unknown as { __sd?: { map: unknown } }).__sd;
  if (hooks) hooks.map = map;
}
