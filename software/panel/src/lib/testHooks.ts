// Test build only (vite --mode test). In the production build these branches are removed,
// and scripts/check-offline.mjs fails if the hook name is found in dist/.
import type { ViewId } from "../layout/layoutModel";
import { useLiveStore } from "../state/liveStore";

export const TEST_HOOKS = import.meta.env.VITE_SD_TEST_HOOKS === "1";

export function installTestHooks(): void {
  if (!TEST_HOOKS) return;
  (window as unknown as { __sd?: unknown }).__sd = {
    liveStore: useLiveStore,
    map: null,
  };
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
