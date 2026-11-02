import { readFileSync, rmSync } from "node:fs";
import { tmpdir } from "node:os";
import { join } from "node:path";

/** Where the preview server writes requests that escaped the fake ground (vite.config.ts). */
export const UNROUTED_LOG = join(tmpdir(), "sd-panel-e2e-unrouted.log");

/** Global setup: start from an empty log; the returned teardown fails the run on any entry,
 * including requests made after a test's own checks ran (while its page was closing). */
export default function globalSetup(): () => void {
  rmSync(UNROUTED_LOG, { force: true });
  return () => {
    let seen: string;
    try {
      seen = readFileSync(UNROUTED_LOG, "utf8");
    } catch {
      return; // nothing escaped
    }
    if (seen.trim()) {
      throw new Error(`Requests escaped the fake ground and reached the preview server:\n${seen}`);
    }
  };
}
