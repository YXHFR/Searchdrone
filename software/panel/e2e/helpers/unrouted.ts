import { createHash } from "node:crypto";
import { readFileSync, rmSync } from "node:fs";
import { tmpdir } from "node:os";
import { join } from "node:path";
import { fileURLToPath } from "node:url";
import type { FullConfig } from "@playwright/test";

const PANEL_DIR = fileURLToPath(new URL("../..", import.meta.url));

/** Where the preview server writes requests that escaped the fake ground (vite.config.ts);
 * one file per checkout, so parallel runs from other worktrees don't clear it. */
export const UNROUTED_LOG = join(
  tmpdir(),
  `sd-panel-e2e-unrouted-${createHash("sha256").update(PANEL_DIR).digest("hex").slice(0, 12)}.log`,
);

/** Lines a closing page can produce: reads only. A request that changes something, or a
 * socket, escaping is never page-close noise, so it fails the run. */
const READ_ONLY = /^\S+ (GET|HEAD) /;

/** Global setup: check the preview server has the guard (a reused server started without
 * it would send escaped requests to :8080 and nothing would see them) and start from an
 * empty log. During a test every /api, /ws and /tiles request is routed (fakeGround), and
 * one that reaches the server fails that test in verify(). Once a test's page starts
 * closing, Playwright stops handling its routes and sends in-flight requests to the
 * network whatever the test does, so the returned teardown only reports GET and HEAD
 * lines; any other method, or a socket, fails the run. */
export default async function globalSetup(config: FullConfig): Promise<() => void> {
  const base = config.webServer?.url ?? "http://127.0.0.1:4173";
  const probe = await fetch(`${base}/api/__sd_guard_probe`);
  if (probe.headers.get("x-sd-unrouted") !== "1") {
    throw new Error(
      `The preview server at ${base} has no unrouted-request guard (a server started without ` +
        "SD_E2E_UNROUTED_LOG is being reused): stop it and run again.",
    );
  }
  rmSync(UNROUTED_LOG, { force: true });
  return () => {
    let seen: string;
    try {
      seen = readFileSync(UNROUTED_LOG, "utf8");
    } catch {
      return; // nothing reached the server
    }
    const lines = seen.split("\n").filter((l) => l.trim());
    const serious = lines.filter((l) => !READ_ONLY.test(l));
    if (serious.length) {
      throw new Error(
        `Requests that change something, or sockets, escaped the fake ground:\n${serious.join("\n")}`,
      );
    }
    if (lines.length) {
      console.warn(
        "Reads that reached the preview server (answered 502). One made during a test failed " +
          `that test; the others were sent on while a page was closing:\n${lines.join("\n")}`,
      );
    }
  };
}
