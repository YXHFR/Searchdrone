import { createHash } from "node:crypto";
import { readFileSync, rmSync } from "node:fs";
import { tmpdir } from "node:os";
import { basename, join } from "node:path";
import { fileURLToPath } from "node:url";
import type { FullConfig } from "@playwright/test";

const PANEL_DIR = fileURLToPath(new URL("../..", import.meta.url));

/** Where the preview server writes requests that escaped the fake ground (vite.config.ts);
 * one file per checkout, so parallel runs from other worktrees don't clear it. */
export const UNROUTED_LOG = join(
  tmpdir(),
  `sd-panel-e2e-unrouted-${createHash("sha256").update(PANEL_DIR).digest("hex").slice(0, 12)}.log`,
);

/** The line fakeGround writes once a test's own checks have run, just before its page
 * closes. Requests the server logs for that test after it were sent while the page closed. */
export function endMarker(testId: string, at = new Date()): string {
  return `${at.toISOString()} END ${testId}\n`;
}

/** Splits the log: requests that escaped while their test ran (they fail the run) and
 * requests sent while a page was closing. Once a page starts closing, Playwright stops
 * handling its routes and sends what is in flight on to the network whatever the test
 * does (the panel's POST .../missions/open on load among them), so those are reported,
 * not failed. A request is "while closing" when it was logged after its test's END marker
 * or carries no test header (sent after the page's headers were dropped). One whose test
 * never wrote END (the worker died mid-test) counts as during the test. */
export function classify(log: string): { during: string[]; closing: string[] } {
  const ends = new Map<string, string>();
  const requests: { at: string; test: string | null; line: string }[] = [];
  for (const line of log.split("\n")) {
    if (!line.trim()) continue;
    const [, endAt, endOf] = /^(\S+) END (\S+)$/.exec(line) ?? [];
    if (endAt && endOf) {
      ends.set(endOf, endAt);
      continue;
    }
    const req = /^(\S+) \S+ \S+ \((.*?); referer /.exec(line);
    const header = req?.[2] ?? "";
    requests.push({
      at: req?.[1] ?? "",
      test: header && header !== "no test header" ? (header.split(" ")[0] ?? null) : null,
      line,
    });
  }
  const during: string[] = [];
  const closing: string[] = [];
  for (const r of requests) {
    const end = r.test ? ends.get(r.test) : undefined;
    const afterEnd = r.test === null || (end !== undefined && r.at > end);
    (afterEnd ? closing : during).push(r.line);
  }
  return { during, closing };
}

/** Global setup: check that the preview server has the guard and writes to this checkout's
 * log (a server reused from elsewhere would send escaped requests to :8080, or log them
 * where nothing reads them), and start from an empty log. During a test every /api, /ws
 * and /tiles request is routed (fakeGround), and a response from the guard fails that
 * test in verify(). The returned teardown also fails the run on anything else that
 * escaped while a test ran, e.g. a request whose answer never came back. */
export default async function globalSetup(config: FullConfig): Promise<() => void> {
  const base = config.webServer?.url ?? "http://127.0.0.1:4173";
  const probe = await fetch(`${base}/api/__sd_guard_probe`);
  if (probe.headers.get("x-sd-unrouted") !== basename(UNROUTED_LOG)) {
    throw new Error(
      `The preview server at ${base} does not guard against requests that escape the fake ` +
        `ground into ${UNROUTED_LOG} (a server started elsewhere, or without ` +
        "SD_E2E_UNROUTED_LOG, is being reused): stop it and run again.",
    );
  }
  rmSync(UNROUTED_LOG, { force: true });
  return () => {
    let log: string;
    try {
      log = readFileSync(UNROUTED_LOG, "utf8");
    } catch {
      return; // nothing written
    }
    const { during, closing } = classify(log);
    if (closing.length) {
      console.warn(
        `Requests sent on while a test's page was closing (answered 502):\n${closing.join("\n")}`,
      );
    }
    if (during.length) {
      throw new Error(
        `Requests escaped the fake ground while their test ran:\n${during.join("\n")}`,
      );
    }
  };
}
