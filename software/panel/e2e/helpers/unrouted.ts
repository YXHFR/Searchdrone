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

/** How long after a test's END marker a request without the test header still counts as
 * sent by that closing page. A full mock run measured such requests 0.9 s and 2.2 s after
 * the END before them; 5 s leaves room for a slower CI runner. */
export const CLOSE_WINDOW_MS = 5_000;

/** Splits the log: requests that escaped while their test ran (they fail the run) and
 * requests sent while a page was closing. Once a page starts closing, Playwright stops
 * handling its routes and sends what is in flight on to the network whatever the test
 * does (the panel's POST .../missions/open on load among them), so those are reported,
 * not failed. A request is "while closing" when it was logged after its own test's END
 * marker, or, without the test header (the page had dropped its headers), within
 * CLOSE_WINDOW_MS after some test's END. Everything else fails: a request before its
 * test's END, one whose test never wrote END (the worker died mid-test), one without the
 * header and no END just before it, and any line that cannot be read. With several
 * workers some test has usually just ended, so the window only bounds header-less
 * requests loosely; with one worker it is tight. */
export function classify(log: string): { during: string[]; closing: string[] } {
  const ends = new Map<string, number>();
  const endTimes: number[] = [];
  const requests: { at: number; test: string | null; line: string }[] = [];
  // The guard writes Date#toISOString(); anything looser is not its line.
  const time = (s: string | undefined): number =>
    s && /^\d{4}-\d\d-\d\dT\d\d:\d\d:\d\d\.\d{3}Z$/.test(s) ? Date.parse(s) : Number.NaN;
  const during: string[] = [];
  for (const line of log.split("\n")) {
    if (!line.trim()) continue;
    const [, endAt, endOf] = /^(\S+) END (\S+)$/.exec(line) ?? [];
    if (endOf && !Number.isNaN(time(endAt))) {
      ends.set(endOf, time(endAt));
      endTimes.push(time(endAt));
      continue;
    }
    const [, at, header] = /^(\S+) \S+ \S+ \((.*?); referer /.exec(line) ?? [];
    if (header === undefined || Number.isNaN(time(at))) {
      during.push(`unreadable log line: ${line}`);
      continue;
    }
    const test = header === "no test header" ? null : (header.split(" ")[0] ?? "");
    requests.push({ at: time(at), test, line });
  }
  const closing: string[] = [];
  for (const r of requests) {
    let afterClose: boolean;
    if (r.test === null) {
      afterClose = endTimes.some((e) => e <= r.at && r.at - e <= CLOSE_WINDOW_MS);
    } else {
      const end = ends.get(r.test);
      // The same millisecond as its END counts as after it: the log has no finer order.
      afterClose = end !== undefined && r.at >= end;
    }
    (afterClose ? closing : during).push(r.line);
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
