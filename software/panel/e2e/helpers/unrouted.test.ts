// @vitest-environment node
import { describe, expect, it } from "vitest";
import { CLOSE_WINDOW_MS, classify, endMarker } from "./unrouted";

const req = (at: string, what: string, test: string | null) =>
  `${at} ${what} (${test ? `${test} desktop flow.spec.ts:29` : "no test header"}; referer -; empty/cors)`;
const end = (id: string, at: string) => endMarker(id, new Date(at)).trimEnd();
const later = (at: string, ms: number) => new Date(Date.parse(at) + ms).toISOString();

describe("classify the requests that reached the preview server", () => {
  it("a POST still in flight when its test's page closed is reported, not failed", () => {
    const post = req(
      "2026-11-02T20:00:01.020Z",
      "POST /api/v1/vehicles/sim-01/missions/open",
      "t1.0.0",
    );
    const log = [end("t1.0.0", "2026-11-02T20:00:01.000Z"), post].join("\n");
    expect(classify(log)).toEqual({ during: [], closing: [post] });
  });

  it("anything escaping before its test's END marker fails, whatever the method", () => {
    const get = req("2026-11-02T20:00:00.500Z", "GET /api/v1/missions/M1/charters", "t1.0.0");
    const ws = req("2026-11-02T20:00:00.600Z", "WS /ws/v2/live", "t1.0.0");
    const log = [get, ws, end("t1.0.0", "2026-11-02T20:00:01.000Z")].join("\n");
    expect(classify(log)).toEqual({ during: [get, ws], closing: [] });
  });

  it("a request without the test header shortly after some test's END is from a closing page", () => {
    const at = "2026-11-02T20:00:01.000Z";
    const line = req(later(at, 200), "POST /api/v1/x", null);
    expect(classify([end("t1.0.0", at), line].join("\n"))).toEqual({ during: [], closing: [line] });
  });

  it("a request without the test header and no END just before it fails", () => {
    const at = "2026-11-02T20:00:01.000Z";
    const before = req(later(at, -100), "GET /api/v1/x", null);
    const tooLate = req(later(at, CLOSE_WINDOW_MS + 1), "GET /api/v1/y", null);
    const log = [before, end("t1.0.0", at), tooLate].join("\n");
    expect(classify(log)).toEqual({ during: [before, tooLate], closing: [] });
    expect(classify(before)).toEqual({ during: [before], closing: [] });
  });

  it("a test that never wrote END (its worker died) counts as during the test", () => {
    const line = req("2026-11-02T20:00:02.000Z", "GET /api/v1/x", "t9.0.0");
    expect(classify(line)).toEqual({ during: [line], closing: [] });
  });

  it("markers only apply to their own run of a test (retries have their own id)", () => {
    const retry = req("2026-11-02T20:00:05.000Z", "GET /api/v1/x", "t1.1.0");
    const log = [end("t1.0.0", "2026-11-02T20:00:01.000Z"), retry].join("\n");
    expect(classify(log)).toEqual({ during: [retry], closing: [] });
  });

  it("a request logged in the same millisecond as its own END is from the closing page", () => {
    const at = "2026-11-02T20:00:01.000Z";
    const line = req(at, "GET /api/v1/x", "t1.0.0");
    expect(classify([end("t1.0.0", at), line].join("\n"))).toEqual({ during: [], closing: [line] });
  });

  it("a line it cannot read fails the run instead of being let through", () => {
    for (const bad of [
      "garbage",
      "2026-11-02T20:00:01.000Z GET /api/v1/x",
      "notatime END t1.0.0",
      "1 END t1.0.0",
      "1 GET /api/v1/x (no test header; referer -; empty/cors)",
    ]) {
      expect(classify(bad)).toEqual({ during: [`unreadable log line: ${bad}`], closing: [] });
    }
  });
});
