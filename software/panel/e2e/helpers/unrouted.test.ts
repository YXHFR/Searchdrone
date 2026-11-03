// @vitest-environment node
import { describe, expect, it } from "vitest";
import { classify, endMarker } from "./unrouted";

const req = (at: string, what: string, test: string | null) =>
  `${at} ${what} (${test ? `${test} desktop flow.spec.ts:29` : "no test header"}; referer -; empty/cors)`;

describe("classify the requests that reached the preview server", () => {
  it("a POST still in flight when its test's page closed is reported, not failed", () => {
    const log = [
      endMarker("t1.0.0", new Date("2026-11-02T20:00:01.000Z")).trimEnd(),
      req("2026-11-02T20:00:01.020Z", "POST /api/v1/vehicles/sim-01/missions/open", "t1.0.0"),
    ].join("\n");
    expect(classify(log)).toEqual({ during: [], closing: [log.split("\n")[1]] });
  });

  it("anything escaping before its test's END marker fails, whatever the method", () => {
    const get = req("2026-11-02T20:00:00.500Z", "GET /api/v1/missions/M1/charters", "t1.0.0");
    const ws = req("2026-11-02T20:00:00.600Z", "WS /ws/v2/live", "t1.0.0");
    const log = [get, ws, endMarker("t1.0.0", new Date("2026-11-02T20:00:01.000Z"))].join("\n");
    expect(classify(log)).toEqual({ during: [get, ws], closing: [] });
  });

  it("a request without the test header was sent after the page dropped its headers", () => {
    const line = req("2026-11-02T20:00:02.000Z", "POST /api/v1/x", null);
    expect(classify(line)).toEqual({ during: [], closing: [line] });
  });

  it("a test that never wrote END (its worker died) counts as during the test", () => {
    const line = req("2026-11-02T20:00:02.000Z", "GET /api/v1/x", "t9.0.0");
    expect(classify(line)).toEqual({ during: [line], closing: [] });
  });

  it("markers only apply to their own run of a test (retries have their own id)", () => {
    const retry = req("2026-11-02T20:00:05.000Z", "GET /api/v1/x", "t1.1.0");
    const log = [endMarker("t1.0.0", new Date("2026-11-02T20:00:01.000Z")), retry].join("");
    expect(classify(`${log}\n`)).toEqual({ during: [retry], closing: [] });
  });
});
