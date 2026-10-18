import { afterEach, beforeEach, describe, expect, it, vi } from "vitest";
import type { ActionOutcome } from "../api/types";
import { UUIDV7_RE } from "../lib/uuidv7";
import { type ActionLog, TIMING, VehicleActionMachine } from "./vehicleAction";

interface Call {
  url: string;
  init: RequestInit | undefined;
}

type Reply = Response | "hang" | "neterr";

// A real Response reads its body as a stream, which can take macrotasks under load, so
// a fixed number of microtask flushes would be flaky. The body is answered at once.
function json(status: number, body: unknown): Response {
  const text = JSON.stringify(body);
  const res = new Response(text, { status, headers: { "Content-Type": "application/json" } });
  Object.defineProperty(res, "json", { value: () => Promise.resolve(JSON.parse(text)) });
  return res;
}

const DISPATCHED = {
  action_id: "a1",
  action_seq: 7,
  result: "dispatched",
  paths: { mavlink: "queued", sdlink: "queued" },
};

function outcome(p: Partial<ActionOutcome>): ActionOutcome {
  return {
    action: "stop",
    action_id: "a1",
    action_seq: 7,
    by: "operator:demo",
    detail_sv: null,
    mode_confirmed: null,
    mode_requested: "BRAKE",
    paths: { mavlink: "queued", sdlink: "queued" },
    result: "dispatched",
    t: "2026-10-16T10:00:00Z",
    vehicle_id: "sim-01",
    via: "button",
    ...p,
  };
}

function setup(replies: Reply[], pollReply: () => Response = () => json(200, outcome({}))) {
  const calls: Call[] = [];
  const queue = [...replies];
  const fetchFn = vi.fn((input: RequestInfo | URL, init?: RequestInit): Promise<Response> => {
    const url = String(input);
    calls.push({ url, init });
    if (!init?.method || init.method === "GET") return Promise.resolve(pollReply());
    const r = queue.shift() ?? json(202, DISPATCHED);
    if (r === "hang") {
      return new Promise((_, reject) => {
        init.signal?.addEventListener("abort", () =>
          reject(new DOMException("aborted", "AbortError")),
        );
      });
    }
    if (r === "neterr") return Promise.reject(new TypeError("network"));
    return Promise.resolve(r);
  });
  const log: ActionLog & { calls: [string, string][] } = Object.assign(
    (level: "info" | "critical", message: string) => log.calls.push([level, message]),
    { calls: [] as [string, string][] },
  );
  const onUnauthorized = vi.fn();
  const m = new VehicleActionMachine(() => "sim-01", "stop", {
    fetch: fetchFn as typeof fetch,
    log,
    onUnauthorized,
  });
  const posts = () => calls.filter((c) => c.init?.method === "POST");
  const bodyId = (c: Call | undefined) =>
    JSON.parse(String(c?.init?.body)).client_action_id as string;
  return { m, calls, posts, bodyId, log, onUnauthorized };
}

const flush = async () => {
  for (let i = 0; i < 5; i++) await Promise.resolve();
};

beforeEach(() => vi.useFakeTimers());
afterEach(() => vi.useRealTimers());

describe("VehicleActionMachine", () => {
  it("one press = one POST to the action endpoint with a UUIDv7 and the CSRF header, nothing else", async () => {
    const { m, posts } = setup([json(202, DISPATCHED)]);
    m.trigger();
    expect(m.snapshot.state).toBe("sending");
    await flush();
    expect(posts()).toHaveLength(1);
    const p = posts()[0];
    expect(p?.url).toBe("/api/v1/vehicles/sim-01/actions/stop");
    const headers = p?.init?.headers as Record<string, string>;
    expect(headers["X-SD-Client"]).toBe("panel");
    expect(headers["Content-Type"]).toBe("application/json");
    expect(Object.keys(headers).map((h) => h.toLowerCase())).not.toContain("idempotency-key");
    expect(Object.keys(JSON.parse(String(p?.init?.body)))).toEqual(["client_action_id"]);
    expect(JSON.parse(String(p?.init?.body)).client_action_id).toMatch(UUIDV7_RE);
    expect(m.snapshot.state).toBe("sent");
  });

  it("202 without an outcome for 3 s → unconfirmed, never acked", async () => {
    const { m } = setup([json(202, DISPATCHED)], () =>
      json(200, outcome({ result: "dispatched" })),
    );
    m.trigger();
    await flush();
    await vi.advanceTimersByTimeAsync(TIMING.ackWaitMs + 10);
    expect(m.snapshot.state).toBe("unconfirmed");
    await vi.advanceTimersByTimeAsync(60_000);
    expect(m.snapshot.state).toBe("unconfirmed");
  });

  it("an acked outcome (WS) → acked, then idle after 4 s", async () => {
    const { m } = setup([json(202, DISPATCHED)], () =>
      json(200, outcome({ result: "dispatched" })),
    );
    m.trigger();
    await flush();
    m.outcome(outcome({ result: "acked", mode_confirmed: "BRAKE" }));
    expect(m.snapshot.state).toBe("acked");
    await vi.advanceTimersByTimeAsync(TIMING.ackedShownMs + 10);
    expect(m.snapshot.state).toBe("idle");
  });

  it("an ack over the socket before the 202 arrives is kept and gives acked at once", async () => {
    const { m, calls } = setup([json(202, DISPATCHED)], () =>
      json(200, outcome({ result: "dispatched" })),
    );
    m.trigger();
    expect(m.snapshot.state).toBe("sending");
    m.outcome(outcome({ action_id: "other", result: "acked" })); // not ours: never applied
    m.outcome(outcome({ result: "acked", mode_confirmed: "BRAKE" })); // ours, before the 202
    await flush();
    expect(m.snapshot.state).toBe("acked");
    expect(calls.filter((c) => !c.init?.method || c.init.method === "GET")).toEqual([]);
  });

  it("an early outcome for another action does not ack this press", async () => {
    const { m } = setup([json(202, DISPATCHED)], () =>
      json(200, outcome({ result: "dispatched" })),
    );
    m.trigger();
    m.outcome(outcome({ action_id: "other", result: "acked" }));
    await flush();
    expect(m.snapshot.state).toBe("sent");
  });

  it("a late ack after unconfirmed still gives acked", async () => {
    const { m } = setup([json(202, DISPATCHED)], () =>
      json(200, outcome({ result: "dispatched" })),
    );
    m.trigger();
    await flush();
    await vi.advanceTimersByTimeAsync(4_000);
    expect(m.snapshot.state).toBe("unconfirmed");
    m.outcome(outcome({ result: "already_active" }));
    expect(m.snapshot.state).toBe("acked");
  });

  it("onboard receipt alone is never an ack", async () => {
    const { m } = setup([json(202, DISPATCHED)], () =>
      json(200, outcome({ result: "dispatched" })),
    );
    m.trigger();
    await flush();
    m.outcome(outcome({ result: "dispatched", paths: { mavlink: "queued", sdlink: "sent" } }));
    expect(m.snapshot.state).toBe("sent");
    expect(m.snapshot.onboardReceived).toBe(true);
    m.outcome(outcome({ action_id: "other", result: "acked" }));
    expect(m.snapshot.state).toBe("sent");
  });

  it("polls the outcome when no WS event arrives and acks from the poll", async () => {
    const { m, calls } = setup([json(202, DISPATCHED)], () =>
      json(200, outcome({ result: "acked" })),
    );
    m.trigger();
    await flush();
    await vi.advanceTimersByTimeAsync(TIMING.pollAfterMs + 20);
    expect(calls.some((c) => c.url === "/api/v1/vehicles/sim-01/actions/a1")).toBe(true);
    expect(m.snapshot.state).toBe("acked");
  });

  it("outcome failed → failed with the server text", async () => {
    const { m } = setup([json(202, DISPATCHED)], () =>
      json(200, outcome({ result: "dispatched" })),
    );
    m.trigger();
    await flush();
    m.outcome(outcome({ result: "failed", detail_sv: "Flygkontrollern svarar inte" }));
    expect(m.snapshot.state).toBe("failed");
    expect(m.snapshot.text).toEqual({ key: "detail", detail: "Flygkontrollern svarar inte" });
  });

  it.each([
    [403, { code: "FORBIDDEN", detail_sv: "x" }, "safety.error.forbidden"],
    [503, { code: "NO_PATH", detail_sv: "x" }, "safety.error.noPath"],
    [401, { code: "UNAUTHENTICATED", detail_sv: "x" }, "safety.error.session"],
  ])("HTTP %i → failed (%s)", async (status, body, key) => {
    const { m, posts, onUnauthorized } = setup([json(status, body)]);
    m.trigger();
    await flush();
    await flush();
    expect(m.snapshot.state).toBe("failed");
    expect(m.snapshot.text).toEqual({ key });
    expect(posts()).toHaveLength(1);
    expect(onUnauthorized).toHaveBeenCalledTimes(status === 401 ? 1 : 0);
  });

  it("409 shows detail_sv", async () => {
    const { m } = setup([
      json(409, { code: "NOTHING_TO_RELEASE", detail_sv: "Inget att fortsätta" }),
    ]);
    m.trigger();
    await flush();
    await flush();
    expect(m.snapshot.text).toEqual({ key: "detail", detail: "Inget att fortsätta" });
  });

  it("two timeouts then 202 → 3 POSTs with the same client_action_id, then sent", async () => {
    const { m, posts, bodyId } = setup(["hang", "hang", json(202, DISPATCHED)]);
    m.trigger();
    await vi.advanceTimersByTimeAsync(TIMING.attemptTimeoutMs + 5);
    expect(m.snapshot.state).toBe("retrying");
    await vi.advanceTimersByTimeAsync(TIMING.retryDelayMs + TIMING.attemptTimeoutMs + 5);
    await vi.advanceTimersByTimeAsync(TIMING.retryDelayMs + 5);
    await flush();
    expect(posts()).toHaveLength(3);
    expect(new Set(posts().map(bodyId)).size).toBe(1);
    expect(m.snapshot.state).toBe("sent");
  });

  it("502 and network errors are retried with the same id", async () => {
    const { m, posts, bodyId } = setup([json(502, {}), "neterr", json(202, DISPATCHED)]);
    m.trigger();
    await flush();
    await vi.advanceTimersByTimeAsync(TIMING.retryDelayMs + 5);
    await vi.advanceTimersByTimeAsync(TIMING.retryDelayMs + 5);
    await flush();
    expect(posts()).toHaveLength(3);
    expect(new Set(posts().map(bodyId)).size).toBe(1);
    expect(m.snapshot.state).toBe("sent");
  });

  it("no answer at all → failed after 10–13.5 s, all with the same id", async () => {
    const { m, posts, bodyId } = setup(Array.from({ length: 10 }, () => "hang" as const));
    m.trigger();
    await vi.advanceTimersByTimeAsync(9_900);
    expect(m.snapshot.state).not.toBe("failed");
    await vi.advanceTimersByTimeAsync(3_600);
    expect(m.snapshot.state).toBe("failed");
    expect(m.snapshot.text).toEqual({ key: "safety.error.noGround" });
    expect(new Set(posts().map(bodyId)).size).toBe(1);
  });

  it("a new press has a new id and ignores the old press's answer", async () => {
    const { m, posts, bodyId } = setup([
      json(202, DISPATCHED),
      json(202, { ...DISPATCHED, action_id: "a2" }),
    ]);
    m.trigger();
    await flush();
    m.trigger();
    await flush();
    expect(posts()).toHaveLength(2);
    expect(bodyId(posts()[0])).not.toBe(bodyId(posts()[1]));
    m.outcome(outcome({ action_id: "a1", result: "acked" }));
    expect(m.snapshot.state).toBe("sent");
    m.outcome(outcome({ action_id: "a2", result: "acked" }));
    expect(m.snapshot.state).toBe("acked");
  });

  it("unconfirmed stays and raises a critical alarm after 30 s; dismiss → idle", async () => {
    const { m, log } = setup([json(202, DISPATCHED)], () =>
      json(200, outcome({ result: "dispatched" })),
    );
    m.trigger();
    await flush();
    await vi.advanceTimersByTimeAsync(TIMING.ackWaitMs + TIMING.alarmAfterMs + 50);
    expect(m.snapshot.state).toBe("unconfirmed");
    expect(log.calls.some(([lvl]) => lvl === "critical")).toBe(true);
    m.dismiss();
    expect(m.snapshot.state).toBe("idle");
  });

  it("closing an unconfirmed message logs a critical event at once", async () => {
    const { m, log } = setup([json(202, DISPATCHED)], () =>
      json(200, outcome({ result: "dispatched" })),
    );
    m.trigger();
    await flush();
    await vi.advanceTimersByTimeAsync(TIMING.ackWaitMs + 10);
    m.dismiss();
    expect(log.calls.filter(([lvl]) => lvl === "critical")).toHaveLength(1);
  });

  it("a vehicle id that arrives mid-press does not cancel the press (emergency page → config loaded)", async () => {
    let target: string | null = null;
    const urls: string[] = [];
    const bodies: string[] = [];
    let n = 0;
    const fetchFn = vi.fn((input: RequestInfo | URL, init?: RequestInit) => {
      urls.push(String(input));
      if (init?.method !== "POST")
        return Promise.resolve(json(200, outcome({ result: "dispatched" })));
      bodies.push(JSON.parse(String(init.body)).client_action_id);
      n += 1;
      if (n === 1) return Promise.reject(new TypeError("ground down"));
      return Promise.resolve(
        json(202, { actions: [{ ...DISPATCHED, action_id: "a9", vehicle_id: "sim-01" }] }),
      );
    });
    const log = vi.fn();
    const m = new VehicleActionMachine(() => target, "stop", {
      fetch: fetchFn as typeof fetch,
      log,
    });
    m.trigger();
    await flush();
    expect(m.snapshot.state).toBe("retrying");
    target = "sim-01"; // config arrives while the press is being retried
    await vi.advanceTimersByTimeAsync(TIMING.retryDelayMs + 5);
    await flush();
    expect(m.snapshot.state).toBe("sent");
    expect(urls.filter((u) => u.endsWith("stop_all"))).toHaveLength(2);
    expect(new Set(bodies).size).toBe(1);
    // The next press uses the vehicle id.
    m.trigger();
    await flush();
    expect(urls.at(-1)).toBe("/api/v1/vehicles/sim-01/actions/stop");
  });

  it("a 202 whose body never arrives is retried, not left in 'sending'", async () => {
    let n = 0;
    const fetchFn = vi.fn((_: RequestInfo | URL, init?: RequestInit) => {
      n += 1;
      if (n === 1) {
        const stalled = new ReadableStream({
          start(c) {
            init?.signal?.addEventListener("abort", () =>
              c.error(new DOMException("aborted", "AbortError")),
            );
          },
        });
        return Promise.resolve(new Response(stalled, { status: 202 }));
      }
      return Promise.resolve(json(202, DISPATCHED));
    });
    const m = new VehicleActionMachine(() => "sim-01", "stop", {
      fetch: fetchFn as typeof fetch,
      log: () => {},
    });
    m.trigger();
    await vi.advanceTimersByTimeAsync(TIMING.attemptTimeoutMs + TIMING.retryDelayMs + 20);
    await flush();
    expect(n).toBe(2);
    expect(m.snapshot.state).toBe("sent");
  });

  it("without a vehicle id it uses stop_all and acks when every vehicle acked", async () => {
    const calls: string[] = [];
    const fetchFn = vi.fn((input: RequestInfo | URL) => {
      calls.push(String(input));
      return Promise.resolve(
        json(202, {
          actions: [
            { ...DISPATCHED, action_id: "a9", vehicle_id: "sim-01" },
            { ...DISPATCHED, action_id: "b9", vehicle_id: "sim-02" },
          ],
        }),
      );
    });
    const m = new VehicleActionMachine(() => null, "stop", {
      fetch: fetchFn as typeof fetch,
      log: () => {},
    });
    m.trigger();
    await flush();
    expect(calls[0]).toBe("/api/v1/actions/stop_all");
    m.outcome(outcome({ action_id: "a9", result: "acked" }));
    expect(m.snapshot.state).toBe("sent");
    m.outcome(outcome({ action_id: "b9", vehicle_id: "sim-02", result: "acked" }));
    expect(m.snapshot.state).toBe("acked");
  });
});
