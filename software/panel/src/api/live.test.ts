import { afterEach, beforeEach, describe, expect, it, vi } from "vitest";
import { useLiveStore } from "../state/liveStore";
import { useSessionStore } from "../state/sessionStore";
import { startLive } from "./live";

const sockets: {
  onmessage: ((e: { data: string }) => void) | null;
  onclose: (() => void) | null;
}[] = [];
class FakeWs {
  readyState = 1;
  onmessage: ((e: { data: string }) => void) | null = null;
  onclose: (() => void) | null = null;
  constructor() {
    sockets.push(this);
  }
  send(): void {}
  close(): void {
    this.onclose?.();
  }
}

const calls: string[] = [];
function json(body: unknown): Response {
  return new Response(JSON.stringify(body), {
    status: 200,
    headers: { "Content-Type": "application/json" },
  });
}

beforeEach(() => {
  sockets.length = 0;
  calls.length = 0;
  vi.stubGlobal("WebSocket", FakeWs);
  vi.stubGlobal(
    "fetch",
    vi.fn((url: string, init?: RequestInit) => {
      calls.push(`${init?.method ?? "GET"} ${url}`);
      if (url.endsWith("/state"))
        return Promise.resolve(json({ vehicle_id: "sim-01", telemetry: null }));
      return Promise.resolve(json({ id: "M1", title: "Uppdrag" }));
    }),
  );
  useSessionStore.getState().set({ session: { role: "pilot" } as never });
});
afterEach(() => vi.unstubAllGlobals());

const hello = (epoch: string) =>
  sockets.at(-1)?.onmessage?.({
    data: JSON.stringify({
      seq: null,
      topic: "system",
      type: "hello",
      data: { server_epoch: epoch },
    }),
  });
const settle = () => new Promise((r) => setTimeout(r, 0));

describe("startLive", () => {
  it("opens the mission once, reloads state per epoch and the mission only after a server restart", async () => {
    const stop = startLive("sim-01");
    await settle();
    expect(calls).toEqual(["POST /api/v1/vehicles/sim-01/missions/open"]);
    hello("e1");
    await settle();
    expect(calls.slice(1)).toEqual(["GET /api/v1/vehicles/sim-01/state"]);
    sockets.at(-1)?.onclose?.();
    await new Promise((r) => setTimeout(r, 700));
    hello("e2");
    await settle();
    await settle();
    expect(calls.slice(2)).toEqual([
      "GET /api/v1/vehicles/sim-01/state",
      "POST /api/v1/vehicles/sim-01/missions/open",
    ]);
    expect(useLiveStore.getState().mission?.id).toBe("M1");
    stop();
  });

  it("observers read the active mission and never open one", async () => {
    useSessionStore.getState().set({ session: { role: "observer" } as never });
    const stop = startLive("sim-01");
    await settle();
    expect(calls).toEqual(["GET /api/v1/missions/active?vehicle_id=sim-01"]);
    stop();
  });
});
