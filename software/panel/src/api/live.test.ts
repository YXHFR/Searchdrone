import { afterEach, beforeEach, describe, expect, it, vi } from "vitest";
import { actionBus } from "../state/actionBus";
import { useChatStore } from "../state/chatStore";
import { useLiveStore } from "../state/liveStore";
import { useMissionStore } from "../state/missionStore";
import { useSessionStore } from "../state/sessionStore";
import { useUiStore } from "../state/uiStore";
import { startLive } from "./live";

const sockets: {
  onmessage: ((e: { data: string }) => void) | null;
  onclose: (() => void) | null;
  sent: string[];
}[] = [];
class FakeWs {
  readyState = 1;
  onmessage: ((e: { data: string }) => void) | null = null;
  onclose: (() => void) | null = null;
  sent: string[] = [];
  constructor() {
    sockets.push(this);
  }
  send(m: string): void {
    this.sent.push(m);
  }
  close(): void {
    this.onclose?.();
  }
}

const calls: string[] = [];
let gate: Promise<void> = Promise.resolve();
function json(body: unknown): Response {
  return new Response(JSON.stringify(body), {
    status: 200,
    headers: { "Content-Type": "application/json" },
  });
}
const mission = { id: "M1", title: "Uppdrag", status: "DRAFT", active_charter_id: null };

function answer(url: string): unknown {
  if (url.endsWith("/state")) return { vehicle_id: "sim-01", telemetry: null };
  if (url.endsWith("/missions/open") || url.includes("/missions/active")) return mission;
  if (url.endsWith("/missions/M1"))
    return {
      mission,
      charter: null,
      clearance: null,
      latch: null,
      active_command: null,
      pending_approvals: [],
      sectors: {},
    };
  if (url.includes("/chat")) return { items: [], next_before: null };
  if (url.includes("/sectors")) return { type: "FeatureCollection", features: [] };
  if (url.includes("/track"))
    return { type: "Feature", geometry: { type: "LineString", coordinates: [[1, 2, 3]] } };
  return [];
}

beforeEach(() => {
  sockets.length = 0;
  calls.length = 0;
  gate = Promise.resolve();
  useLiveStore.getState().reset();
  useMissionStore.getState().reset(null);
  useChatStore.getState().clear();
  vi.stubGlobal("WebSocket", FakeWs);
  vi.stubGlobal(
    "fetch",
    vi.fn(async (url: string, init?: RequestInit) => {
      calls.push(`${init?.method ?? "GET"} ${url}`);
      await gate;
      return json(answer(url));
    }),
  );
  useSessionStore.getState().set({ session: { role: "pilot" } as never });
});
afterEach(() => vi.unstubAllGlobals());

const send = (msg: unknown) => sockets.at(-1)?.onmessage?.({ data: JSON.stringify(msg) });
const hello = (epoch: string) =>
  send({ seq: null, topic: "system", type: "hello", data: { server_epoch: epoch } });
const settle = async () => {
  for (let i = 0; i < 20; i += 1) await new Promise((r) => setTimeout(r, 0));
};

const RESYNC = [
  "GET /api/v1/vehicles/sim-01/state",
  "POST /api/v1/vehicles/sim-01/missions/open",
  "GET /api/v1/missions/M1",
  "GET /api/v1/missions/M1/charters",
  "GET /api/v1/missions/M1/approvals?status=pending",
  "GET /api/v1/missions/M1/sectors",
  "GET /api/v1/missions/M1/annotations",
  "GET /api/v1/missions/M1/finds",
  "GET /api/v1/missions/M1/chat?limit=50",
  "GET /api/v1/missions/M1/track?max_points=5000",
];

describe("startLive", () => {
  it("resyncs in the fixed order at start, not again on the first hello, again on a new epoch", async () => {
    const stop = startLive("sim-01");
    await settle();
    expect(calls).toEqual(RESYNC);
    expect(useMissionStore.getState().track).toEqual([[1, 2]]);
    hello("e1");
    await settle();
    expect(calls).toHaveLength(RESYNC.length);
    sockets.at(-1)?.onclose?.();
    await new Promise((r) => setTimeout(r, 700));
    hello("e2");
    await settle();
    expect(calls.slice(RESYNC.length)).toEqual(RESYNC);
    expect(useLiveStore.getState().mission?.id).toBe("M1");
    stop();
  });

  it("resync_required runs the resync again", async () => {
    const stop = startLive("sim-01");
    await settle();
    hello("e1");
    await settle();
    send({ type: "resync_required", data: {} });
    await settle();
    expect(calls.slice(RESYNC.length)).toEqual(RESYNC);
    stop();
  });

  it("observers read the active mission and never open one", async () => {
    useSessionStore.getState().set({ session: { role: "observer" } as never });
    const stop = startLive("sim-01");
    await settle();
    expect(calls[1]).toBe("GET /api/v1/missions/active?vehicle_id=sim-01");
    expect(calls.some((c) => c.includes("/missions/open"))).toBe(false);
    stop();
  });

  it("holds events back during a resync and applies them after it, in seq order", async () => {
    let release: () => void = () => undefined;
    gate = new Promise<void>((r) => {
      release = r;
    });
    const stop = startLive("sim-01");
    hello("e1");
    const approval = (status: string) => ({
      id: "AP1",
      kind: "command",
      mission_id: "M1",
      subject_id: "K1",
      subject_version: 1,
      status,
      required_confirmations: [],
      requested_at: "2026-10-16T10:00:00Z",
    });
    send({
      seq: 6,
      topic: "mission",
      type: "approval.updated",
      mission_id: "M1",
      data: approval("approved"),
    });
    send({
      seq: 5,
      topic: "mission",
      type: "approval.requested",
      mission_id: "M1",
      data: approval("pending"),
    });
    expect(useMissionStore.getState().approvals.AP1).toBeUndefined();
    release();
    await settle();
    expect(useMissionStore.getState().approvals.AP1?.status).toBe("approved");
    stop();
  });

  it("passes a STOP / FLY HOME outcome on at once, even during a resync", async () => {
    let release: () => void = () => undefined;
    gate = new Promise<void>((r) => {
      release = r;
    });
    const seen: string[] = [];
    const off = actionBus.subscribe((o) => seen.push(`${o.action_id}:${o.result}`));
    const stop = startLive("sim-01");
    hello("e1");
    send({
      seq: 7,
      topic: "vehicle",
      type: "vehicle.action",
      vehicle_id: "sim-01",
      data: { action_id: "A1", result: "acked" },
    });
    expect(seen).toEqual(["A1:acked"]); // not held back until the resync ends
    release();
    await settle();
    expect(seen).toEqual(["A1:acked"]);
    off();
    stop();
  });

  it("subscribes again when the mission arrives after the hello", async () => {
    let release: () => void = () => undefined;
    gate = new Promise<void>((r) => {
      release = r;
    });
    const stop = startLive("sim-01");
    hello("e1");
    const before = sockets.at(-1)?.sent.length ?? 0;
    release();
    await settle();
    const subs = (sockets.at(-1)?.sent ?? []).slice(before).map((m) => JSON.parse(m));
    expect(subs.some((m) => m.op === "subscribe" && m.mission_id === "M1")).toBe(true);
    stop();
  });

  it("a new probability-map version makes the map layers fetch again", async () => {
    const stop = startLive("sim-01");
    hello("e1");
    await settle();
    const before = useMissionStore.getState().mapRev;
    send({
      seq: 5,
      topic: "probability_map",
      type: "probability_map.updated",
      mission_id: "M1",
      data: { version: 2 },
    });
    await settle();
    expect(useMissionStore.getState().mapRev).toBe(before + 1);
    const subs = (sockets.at(-1)?.sent ?? []).map((m) => JSON.parse(m));
    expect(subs.some((m) => m.op === "subscribe" && m.topics.includes("probability_map"))).toBe(
      true,
    );
    stop();
  });

  it("words alerts from sv.json; red ones stay until acknowledged; else message_sv", async () => {
    useUiStore.setState({ toasts: [], eventLog: [] });
    const stop = startLive("sim-01");
    await settle();
    const alert = (data: unknown) => send({ seq: null, topic: "alerts", type: "alert", data });
    alert({ code: "GPS_LOST", severity: "critical", message_sv: "x", dr_timeout_s: 30 });
    alert({ code: "GPS_OK", severity: "info", message_sv: "x", after_spoof: false });
    alert({ code: "BATTERY_RTL", severity: "critical", message_sv: "Batteriet är lågt" });
    const ui = useUiStore.getState();
    expect(ui.toasts.map((x) => [x.sticky ?? false, x.text.slice(0, 13)])).toEqual([
      [true, "GPS är borta."],
      [false, "GPS fungerar "],
    ]);
    expect(ui.eventLog.map((x) => x.message.slice(0, 13))).toEqual([
      "Batteriet är ",
      "GPS fungerar ",
      "GPS är borta.",
    ]);
    expect(ui.eventLog[2]?.level).toBe("critical");
    stop();
  });

  it("tells the pilot when person detection is off on board", async () => {
    useUiStore.setState({ toasts: [], eventLog: [] });
    const stop = startLive("sim-01");
    await settle();
    send({
      seq: null,
      topic: "vehicle",
      type: "executor.perception_off",
      vehicle_id: "sim-01",
      data: { reason: "sources_failed", error: "ModelError" },
    });
    const ui = useUiStore.getState();
    expect(ui.toasts.map((x) => x.text.slice(0, 26))).toEqual(["Persondetekteringen är avs"]);
    expect(ui.eventLog.map((x) => x.message.slice(0, 26))).toEqual(["Persondetekteringen är avs"]);
    send({
      seq: null,
      topic: "vehicle",
      type: "executor.perception_on",
      vehicle_id: "sim-01",
      data: {},
    });
    expect(useUiStore.getState().toasts.map((x) => x.text)).toContain(
      "Persondetekteringen fungerar igen ombord.",
    );
    stop();
  });
});
