import { afterEach, beforeEach, describe, expect, it, vi } from "vitest";
import { backoffDelay, type LiveEnvelope, LiveSocket, WS_TIMING } from "./ws";

class FakeSocket {
  readyState = 1;
  sent: Record<string, unknown>[] = [];
  closed = false;
  onmessage: ((ev: { data: string }) => void) | null = null;
  onclose: (() => void) | null = null;
  constructor(readonly url: string) {}
  send(s: string): void {
    this.sent.push(JSON.parse(s));
  }
  close(): void {
    if (this.closed) return;
    this.closed = true;
    this.readyState = 3;
    this.onclose?.();
  }
  recv(msg: Partial<LiveEnvelope>): void {
    this.onmessage?.({ data: JSON.stringify(msg) });
  }
  hello(epoch: string): void {
    this.recv({ seq: null, topic: "system", type: "hello", data: { server_epoch: epoch } });
  }
}

function setup() {
  const sockets: FakeSocket[] = [];
  const states: string[] = [];
  const envelopes: LiveEnvelope[] = [];
  const epochs: string[] = [];
  const live = new LiveSocket(
    {
      vehicleId: "sim-01",
      missionId: () => "M1",
      url: "ws://test/ws/v1/live",
      random: () => 0.5, // no jitter
      createSocket: (url) => {
        const s = new FakeSocket(url);
        sockets.push(s);
        return s as unknown as WebSocket;
      },
    },
    {
      onState: (s) => states.push(s),
      onEnvelope: (e) => envelopes.push(e),
      onNewEpoch: (e) => epochs.push(e),
    },
  );
  const last = (): FakeSocket => sockets[sockets.length - 1] as FakeSocket;
  return { live, sockets, states, envelopes, epochs, last };
}

const sub = (s: FakeSocket) => s.sent.filter((m) => m.op === "subscribe");

beforeEach(() => vi.useFakeTimers());
afterEach(() => vi.useRealTimers());

describe("LiveSocket", () => {
  it("connects with only the URL (no ticket) and subscribes after hello", () => {
    const { live, last, states, epochs } = setup();
    live.start();
    expect(last().url).toBe("ws://test/ws/v1/live");
    expect(last().sent).toEqual([]);
    last().hello("e1");
    expect(states.at(-1)).toBe("up");
    expect(epochs).toEqual(["e1"]);
    expect(sub(last())[0]).toMatchObject({
      vehicle_id: "sim-01",
      mission_id: "M1",
      server_epoch: "e1",
      since_seq: null,
    });
  });

  it("drops duplicate and older seq in the same epoch, keeps telemetry (seq null)", () => {
    const { live, last, envelopes } = setup();
    live.start();
    last().hello("e1");
    last().recv({ seq: 5, topic: "mission", type: "mission.updated", data: {} });
    last().recv({ seq: 5, topic: "mission", type: "mission.updated", data: {} });
    last().recv({ seq: 3, topic: "mission", type: "mission.updated", data: {} });
    last().recv({ seq: null, topic: "telemetry", type: "telemetry", data: {} });
    last().recv({ seq: null, topic: "telemetry", type: "telemetry", data: {} });
    expect(envelopes.map((e) => e.seq)).toEqual([5, null, null]);
  });

  it("same epoch after a reconnect → since_seq; new epoch after seq 5000 → resync and seq 3 applies", async () => {
    const { live, sockets, last, envelopes, epochs } = setup();
    live.start();
    last().hello("e1");
    last().recv({ seq: 5000, topic: "mission", type: "x", data: {} });
    last().close();
    await vi.advanceTimersByTimeAsync(WS_TIMING.backoffMs[0] + 1);
    expect(sockets).toHaveLength(2);
    last().hello("e1");
    expect(sub(last())[0]).toMatchObject({ since_seq: 5000 });
    expect(epochs).toEqual(["e1"]);
    last().close();
    await vi.advanceTimersByTimeAsync(WS_TIMING.backoffMs[0] + 1);
    last().hello("e2");
    expect(epochs).toEqual(["e1", "e2"]);
    expect(sub(last())[0]).toMatchObject({ since_seq: null, server_epoch: "e2" });
    last().recv({ seq: 3, topic: "mission", type: "mission.updated", data: {} });
    expect(envelopes.at(-1)?.seq).toBe(3);
  });

  it("resync_required → resync and a fresh subscribe", () => {
    const { live, last, epochs } = setup();
    live.start();
    last().hello("e1");
    last().recv({ seq: null, topic: "system", type: "resync_required", data: {} });
    expect(epochs).toEqual(["e1", "e1"]);
    expect(sub(last()).at(-1)).toMatchObject({ since_seq: null });
  });

  it("ignores unknown topics without errors (passes them on to be ignored)", () => {
    const { live, last, envelopes } = setup();
    live.start();
    last().hello("e1");
    last().recv({ seq: 1, topic: "weather", type: "rain", data: {} });
    expect(envelopes).toHaveLength(1);
  });

  it("backs off 0.5, 1, 2, 4, then 5 s, and resets after hello", async () => {
    expect([0, 1, 2, 3, 4, 9].map((a) => backoffDelay(a, () => 0.5))).toEqual([
      500, 1000, 2000, 4000, 5000, 5000,
    ]);
    expect(backoffDelay(0, () => 0)).toBe(400);
    expect(backoffDelay(0, () => 1)).toBe(600);
    const { live, sockets, last } = setup();
    live.start();
    last().close();
    await vi.advanceTimersByTimeAsync(499);
    expect(sockets).toHaveLength(1);
    await vi.advanceTimersByTimeAsync(2);
    expect(sockets).toHaveLength(2);
    last().close();
    await vi.advanceTimersByTimeAsync(1001);
    expect(sockets).toHaveLength(3);
    last().hello("e1");
    last().close();
    await vi.advanceTimersByTimeAsync(501);
    expect(sockets).toHaveLength(4);
  });

  it("15 s of silence → down at once and a new connection", async () => {
    const { live, sockets, last, states } = setup();
    live.start();
    last().hello("e1");
    last().onclose = null; // a dead TCP path: close() never reports back
    await vi.advanceTimersByTimeAsync(WS_TIMING.silenceMs + 1);
    expect(states.at(-1)).toBe("down");
    await vi.advanceTimersByTimeAsync(WS_TIMING.backoffMs[0] + 1);
    expect(sockets).toHaveLength(2);
  });

  it("pings every 10 s", async () => {
    const { live, last } = setup();
    live.start();
    last().hello("e1");
    last().recv({ seq: 1, topic: "x", type: "y", data: {} });
    await vi.advanceTimersByTimeAsync(WS_TIMING.pingEveryMs + 1);
    expect(last().sent.filter((m) => m.op === "ping")).toHaveLength(1);
  });

  it("stop() closes and never reconnects", async () => {
    const { live, sockets } = setup();
    live.start();
    live.stop();
    await vi.advanceTimersByTimeAsync(20_000);
    expect(sockets).toHaveLength(1);
  });
});
