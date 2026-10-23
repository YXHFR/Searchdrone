// LiveSocket: the only place that opens the WebSocket. Authenticated by the session cookie
// (same origin), never by a ticket or code in the URL.
import { paths } from "./paths";

// The live-feed envelope is typed here until the ground server's WebSocket models land in
// sd_schemas with the live wiring (B1-13); then these local types are replaced by imports.
export interface LiveEnvelope {
  seq: number | null;
  ts?: string;
  topic: string;
  type: string;
  vehicle_id?: string | null;
  mission_id?: string | null;
  data: unknown;
}

const OPEN = 1; // WebSocket.OPEN

export const WS_TIMING = {
  pingEveryMs: 10_000,
  silenceMs: 15_000,
  backoffMs: [500, 1_000, 2_000, 4_000, 5_000],
  jitter: 0.2,
} as const;

export const TOPICS = [
  "telemetry",
  "vehicle",
  "mission",
  "commands",
  "approvals",
  "gate",
  "chat",
  "alerts",
  "detections",
  "system",
  "search_profile",
] as const;

export interface LiveSocketHandlers {
  onState(state: "connecting" | "up" | "down"): void;
  onEnvelope(e: LiveEnvelope): void;
  /** New server epoch (first hello, a restart) or `resync_required`: caller resyncs from REST. */
  onNewEpoch(epoch: string, reason: "hello" | "resync_required"): void;
}

export interface LiveSocketOptions {
  vehicleId: string;
  missionId: () => string | null;
  url?: string;
  random?: () => number;
  createSocket?: (url: string) => WebSocket;
}

export function backoffDelay(attempt: number, random: () => number = Math.random): number {
  const base = WS_TIMING.backoffMs[Math.min(attempt, WS_TIMING.backoffMs.length - 1)] ?? 5_000;
  return Math.round(base * (1 + (random() * 2 - 1) * WS_TIMING.jitter));
}

export function liveUrl(loc: Location = window.location): string {
  return `${loc.protocol === "https:" ? "wss:" : "ws:"}//${loc.host}${paths.liveSocket()}`;
}

export class LiveSocket {
  private ws: WebSocket | null = null;
  private epoch: string | null = null;
  private lastSeq: number | null = null;
  private attempt = 0;
  private closed = false;
  private pingTimer: ReturnType<typeof setInterval> | null = null;
  private silenceTimer: ReturnType<typeof setTimeout> | null = null;
  private retryTimer: ReturnType<typeof setTimeout> | null = null;

  constructor(
    private readonly opts: LiveSocketOptions,
    private readonly h: LiveSocketHandlers,
  ) {}

  start(): void {
    this.closed = false;
    this.connect();
  }

  stop(): void {
    this.closed = true;
    this.clearTimers();
    this.ws?.close();
    this.ws = null;
  }

  private connect(): void {
    this.h.onState("connecting");
    const url = this.opts.url ?? liveUrl();
    const ws = this.opts.createSocket ? this.opts.createSocket(url) : new WebSocket(url);
    this.ws = ws;
    ws.onmessage = (ev) => this.onMessage(ev.data);
    ws.onclose = () => this.lost(ws);
    this.armSilence();
  }

  /** Socket closed or silent: report down at once and schedule the next attempt. */
  private lost(ws: WebSocket): void {
    if (this.ws !== ws) return;
    this.ws = null;
    ws.onmessage = null;
    ws.onclose = null;
    this.clearTimers();
    this.h.onState("down");
    if (!this.closed) {
      this.retryTimer = setTimeout(
        () => this.connect(),
        backoffDelay(this.attempt, this.opts.random),
      );
      this.attempt += 1;
    }
    try {
      ws.close();
    } catch {
      // already closed
    }
  }

  private onMessage(raw: unknown): void {
    this.armSilence();
    let e: LiveEnvelope;
    try {
      e = JSON.parse(String(raw)) as LiveEnvelope;
    } catch {
      return;
    }
    if (e.topic === "system" && e.type === "hello") {
      const d = (e.data ?? {}) as { server_epoch?: string };
      const epoch = d.server_epoch ?? null;
      const same = epoch !== null && epoch === this.epoch;
      if (!same) {
        this.lastSeq = null;
        if (epoch) this.h.onNewEpoch(epoch, "hello");
      }
      this.epoch = epoch;
      this.subscribe(same ? this.lastSeq : null);
      this.attempt = 0;
      this.h.onState("up");
      this.startPing();
      return;
    }
    if (e.type === "resync_required") {
      this.lastSeq = null;
      if (this.epoch) this.h.onNewEpoch(this.epoch, "resync_required");
      this.subscribe(null);
      return;
    }
    if (typeof e.seq === "number") {
      if (this.lastSeq !== null && e.seq <= this.lastSeq) return; // duplicate in this epoch
      this.lastSeq = e.seq;
    }
    this.h.onEnvelope(e);
  }

  /** The current mission changed: subscribe again without losing the events in between. */
  resubscribe(): void {
    if (this.epoch !== null) this.subscribe(this.lastSeq ?? 0);
  }

  /** Fire-and-forget message on the open socket (stick values). False when not open. */
  sendOp(msg: { op: string; [k: string]: unknown }): boolean {
    if (!this.ws || this.ws.readyState !== OPEN) return false;
    this.ws.send(JSON.stringify(msg));
    return true;
  }

  private subscribe(sinceSeq: number | null): void {
    this.send({
      op: "subscribe",
      topics: TOPICS,
      vehicle_id: this.opts.vehicleId,
      mission_id: this.opts.missionId(),
      server_epoch: this.epoch,
      since_seq: sinceSeq,
    });
  }

  private send(msg: unknown): void {
    if (this.ws && this.ws.readyState === OPEN) this.ws.send(JSON.stringify(msg));
  }

  private startPing(): void {
    if (this.pingTimer) clearInterval(this.pingTimer);
    this.pingTimer = setInterval(() => this.send({ op: "ping" }), WS_TIMING.pingEveryMs);
  }

  private armSilence(): void {
    if (this.silenceTimer) clearTimeout(this.silenceTimer);
    const ws = this.ws;
    this.silenceTimer = setTimeout(() => {
      if (ws) this.lost(ws);
    }, WS_TIMING.silenceMs);
  }

  private clearTimers(): void {
    if (this.pingTimer) clearInterval(this.pingTimer);
    if (this.silenceTimer) clearTimeout(this.silenceTimer);
    if (this.retryTimer) clearTimeout(this.retryTimer);
    this.pingTimer = this.silenceTimer = this.retryTimer = null;
  }
}
