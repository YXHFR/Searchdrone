// In-page measurements for the perf spec (`just perf`). countVideoFrames and liveProbe run
// inside the page (page.evaluate serialises them), so they use nothing from this module's
// scope. Page clock = (performance.timeOrigin + performance.now()) in whole nanoseconds.

export interface VideoCount {
  frames: number;
  seconds: number;
}

export interface Ping {
  t_send_ns: number;
  t_recv_ns: number;
  server_ns: number;
  server_mono_ns: number;
}

export interface LiveProbe {
  /** [time_boot_ms, t_page_ns] for every telemetry envelope in the window. */
  arrivals: [number, number][];
  pings_start: Ping[];
  pings_end: Ping[];
  errors: string[];
}

export interface LiveProbeArg {
  vehicleId: string;
  pings: number;
  telemetryMs: number;
  keepaliveMs: number;
  pongTimeoutMs: number;
}

/** In the page: count requestVideoFrameCallback callbacks of the panel's <video>. */
export function countVideoFrames(arg: { seconds: number }): Promise<VideoCount> {
  type Rvfc = HTMLVideoElement & { requestVideoFrameCallback?: (cb: () => void) => number };
  const video = document.querySelector('[data-testid="video-element"]') as Rvfc | null;
  if (!video) return Promise.reject(new Error("no <video>"));
  if (!video.requestVideoFrameCallback)
    return Promise.reject(new Error("requestVideoFrameCallback missing"));
  const start = performance.now();
  let frames = 0;
  let done = false;
  return new Promise((resolve) => {
    const step = () => {
      if (done) return;
      frames += 1;
      video.requestVideoFrameCallback?.(step);
    };
    video.requestVideoFrameCallback?.(step);
    // A timer ends the count, so a picture that stops still gives an answer.
    setTimeout(() => {
      done = true;
      resolve({ frames, seconds: (performance.now() - start) / 1000 });
    }, arg.seconds * 1000);
  });
}

/**
 * In the page: a second live WebSocket (same origin, session cookie) subscribed to
 * telemetry. `pings` sequential timed pings, then every telemetry arrival for
 * `telemetryMs` (with a plain keepalive ping so the server keeps the socket), then
 * `pings` timed pings again. Never rejects: what was measured comes back with the errors.
 */
export function liveProbe(arg: LiveProbeArg): Promise<LiveProbe> {
  const nowNs = () => Math.round((performance.timeOrigin + performance.now()) * 1e6);
  const out: LiveProbe = { arrivals: [], pings_start: [], pings_end: [], errors: [] };
  const url = `${location.protocol === "https:" ? "wss:" : "ws:"}//${location.host}/ws/v1/live`;
  const ws = new WebSocket(url);
  let recording = false;
  let closed: string | null = null;
  let subscribed: (() => void) | null = null;
  let pong: ((p: { server_ns: number; server_mono_ns: number | null }) => void) | null = null;

  ws.onmessage = (ev) => {
    const t = nowNs();
    let e: { topic?: string; type?: string; data?: unknown; server_ns?: unknown } & Record<
      string,
      unknown
    >;
    try {
      e = JSON.parse(String(ev.data));
    } catch {
      return;
    }
    if (e.topic === "system" && e.type === "hello") {
      const d = (e.data ?? {}) as { server_epoch?: string };
      ws.send(
        JSON.stringify({
          op: "subscribe",
          topics: ["telemetry"],
          vehicle_id: arg.vehicleId,
          mission_id: null,
          server_epoch: d.server_epoch ?? null,
          since_seq: null,
        }),
      );
      subscribed?.();
      return;
    }
    if (e.type === "pong") {
      if (typeof e.server_ns !== "number" || !pong) return; // a keepalive's pong
      const mono = e.server_mono_ns;
      pong({ server_ns: e.server_ns, server_mono_ns: typeof mono === "number" ? mono : null });
      return;
    }
    if (recording && e.topic === "telemetry" && e.type === "telemetry") {
      const boot = (e.data as { time_boot_ms?: unknown } | null)?.time_boot_ms;
      if (typeof boot === "number") out.arrivals.push([boot, t]);
    }
  };
  ws.onclose = (ev) => {
    closed = `WebSocket closed (${ev.code})`;
  };

  const wait = (ms: number) => new Promise<void>((r) => setTimeout(r, ms));

  const ping = (): Promise<Ping> =>
    new Promise((resolve, reject) => {
      if (closed) {
        reject(new Error(closed));
        return;
      }
      const timer = setTimeout(() => {
        pong = null;
        reject(new Error("no pong within the time limit"));
      }, arg.pongTimeoutMs);
      const tSend = nowNs();
      pong = (p) => {
        const tRecv = nowNs();
        clearTimeout(timer);
        pong = null;
        if (p.server_mono_ns === null) {
          reject(new Error("pong without server_mono_ns"));
          return;
        }
        resolve({
          t_send_ns: tSend,
          t_recv_ns: tRecv,
          server_ns: p.server_ns,
          server_mono_ns: p.server_mono_ns,
        });
      };
      ws.send(JSON.stringify({ op: "ping", client_ns: tSend }));
    });

  const pings = async (into: Ping[], label: string) => {
    for (let i = 0; i < arg.pings; i += 1) {
      try {
        into.push(await ping());
      } catch (err) {
        out.errors.push(`${label}: ${(err as Error).message}`);
        return;
      }
    }
  };

  const run = async () => {
    await new Promise<void>((resolve, reject) => {
      const timer = setTimeout(() => reject(new Error("no hello from the server")), 10_000);
      subscribed = () => {
        clearTimeout(timer);
        resolve();
      };
      ws.onerror = () => {
        clearTimeout(timer);
        reject(new Error("WebSocket error"));
      };
    });
    await wait(500); // let the server finish the subscription and any replay
    await pings(out.pings_start, "pings_start");
    recording = true;
    const end = performance.now() + arg.telemetryMs;
    let nextKeepalive = performance.now() + arg.keepaliveMs;
    while (performance.now() < end && !closed) {
      await wait(Math.min(250, Math.max(0, end - performance.now())));
      if (performance.now() >= nextKeepalive && ws.readyState === WebSocket.OPEN) {
        ws.send(JSON.stringify({ op: "ping" }));
        nextKeepalive += arg.keepaliveMs;
      }
    }
    recording = false;
    if (closed) out.errors.push(`telemetry: ${closed}`);
    await pings(out.pings_end, "pings_end");
  };

  return run()
    .catch((err: unknown) => {
      out.errors.push(`live: ${(err as Error).message}`);
    })
    .then(() => {
      ws.onclose = null;
      ws.close();
      return out;
    });
}
