// WHEP client (receive only) against sd-ground's video proxy. No STUN/TURN servers: the
// panel and MediaMTX are on the same machine or LAN. The session address from the
// server's Location header is only trusted under /api/v1/video/.

const ICE_GATHER_MS = 2000;
const SESSION_PREFIX = "/api/v1/video/";

export class WhepError extends Error {
  constructor(
    readonly status: number,
    message: string,
  ) {
    super(message);
  }
}

export interface WhepSession {
  stream: MediaStream;
  /** Close the peer connection and end the session on the server (once). */
  close(): void;
}

function waitForIce(pc: RTCPeerConnection, ms: number): Promise<void> {
  if (pc.iceGatheringState === "complete") return Promise.resolve();
  return new Promise((resolve) => {
    const done = () => {
      clearTimeout(timer);
      pc.removeEventListener("icegatheringstatechange", check);
      resolve();
    };
    const check = () => {
      if (pc.iceGatheringState === "complete") done();
    };
    const timer = setTimeout(done, ms);
    pc.addEventListener("icegatheringstatechange", check);
  });
}

/** The session address from Location, or null when it is not under /api/v1/video/. */
export function sessionUrl(location: string | null): string | null {
  if (!location) return null;
  let path: string;
  try {
    const url = new URL(location, window.location.origin);
    if (url.origin !== window.location.origin) return null;
    path = url.pathname;
  } catch {
    return null;
  }
  return path.startsWith(SESSION_PREFIX) && !path.includes("..") ? path : null;
}

function endSession(url: string): void {
  void fetch(url, {
    method: "DELETE",
    credentials: "same-origin",
    keepalive: true,
    headers: { "X-SD-Client": "panel", "Content-Type": "application/json" },
    body: "{}",
  }).catch(() => undefined);
}

/** Start a WHEP session: offer → POST to `whepUrl` → answer. Throws WhepError on failure. */
export async function startWhep(whepUrl: string, signal?: AbortSignal): Promise<WhepSession> {
  const pc = new RTCPeerConnection({ iceServers: [] });
  const stream = new MediaStream();
  pc.addTransceiver("video", { direction: "recvonly" });
  pc.addEventListener("track", (ev) => stream.addTrack(ev.track));
  let session: string | null = null;
  let closed = false;
  const close = () => {
    if (closed) return;
    closed = true;
    pc.close();
    if (session) endSession(session);
  };
  try {
    await pc.setLocalDescription(await pc.createOffer());
    await waitForIce(pc, ICE_GATHER_MS);
    const res = await fetch(whepUrl, {
      method: "POST",
      credentials: "same-origin",
      redirect: "manual",
      signal: signal ?? null,
      headers: { "X-SD-Client": "panel", "Content-Type": "application/sdp" },
      body: pc.localDescription?.sdp ?? "",
    });
    if (res.status !== 201) throw new WhepError(res.status, `WHEP ${res.status}`);
    session = sessionUrl(res.headers.get("Location"));
    if (!session) console.warn("WHEP: sessionens adress ligger utanför /api/v1/video/");
    await pc.setRemoteDescription({ type: "answer", sdp: await res.text() });
    return { stream, close };
  } catch (err) {
    close();
    if (err instanceof WhepError) throw err;
    throw new WhepError(0, err instanceof Error ? err.message : String(err));
  }
}
