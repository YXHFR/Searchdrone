// WHEP (WebRTC-HTTP Egress Protocol) client for the live video: one receive-only
// RTCPeerConnection, the SDP offer to the ground station's proxy, and DELETE on the
// session when the view stops. No STUN or TURN servers: the browser and MediaMTX are on
// the same machine or the same LAN.

export const ICE_GATHER_MS = 2000;
const VIDEO_PREFIX = "/api/v1/video/";

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
  /** End the session: close the connection and DELETE it on the server (once). */
  close(): Promise<void>;
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

/** The session URL the server gave, only when it stays under /api/v1/video/. */
export function sessionUrl(location: string | null): string | null {
  if (!location) return null;
  try {
    const url = new URL(location, window.location.origin);
    if (url.origin !== window.location.origin || !url.pathname.startsWith(VIDEO_PREFIX)) {
      return null;
    }
    return url.pathname;
  } catch {
    return null;
  }
}

async function deleteSession(url: string): Promise<void> {
  try {
    await fetch(url, {
      method: "DELETE",
      credentials: "same-origin",
      redirect: "manual",
      headers: { "X-SD-Client": "panel", "Content-Type": "application/json" },
      body: "{}",
      keepalive: true,
    });
  } catch {
    // The server ends idle sessions by itself.
  }
}

/** Start a WHEP session. Rejects with WhepError when the server or WebRTC says no. */
export async function startWhep(whepUrl: string, signal?: AbortSignal): Promise<WhepSession> {
  const pc = new RTCPeerConnection({ iceServers: [] });
  const stream = new MediaStream();
  pc.addTransceiver("video", { direction: "recvonly" });
  pc.addEventListener("track", (ev) => stream.addTrack(ev.track));
  let location: string | null = null;
  let closed = false;
  const close = async () => {
    if (closed) return;
    closed = true;
    pc.close();
    if (location) await deleteSession(location);
  };
  try {
    await pc.setLocalDescription(await pc.createOffer());
    await waitForIce(pc, ICE_GATHER_MS);
    const res = await fetch(whepUrl, {
      method: "POST",
      credentials: "same-origin",
      redirect: "manual",
      headers: { "X-SD-Client": "panel", "Content-Type": "application/sdp" },
      body: pc.localDescription?.sdp ?? "",
      signal: signal ?? null,
    });
    if (res.status !== 201) throw new WhepError(res.status, `WHEP ${res.status}`);
    location = sessionUrl(res.headers.get("Location"));
    if (!location) console.error("WHEP: Location saknas eller ligger utanför /api/v1/video/");
    await pc.setRemoteDescription({ type: "answer", sdp: await res.text() });
    return { stream, close };
  } catch (err) {
    await close();
    if (err instanceof WhepError) throw err;
    throw new WhepError(0, err instanceof Error ? err.message : String(err));
  }
}
