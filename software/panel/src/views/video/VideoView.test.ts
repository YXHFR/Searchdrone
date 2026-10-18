import { afterEach, beforeEach, describe, expect, it, vi } from "vitest";
import { connect, type Mode, NO_VIDEO_MS, STALL_MS, WHEP_PLAY_MS } from "./VideoView";
import { startWhep, type WhepSession } from "./whepClient";

vi.mock("./whepClient", () => ({ startWhep: vi.fn() }));

const POLL_MS = 250;

/** The parts of a <video> element that connect() uses; the test moves currentTime. */
function fakeVideo(native = true) {
  return {
    currentTime: 0,
    readyState: 0,
    srcObject: null as MediaStream | null,
    src: "",
    play: vi.fn(() => Promise.resolve()),
    load: vi.fn(),
    removeAttribute: vi.fn(),
    canPlayType: vi.fn(() => (native ? "maybe" : "")),
  };
}
type FakeVideo = ReturnType<typeof fakeVideo>;

interface Whep {
  session: WhepSession;
  close: ReturnType<typeof vi.fn>;
  resolve: () => void;
  reject: () => void;
  lost: () => void;
}

/** startWhep under test control: resolve/reject the POST and fire onLost. */
function fakeWhep(): Whep {
  const close = vi.fn(() => Promise.resolve());
  const session = { stream: {} as MediaStream, close } as WhepSession;
  const w = { session, close } as Whep;
  vi.mocked(startWhep).mockImplementationOnce((_url, _signal, onLost) => {
    w.lost = () => onLost?.();
    return new Promise((resolve, reject) => {
      w.resolve = () => resolve(session);
      w.reject = () => reject(new Error("no"));
    });
  });
  return w;
}

/** Advance in poll steps; `moving` makes the picture move on every poll. */
async function run(el: FakeVideo, ms: number, moving: boolean) {
  for (let t = 0; t < ms; t += POLL_MS) {
    if (moving) {
      el.readyState = 4;
      el.currentTime += POLL_MS / 1000;
    }
    await vi.advanceTimersByTimeAsync(POLL_MS);
  }
}

let modes: Mode[];
let stop: (() => void) | null;
const mode = () => modes.at(-1);

function start(el: FakeVideo, hls: string | null = null) {
  stop = connect(el as unknown as HTMLVideoElement, "/whep", hls, (m) => modes.push(m));
}

beforeEach(() => {
  vi.useFakeTimers();
  modes = [];
  stop = null;
});
afterEach(() => {
  stop?.();
  vi.useRealTimers();
  vi.mocked(startWhep).mockReset();
});

describe("connect: a playing picture that stops", () => {
  it("WebRTC: frames stop for 3 s → none, session closed once", async () => {
    const el = fakeVideo();
    const w = fakeWhep();
    start(el);
    w.resolve();
    await run(el, 1000, true);
    expect(mode()).toBe("webrtc");
    await run(el, STALL_MS - 2 * POLL_MS, false);
    expect(mode()).toBe("webrtc");
    await run(el, 2 * POLL_MS, false);
    expect(mode()).toBe("none");
    expect(w.close).toHaveBeenCalledTimes(1);
    expect(el.srcObject).toBeNull();
  });

  it("WebRTC: the time runs on but no new frames come for 3 s → none", async () => {
    const el = fakeVideo();
    let frames = 0;
    const counted = Object.assign(el, {
      getVideoPlaybackQuality: () => ({ totalVideoFrames: frames }),
    });
    const w = fakeWhep();
    start(counted);
    w.resolve();
    for (let t = 0; t < 1000; t += POLL_MS) {
      frames += 4;
      await run(el, POLL_MS, true);
    }
    expect(mode()).toBe("webrtc");
    await run(el, STALL_MS, true); // currentTime moves, the frame count does not
    expect(mode()).toBe("none");
    expect(w.close).toHaveBeenCalledTimes(1);
  });

  it("WebRTC: a short hiccup under 3 s keeps the picture", async () => {
    const el = fakeVideo();
    const w = fakeWhep();
    start(el);
    w.resolve();
    await run(el, 1000, true);
    await run(el, STALL_MS - 2 * POLL_MS, false);
    await run(el, 1000, true);
    await run(el, STALL_MS - 2 * POLL_MS, false);
    expect(mode()).toBe("webrtc");
  });

  it("WebRTC: a lost connection after playback → none at once", async () => {
    const el = fakeVideo();
    const w = fakeWhep();
    start(el);
    w.resolve();
    await run(el, 1000, true);
    w.lost();
    expect(mode()).toBe("none");
    expect(w.close).toHaveBeenCalledTimes(1);
  });

  it("HLS: a started HLS picture that stops for 3 s → none", async () => {
    const el = fakeVideo();
    const w = fakeWhep();
    start(el, "/hls.m3u8");
    w.reject();
    await vi.advanceTimersByTimeAsync(0);
    expect(mode()).toBe("hls");
    expect(el.src).toBe("/hls.m3u8");
    await run(el, 1000, true);
    expect(mode()).toBe("hls");
    await run(el, STALL_MS, false);
    expect(mode()).toBe("none");
    expect(el.removeAttribute).toHaveBeenCalledWith("src");
  });

  it("HLS that never started is left to the 10 s no-video timer", async () => {
    const el = fakeVideo();
    const w = fakeWhep();
    start(el, "/hls.m3u8");
    w.reject();
    await run(el, NO_VIDEO_MS - POLL_MS, false);
    expect(mode()).toBe("hls");
    await run(el, POLL_MS, false);
    expect(mode()).toBe("none");
  });
});

describe("connect: lost connection before playback", () => {
  it("onLost while connecting switches to HLS", async () => {
    const el = fakeVideo();
    const w = fakeWhep();
    start(el, "/hls.m3u8");
    w.resolve();
    await vi.advanceTimersByTimeAsync(0);
    w.lost();
    expect(mode()).toBe("hls");
    expect(w.close).toHaveBeenCalledTimes(1);
  });

  it("a WHEP session that resolves after the switch to HLS is closed, not shown", async () => {
    const el = fakeVideo();
    const w = fakeWhep();
    start(el, "/hls.m3u8");
    await vi.advanceTimersByTimeAsync(WHEP_PLAY_MS);
    expect(mode()).toBe("hls");
    w.resolve();
    await vi.advanceTimersByTimeAsync(0);
    expect(w.close).toHaveBeenCalledTimes(1);
    expect(el.srcObject).toBeNull();
  });
});
