// Reads the time code strip that the simulator's test picture draws into every frame
// (sd_sim.timecode): the wall clock in milliseconds, its 48 lowest bits, most
// significant first, as 48 boxes of 16 × 16 px in two rows of 24 at the top of the
// picture. White = 1, black = 0. Each box is the mean of its middle 8 × 8 px with the
// threshold 128; at most one box may be uncertain (mean between 64 and 192), more makes
// the sample invalid.
//
// decodeStrip and collectSamples run inside the page (page.evaluate serialises them), so
// they use nothing from this module's scope.

export interface Sample {
  /** Date.now() when the frame was read. */
  now: number;
  /** The time in the frame, or null when the strip could not be read. */
  value: number | null;
}

export interface Summary {
  valid: number;
  invalid: number;
  /** now − value for every valid sample, in ms. */
  delays: number[];
  median: number | null;
}

/** RGBA pixels (canvas ImageData order) of at least the top 32 rows → the time, or null. */
export function decodeStrip(rgba: ArrayLike<number>, width: number): number | null {
  const BITS = 48;
  const BOX = 16;
  const PER_ROW = 24;
  if (width < PER_ROW * BOX) return null;
  let value = 0;
  let uncertain = 0;
  for (let i = 0; i < BITS; i += 1) {
    const x0 = (i % PER_ROW) * BOX + 4;
    const y0 = Math.floor(i / PER_ROW) * BOX + 4;
    let sum = 0;
    for (let y = y0; y < y0 + 8; y += 1) {
      for (let x = x0; x < x0 + 8; x += 1) {
        const p = (y * width + x) * 4;
        sum += 0.299 * (rgba[p] ?? 0) + 0.587 * (rgba[p + 1] ?? 0) + 0.114 * (rgba[p + 2] ?? 0);
      }
    }
    const mean = sum / 64;
    if (mean > 64 && mean < 192) uncertain += 1;
    // 2^48 does not fit in 32-bit integers: multiply instead of shifting.
    value = value * 2 + (mean >= 128 ? 1 : 0);
  }
  return uncertain > 1 ? null : value;
}

/** Test helper: an RGBA picture with the strip for `value` (and a grey background). */
export function stampStrip(width: number, height: number, value: number): Uint8ClampedArray {
  const rgba = new Uint8ClampedArray(width * height * 4).fill(90);
  for (let i = 0; i < 48; i += 1) {
    const bit = Math.floor(value / 2 ** (47 - i)) % 2;
    const x0 = (i % 24) * 16;
    const y0 = Math.floor(i / 24) * 16;
    for (let y = y0; y < y0 + 16; y += 1) {
      for (let x = x0; x < x0 + 16; x += 1) {
        const p = (y * width + x) * 4;
        const c = bit ? 255 : 0;
        rgba[p] = c;
        rgba[p + 1] = c;
        rgba[p + 2] = c;
        rgba[p + 3] = 255;
      }
    }
  }
  return rgba;
}

/**
 * In the page: read `count` frames of the first <video> (one per new frame, via
 * requestVideoFrameCallback when the browser has it) with canvas.drawImage +
 * getImageData, decode them with the decoder installed as window.__sdDecodeStrip.
 */
export function collectSamples(arg: { count: number; timeoutMs: number }): Promise<Sample[]> {
  type Decode = (rgba: ArrayLike<number>, width: number) => number | null;
  const decode = (window as unknown as { __sdDecodeStrip: Decode }).__sdDecodeStrip;
  const video = document.querySelector("video") as HTMLVideoElement | null;
  if (!video) return Promise.reject(new Error("no <video>"));
  const canvas = document.createElement("canvas");
  const ctx = canvas.getContext("2d", { willReadFrequently: true });
  if (!ctx) return Promise.reject(new Error("no 2d context"));
  const samples: Sample[] = [];
  const end = Date.now() + arg.timeoutMs;
  return new Promise((resolve) => {
    const step = () => {
      if (samples.length >= arg.count || Date.now() > end) {
        resolve(samples);
        return;
      }
      const w = video.videoWidth;
      if (w > 0 && video.readyState >= 2) {
        canvas.width = w;
        canvas.height = 32;
        ctx.drawImage(video, 0, 0, w, 32, 0, 0, w, 32);
        const now = Date.now();
        samples.push({ now, value: decode(ctx.getImageData(0, 0, w, 32).data, w) });
      }
      const v = video as HTMLVideoElement & {
        requestVideoFrameCallback?: (cb: () => void) => number;
      };
      if (v.requestVideoFrameCallback) v.requestVideoFrameCallback(step);
      else setTimeout(step, 70);
    };
    step();
  });
}

export function summarize(samples: Sample[]): Summary {
  const delays: number[] = [];
  let invalid = 0;
  for (const s of samples) {
    // A value far from the clock is a misread, not a delay.
    if (s.value === null || Math.abs(s.now - s.value) > 60_000) invalid += 1;
    else delays.push(s.now - s.value);
  }
  const sorted = [...delays].sort((a, b) => a - b);
  const mid = sorted.length / 2;
  const median =
    sorted.length === 0
      ? null
      : sorted.length % 2
        ? (sorted[Math.floor(mid)] ?? null)
        : ((sorted[mid - 1] ?? 0) + (sorted[mid] ?? 0)) / 2;
  return { valid: delays.length, invalid, delays, median };
}
