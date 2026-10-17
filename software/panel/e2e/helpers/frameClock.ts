// Reads the machine-readable time strip that the synthetic camera (sd_sim.video_relay)
// draws in the top row of every frame: 48 squares of 16 × 16 px carrying the lowest 48
// bits of CLOCK_REALTIME in milliseconds, most significant bit first, white = 1.
//
// Each square is the mean luma of its middle 8 × 8 pixels against the threshold 128. A
// square whose mean lies within 32 of the threshold is "unsure"; a sample with more than
// one unsure square is invalid. The latency of a sample is Date.now() at the moment the
// frame was drawn minus the decoded time (the browser and the simulator share a clock).
import type { Page } from "@playwright/test";

export const BITS = 48;
export const SQUARE = 16;
export const STRIP_WIDTH = BITS * SQUARE;
export const MAX_UNSURE = 1;
const MASK = 2 ** BITS;
const ROWS = { from: 4, to: 12 }; // the middle 8 rows of the strip

export interface FrameClock {
  ms: number;
  unsure: number;
  valid: boolean;
}

export interface Sample extends FrameClock {
  now: number;
  latencyMs: number;
}

const luma = (r: number, g: number, b: number) => 0.299 * r + 0.587 * g + 0.114 * b;

/** Decode from RGBA pixels of rows 4–11 of the strip (`width` px per row, ≥ 768). */
export function decodeRows(rgba: ArrayLike<number>, width: number): FrameClock {
  let ms = 0;
  let unsure = 0;
  for (let i = 0; i < BITS; i++) {
    let total = 0;
    for (let y = 0; y < ROWS.to - ROWS.from; y++) {
      for (let x = i * SQUARE + 4; x < i * SQUARE + 12; x++) {
        const o = (y * width + x) * 4;
        total += luma(rgba[o] ?? 0, rgba[o + 1] ?? 0, rgba[o + 2] ?? 0);
      }
    }
    const mean = total / 64;
    if (Math.abs(mean - 128) < 32) unsure += 1;
    ms = ms * 2 + (mean >= 128 ? 1 : 0); // 48 bits stay exact in a double
  }
  return { ms, unsure, valid: unsure <= MAX_UNSURE };
}

/** Decode from a whole RGBA image (`width` × at least 16 rows). */
export function decodeImage(rgba: ArrayLike<number>, width: number): FrameClock {
  const start = ROWS.from * width * 4;
  const end = ROWS.to * width * 4;
  const rows = Array.prototype.slice.call(rgba, start, end) as number[];
  return decodeRows(rows, width);
}

/** An RGBA image with the strip for `ms` (for tests): `width` × `height`, grey elsewhere. */
export function encodeImage(ms: number, width = 1280, height = 32): Uint8ClampedArray {
  const out = new Uint8ClampedArray(width * height * 4).fill(90);
  const value = ((ms % MASK) + MASK) % MASK;
  for (let i = 0; i < BITS; i++) {
    const bit = Math.floor(value / 2 ** (BITS - 1 - i)) % 2;
    for (let y = 0; y < SQUARE; y++) {
      for (let x = i * SQUARE; x < (i + 1) * SQUARE; x++) {
        const o = (y * width + x) * 4;
        out.fill(bit ? 255 : 0, o, o + 3);
      }
    }
  }
  for (let o = 3; o < out.length; o += 4) out[o] = 255;
  return out;
}

/** The lowest 48 bits of a time in milliseconds, like the strip carries it. */
export function low48(ms: number): number {
  return ((ms % MASK) + MASK) % MASK;
}

/** One sample from the `<video>` matched by `selector`: decoded time, now and latency. */
export async function sampleVideo(page: Page, selector: string): Promise<Sample> {
  const raw = await page.evaluate(
    ({ selector, rows, width }) => {
      const video = document.querySelector<HTMLVideoElement>(selector);
      if (!video || video.videoWidth < width) return null;
      const canvas = document.createElement("canvas");
      canvas.width = width;
      canvas.height = rows.to;
      const ctx = canvas.getContext("2d", { willReadFrequently: true });
      if (!ctx) return null;
      ctx.drawImage(video, 0, 0, width, rows.to, 0, 0, width, rows.to);
      const now = Date.now();
      const data = ctx.getImageData(0, rows.from, width, rows.to - rows.from).data;
      return { now, pixels: Array.from(data) };
    },
    { selector, rows: ROWS, width: STRIP_WIDTH },
  );
  if (!raw) return { ms: 0, unsure: BITS, valid: false, now: Date.now(), latencyMs: Number.NaN };
  const clock = decodeRows(raw.pixels, STRIP_WIDTH);
  return { ...clock, now: raw.now, latencyMs: low48(raw.now) - clock.ms };
}

export function median(values: number[]): number {
  const s = [...values].sort((a, b) => a - b);
  const mid = Math.floor(s.length / 2);
  return s.length % 2 ? (s[mid] ?? Number.NaN) : ((s[mid - 1] ?? 0) + (s[mid] ?? 0)) / 2;
}
