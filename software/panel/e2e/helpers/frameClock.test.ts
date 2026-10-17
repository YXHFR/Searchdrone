import { describe, expect, it } from "vitest";
import { decodeStrip, type Sample, stampStrip, summarize } from "./frameClock";

const W = 640;
const H = 48;
const T = 1_792_000_123_456; // ms, about 2026-10

function box(rgba: Uint8ClampedArray, index: number, shade: number): void {
  const x0 = (index % 24) * 16;
  const y0 = Math.floor(index / 24) * 16;
  for (let y = y0; y < y0 + 16; y += 1)
    for (let x = x0; x < x0 + 16; x += 1) {
      const p = (y * W + x) * 4;
      rgba[p] = shade;
      rgba[p + 1] = shade;
      rgba[p + 2] = shade;
    }
}

describe("frameClock", () => {
  it("decodes a generated picture with a known time", () => {
    expect(decodeStrip(stampStrip(W, H, T), W)).toBe(T);
    expect(decodeStrip(stampStrip(W, H, 0), W)).toBe(0);
    expect(decodeStrip(stampStrip(W, H, 2 ** 48 - 1), W)).toBe(2 ** 48 - 1);
  });

  it("matches the Python encoder's bit order (MSB first, row by row)", () => {
    // Only the last box (index 47, row 2, column 24) white = 1.
    const rgba = stampStrip(W, H, 0);
    box(rgba, 47, 255);
    expect(decodeStrip(rgba, W)).toBe(1);
    const first = stampStrip(W, H, 0);
    box(first, 0, 255);
    expect(decodeStrip(first, W)).toBe(2 ** 47);
  });

  it("survives compression noise", () => {
    const rgba = stampStrip(W, H, T);
    for (let i = 0; i < rgba.length; i += 1)
      rgba[i] = Math.min(255, Math.max(0, (rgba[i] ?? 0) + ((i * 37) % 41) - 20));
    expect(decodeStrip(rgba, W)).toBe(T);
  });

  it("accepts one uncertain box and rejects two", () => {
    const one = stampStrip(W, H, T);
    box(one, 10, 150);
    expect(decodeStrip(one, W)).not.toBeNull();
    const two = stampStrip(W, H, T);
    box(two, 10, 150);
    box(two, 30, 100);
    expect(decodeStrip(two, W)).toBeNull();
  });

  it("refuses a picture too narrow for the strip", () => {
    expect(decodeStrip(new Uint8ClampedArray(320 * 32 * 4), 320)).toBeNull();
  });

  it("summarizes delays and counts misreads", () => {
    const samples: Sample[] = [
      { now: T + 100, value: T },
      { now: T + 300, value: T },
      { now: T + 200, value: T },
      { now: T, value: null },
      { now: T, value: 5 },
    ];
    const s = summarize(samples);
    expect(s.valid).toBe(3);
    expect(s.invalid).toBe(2);
    expect(s.median).toBe(200);
    expect(summarize([]).median).toBeNull();
  });
});
