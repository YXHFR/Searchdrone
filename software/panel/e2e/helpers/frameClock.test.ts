import { describe, expect, it } from "vitest";
import {
  BITS,
  decodeImage,
  decodeRows,
  encodeImage,
  low48,
  median,
  SQUARE,
  STRIP_WIDTH,
} from "./frameClock";

const KNOWN = 1_792_272_650_651; // a time in 2026, in milliseconds

describe("frameClock", () => {
  it("decodes a generated picture with a known time", () => {
    const img = encodeImage(KNOWN, 1280, 32);
    const clock = decodeImage(img, 1280);
    expect(clock).toEqual({ ms: low48(KNOWN), unsure: 0, valid: true });
  });

  it("works at the strip's own width and with the extremes", () => {
    for (const ms of [0, 1, 2 ** 47, 2 ** BITS - 1, Date.UTC(2030, 0, 1)]) {
      expect(decodeImage(encodeImage(ms, STRIP_WIDTH, SQUARE), STRIP_WIDTH).ms).toBe(low48(ms));
    }
  });

  it("is most significant bit first", () => {
    const img = encodeImage(1, STRIP_WIDTH, SQUARE);
    const lastSquare = ((SQUARE / 2) * STRIP_WIDTH + (BITS - 1) * SQUARE + 8) * 4;
    expect(img[lastSquare]).toBe(255);
    expect(img[((SQUARE / 2) * STRIP_WIDTH + 8) * 4]).toBe(0);
  });

  it("accepts one unsure square and rejects two", () => {
    const img = encodeImage(KNOWN, STRIP_WIDTH, SQUARE);
    const grey = (square: number) => {
      for (let y = 0; y < SQUARE; y++)
        for (let x = square * SQUARE; x < (square + 1) * SQUARE; x++)
          img.fill(128, (y * STRIP_WIDTH + x) * 4, (y * STRIP_WIDTH + x) * 4 + 3);
    };
    grey(3);
    expect(decodeImage(img, STRIP_WIDTH)).toMatchObject({ unsure: 1, valid: true });
    grey(20);
    expect(decodeImage(img, STRIP_WIDTH)).toMatchObject({ unsure: 2, valid: false });
  });

  it("tolerates compression noise below the margin", () => {
    const img = encodeImage(KNOWN, STRIP_WIDTH, SQUARE);
    for (let o = 0; o < img.length; o += 4) {
      const n = ((o * 7919) % 41) - 20; // ±20 deterministic noise
      for (let c = 0; c < 3; c++) img[o + c] = (img[o + c] ?? 0) + n;
    }
    expect(decodeImage(img, STRIP_WIDTH)).toEqual({ ms: low48(KNOWN), unsure: 0, valid: true });
  });

  it("an empty picture is invalid", () => {
    const rows = new Array(STRIP_WIDTH * 8 * 4).fill(128);
    expect(decodeRows(rows, STRIP_WIDTH).valid).toBe(false);
  });

  it("median", () => {
    expect(median([3, 1, 2])).toBe(2);
    expect(median([4, 1, 2, 3])).toBe(2.5);
  });
});
