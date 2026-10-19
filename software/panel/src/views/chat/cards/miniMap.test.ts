import { describe, expect, it } from "vitest";
import { fit, outerRing } from "./miniMap";

describe("mini map fit", () => {
  it("draws nothing without points", () => {
    expect(fit([], 300, 120, 10)).toBeNull();
  });

  it("keeps every point inside the padded box, north up", () => {
    const pts = [
      [18.0, 59.0],
      [18.01, 59.005],
    ];
    const f = fit(pts, 300, 120, 10);
    if (!f) throw new Error("no fit");
    for (const [lon, lat] of pts) {
      expect(f.x(lon ?? 0)).toBeGreaterThanOrEqual(10 - 1e-9);
      expect(f.x(lon ?? 0)).toBeLessThanOrEqual(290 + 1e-9);
      expect(f.y(lat ?? 0)).toBeGreaterThanOrEqual(10 - 1e-9);
      expect(f.y(lat ?? 0)).toBeLessThanOrEqual(110 + 1e-9);
    }
    expect(f.y(59.005)).toBeLessThan(f.y(59.0));
    expect(f.scaleM / f.mpp).toBeGreaterThanOrEqual(40);
  });

  it("does not blow a single point up to street level", () => {
    expect(fit([[18, 59]], 300, 120, 10)?.mpp).toBe(0.5);
  });

  it("reads the outer ring of a polygon or the first of a multipolygon", () => {
    const ring = [
      [0, 0],
      [1, 0],
      [1, 1],
      [0, 0],
    ];
    expect(outerRing({ type: "Polygon", coordinates: [ring] })).toEqual(ring);
    expect(outerRing({ type: "MultiPolygon", coordinates: [[ring]] })).toEqual(ring);
    expect(outerRing({ type: "Point", coordinates: [0, 0] })).toBeNull();
  });
});
