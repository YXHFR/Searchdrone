import { describe, expect, it } from "vitest";
import { decimalText, dmsText, sweref99tm, swerefText } from "./coords";

describe("coordinate formats", () => {
  it("matches PROJ's EPSG:3006 at the public simulator site to the millimetre", () => {
    // Reference values from pyproj (EPSG:4326 -> EPSG:3006).
    const a = sweref99tm(59.28842, 18.15953);
    expect(a.n).toBeCloseTo(6_576_436.044, 2);
    expect(a.e).toBeCloseTo(679_958.3, 2);
    const b = sweref99tm(59.2912, 18.1641);
    expect(b.n).toBeCloseTo(6_576_757.742, 2);
    expect(b.e).toBeCloseTo(680_203.766, 2);
    expect(swerefText(59.28842, 18.15953)).toBe("N 6576436 E 679958");
    expect(swerefText(-35.363261, 149.16523)).toBeNull();
  });

  it("writes decimal degrees and DMS with hemisphere letters", () => {
    expect(decimalText(59.28842, 18.15953)).toBe("59.288420, 18.159530");
    expect(dmsText(59.28842, 18.15953)).toBe("59°17′18.3″N 18°09′34.3″E");
    expect(dmsText(-35.363261, 149.16523)).toBe("35°21′47.7″S 149°09′54.8″E");
  });

  it("carries a rounded 60 seconds into the minute", () => {
    expect(dmsText(59.28842, 18.1666661)).toBe("59°17′18.3″N 18°10′00.0″E");
  });
});
