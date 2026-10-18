import { describe, expect, it } from "vitest";
import { square } from "../../e2e/fixtures";
import type { SectorFeature } from "../state/missionStore";
import { coverage, doneCount, sectorRows } from "./mission";

const sector = (
  id: string,
  status: string,
  progress: number,
  extra: Record<string, unknown> = {},
): SectorFeature => ({
  type: "Feature",
  geometry: square(50),
  properties: { sector_id: id, name: id, status, progress, ...extra },
});

describe("sector rows", () => {
  it("rounds progress to whole percent and clamps it", () => {
    const rows = sectorRows([
      sector("S1", "done", 1),
      sector("S2", "active", 0.404),
      sector("S3", "pending", 1.4),
    ]);
    expect(rows.map((r) => r.pct)).toEqual([100, 40, 100]);
    expect(doneCount(rows)).toBe(1);
  });

  it("reads an unknown status as pending rather than inventing one", () => {
    expect(sectorRows([sector("S1", "weird", 0)])[0]?.status).toBe("pending");
    expect(sectorRows([sector("S1", "skipped", 0)])[0]?.status).toBe("skipped");
  });
});

describe("coverage", () => {
  it("is nothing before there are sectors", () => {
    expect(coverage([])).toBeNull();
  });

  it("sums the reported footprint against the sectors' area", () => {
    const c = coverage([
      sector("S1", "done", 1, { area_m2: 40_000, covered: square(50) }),
      sector("S2", "active", 0.4, { area_m2: 60_000 }),
    ]);
    expect(c?.totalHa).toBeCloseTo(10, 5);
    expect(c?.sweptHa).toBeGreaterThan(0.9);
    expect(c?.sweptHa).toBeLessThan(1.1);
  });

  it("measures the sector when the server sent no area", () => {
    const c = coverage([sector("S1", "pending", 0)]);
    expect(c?.totalHa).toBeGreaterThan(0.9);
    expect(c?.sweptHa).toBe(0);
  });
});
