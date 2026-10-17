import { describe, expect, it } from "vitest";
import type { Annotation, Charter, Command, Telemetry } from "../../api/types";
import { areaM2, distanceM, selfIntersects } from "../../lib/geo";
import {
  droneData,
  nextLabel,
  pilotRangeData,
  rectangle,
  sectorsData,
  targetData,
  trackData,
} from "./mapData";

const HOME = { lat: -35.363261, lon: 149.16523 };

describe("map data", () => {
  it("the drone is one point, marked stale after 3 s", () => {
    const t = { lat: HOME.lat, lon: HOME.lon, heading_deg: 10, alt_rel_home_m: 41.6 } as Telemetry;
    expect(droneData(t, 500).features[0]?.properties).toMatchObject({ stale: false, alt: 42 });
    expect(droneData(t, 3_500).features[0]?.properties.stale).toBe(true);
    expect(droneData(null, 0).features).toEqual([]);
  });

  it("the pilot range is the pilot point and a 500 m circle around it", () => {
    const c = { body: { pilot_position: HOME, max_range_from_pilot_m: 500 } } as unknown as Charter;
    const f = pilotRangeData(c).features;
    expect(f).toHaveLength(2);
    const ring = ((f[1]?.geometry.coordinates ?? []) as number[][][])[0] ?? [];
    for (const p of ring)
      expect(Math.abs(distanceM(HOME, { lon: p[0] ?? 0, lat: p[1] ?? 0 }) - 500)).toBeLessThan(5);
  });

  it("the target is the running order, else a pending card's point", () => {
    const run = { skill: "goto", params: { lat: 1, lon: 2 } } as unknown as Command;
    const card = { skill: "goto", params: { lat: 3, lon: 4 } } as unknown as Command;
    expect(targetData(run, [card]).features[0]?.geometry.coordinates).toEqual([2, 1]);
    expect(targetData(null, [card]).features[0]?.geometry.coordinates).toEqual([4, 3]);
    expect(targetData(null, []).features).toEqual([]);
  });

  it("the track needs two points", () => {
    expect(trackData([[1, 2]]).features).toEqual([]);
    expect(
      trackData([
        [1, 2],
        [1, 3],
      ]).features,
    ).toHaveLength(1);
  });

  it("a sector's label shows its progress (a fraction from the ground) in per cent", () => {
    const fc = sectorsData([
      {
        type: "Feature",
        geometry: {
          type: "Polygon",
          coordinates: [
            [
              [0, 0],
              [0, 1],
              [1, 1],
              [0, 0],
            ],
          ],
        },
        properties: { sector_id: "A", name: "A", status: "active", progress: 0.476 },
      },
    ]);
    expect(fc.features[0]?.properties?.label).toBe("A · 48 %");
  });

  it("labels: the next free P<n> and ar-<n>", () => {
    const a = (label: string) => ({ label }) as Annotation;
    expect(nextLabel([a("P1"), a("P3"), a("ar-1")], "pin")).toBe("P4");
    expect(nextLabel([a("P1")], "area")).toBe("ar-1");
  });

  it("geometry: rectangle, area and self-crossing", () => {
    const r = rectangle([0, 0], [0.001, 0.001]);
    expect(r).toHaveLength(4);
    const ha =
      areaM2({ type: "Polygon", coordinates: [[...r, r[0] as [number, number]]] }) / 10_000;
    expect(ha).toBeGreaterThan(1.2);
    expect(ha).toBeLessThan(1.25);
    expect(
      selfIntersects([
        [0, 0],
        [1, 1],
        [1, 0],
        [0, 1],
        [0, 0],
      ]),
    ).toBe(true);
    expect(
      selfIntersects([
        [0, 0],
        [1, 0],
        [1, 1],
        [0, 1],
        [0, 0],
      ]),
    ).toBe(false);
  });
});
