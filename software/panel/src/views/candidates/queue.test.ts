import { describe, expect, it } from "vitest";
import { candidatesData } from "../map/mapData";
import { card } from "./fixtures";
import { ageMinutes, crops, fieldLabel, findNumbers, groupQueue } from "./queue";

describe("candidate queue", () => {
  it("splits open cards by priority class and leaves merged ones out", () => {
    const q = groupQueue([
      card({ seq: 1, priority: 0.5, priority_class: "medium" }),
      card({ seq: 2, priority: 0.9 }),
      card({ seq: 3, priority: 0.1, priority_class: "low", state: "new" }),
      card({ seq: 4, state: "confirmed" }),
      card({ seq: 5, state: "rejected" }),
      card({ seq: 6, state: "rejected" }),
      card({ seq: 7, state: "merged" }),
    ]);
    expect(q.review.map((c) => c.seq)).toEqual([2, 1]);
    expect(q.low.map((c) => c.seq)).toEqual([3]);
    expect(q.confirmed.map((c) => c.seq)).toEqual([4]);
    expect(q.rejected.map((c) => c.seq)).toEqual([6, 5]);
  });

  it("numbers confirmed finds in candidate order", () => {
    const f = findNumbers([
      card({ seq: 9, state: "confirmed" }),
      card({ seq: 3, state: "confirmed" }),
      card({ seq: 4 }),
    ]);
    expect(f.get("c-3")).toBe(1);
    expect(f.get("c-9")).toBe(2);
    expect(f.has("c-4")).toBe(false);
  });

  it("names fields, orders pictures and counts the age", () => {
    expect(fieldLabel("garment:0")).toEqual({ key: "garment", n: 1 });
    expect(fieldLabel("hair")).toEqual({ key: "hair" });
    const row = { distance_m: 1, label: "person", lat: 0, lon: 0, px_size: 40, score: 0.5 };
    const pics = crops([
      {
        ...row,
        detection_id: "b",
        captured_utc: "2026-01-01T10:02:00Z",
        has_crop: true,
        sigma95_m: 5,
      },
      {
        ...row,
        detection_id: "a",
        captured_utc: "2026-01-01T10:01:00Z",
        has_crop: true,
        sigma95_m: 5,
      },
      {
        ...row,
        detection_id: "c",
        captured_utc: "2026-01-01T10:00:00Z",
        has_crop: false,
        sigma95_m: 5,
      },
    ]);
    expect(pics.map((p) => p.detection_id)).toEqual(["a", "b"]);
    expect(ageMinutes(card(), Date.parse("2026-01-01T10:05:30Z"))).toBe(5);
    expect(ageMinutes(card(), Date.parse("2026-01-01T09:00:00Z"))).toBe(0);
  });

  it("draws open candidates and finds on the map with their circle, not rejected ones", () => {
    const cards = [
      card({ seq: 1 }),
      card({ seq: 2, state: "confirmed" }),
      card({ seq: 3, state: "rejected" }),
    ];
    const fc = candidatesData(cards, findNumbers(cards));
    const labels = fc.features
      .filter((f) => f.geometry.type === "Point")
      .map((f) => f.properties.label);
    expect(labels).toEqual(["K1", "F1"]);
    expect(fc.features.filter((f) => f.geometry.type === "Polygon")).toHaveLength(2);
  });
});
