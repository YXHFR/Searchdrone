import { describe, expect, it } from "vitest";
import type { Find } from "../../api/types";
import { findsData } from "./mapData";

const base: Find = {
  find_id: "f1",
  label: "red_block",
  name_sv: "rött block",
  kind: "target",
  source: "color",
  lat: -35.3615,
  lon: 149.1652,
  sigma95_m: 4,
  score: 0.71,
  count: 1,
  thumb_detection_id: null,
  synthetic: true,
};

const label = (f: Find): string =>
  String(findsData([f]).features[0]?.properties.label).replace(/\s/g, " ");

describe("finds on the map", () => {
  it("one point per find, at its place, with its kind", () => {
    const fc = findsData([base, { ...base, find_id: "f2", kind: "context" }]);
    expect(fc.features).toHaveLength(2);
    expect(fc.features[0]?.geometry.coordinates).toEqual([149.1652, -35.3615]);
    expect(fc.features[1]?.properties).toMatchObject({ id: "f2", kind: "context" });
  });

  it("the label names the class, the score and how often it was seen", () => {
    expect(label(base)).toBe("rött block 71 %");
    expect(label({ ...base, count: 4 })).toBe("rött block 71 % ×4");
  });

  it("the simulator's labels say facit, never a score", () => {
    expect(label({ ...base, name_sv: "bil", source: "oracle", score: 1 })).toBe("bil (facit)");
  });
});
