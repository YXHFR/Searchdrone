import { describe, expect, it } from "vitest";
import type { MissionCharter } from "../../../api/types";
import { changedFromDefault, charterDiff, charterProblems, limitProblems, looser } from "./limits";

const ABS = {
  max_alt_agl_m: 120,
  max_speed_ms: 15,
  max_range_from_pilot_m: 500,
  min_battery_rtl_pct: 15,
  min_alt_agl_m: 5,
  max_area_km2: 1,
};
const HOME = { lat: -35.363261, lon: 149.16523 };
const north = (m: number) => ({ lat: HOME.lat + m / 111_320, lon: HOME.lon });
const sq = {
  type: "Polygon",
  coordinates: [
    [
      [149.165, -35.36],
      [149.166, -35.36],
      [149.166, -35.361],
      [149.165, -35.361],
      [149.165, -35.36],
    ],
  ],
};
const charter = (p: Partial<MissionCharter> = {}): MissionCharter =>
  ({
    area: sq,
    home: { ...HOME, alt_amsl_m: 584 },
    pilot_position: HOME,
    alt_min_agl_m: 15,
    alt_max_agl_m: 60,
    max_speed_ms: 8,
    max_range_from_pilot_m: 500,
    battery_rtl_pct: 15,
    allowed_skills: ["goto"],
    sectors: [],
    ...p,
  }) as MissionCharter;
const fields = (c: MissionCharter, home = HOME) =>
  charterProblems(c, ABS, home).map((p) => p.field);

describe("limit checks against the server's numbers", () => {
  it("each row of the table turns red", () => {
    expect(fields(charter({ alt_max_agl_m: 121 }))).toContain("max_alt_agl_m");
    expect(fields(charter({ max_speed_ms: 16 }))).toContain("max_speed_ms");
    expect(fields(charter({ max_range_from_pilot_m: 501 }))).toContain("max_range_from_pilot_m");
    expect(fields(charter({ battery_rtl_pct: 10 }))).toContain("battery_rtl_pct");
    expect(fields(charter({ alt_min_agl_m: 4 }))).toContain("alt_min_agl_m");
    expect(fields(charter({ alt_min_agl_m: 60 }))).toContain("alt_min_agl_m");
    expect(fields(charter({ pilot_position: north(11) }))).toContain("pilot_position");
    expect(fields(charter(), north(11))).toContain("home");
    expect(fields(charter())).toEqual([]);
  });

  it("the cap is the server's: 150 m is red under 120 and fine (looser) under 200", () => {
    expect(fields(charter({ alt_max_agl_m: 110 }))).toEqual([]);
    expect(
      charterProblems(charter({ alt_max_agl_m: 110 }), { ...ABS, max_alt_agl_m: 100 }, HOME).map(
        (p) => p.field,
      ),
    ).toContain("max_alt_agl_m");
    expect(
      charterProblems(charter({ alt_max_agl_m: 150 }), { ...ABS, max_alt_agl_m: 200 }, HOME),
    ).toEqual([]);
    expect(changedFromDefault("max_alt_agl_m", 150, { max_alt_agl_m: 120 })).toBe("looser");
    expect(changedFromDefault("battery_rtl_pct", 20, { battery_rtl_pct: 15 })).toBe("tighter");
    expect(changedFromDefault("max_speed_ms", 15, { max_speed_ms: 15 })).toBeNull();
  });

  it("the limits form: above the cap and below the battery floor", () => {
    expect(limitProblems({ max_alt_agl_m: 130 }, ABS).map((p) => p.key)).toEqual([
      "limits.aboveCap",
    ]);
    expect(limitProblems({ battery_rtl_pct: 10 }, ABS).map((p) => p.key)).toEqual([
      "limits.belowFloor",
    ]);
    expect(limitProblems({ max_alt_agl_m: 100, battery_rtl_pct: 20 }, ABS)).toEqual([]);
  });
});

describe("what counts as looser", () => {
  it("one case per field", () => {
    expect(looser("alt_max_agl_m", 60, 80)).toBe(true);
    expect(looser("max_speed_ms", 8, 6)).toBe(false);
    expect(looser("alt_min_agl_m", 15, 10)).toBe(true);
    expect(looser("battery_rtl_pct", 15, 20)).toBe(false);
    expect(looser("geofence_buffer_m", 20, 50)).toBe(true);
    expect(looser("ai_silent_timeout_s", 20, 30)).toBe(true);
    expect(looser("on_ai_silent", "continue_plan", "return_home")).toBe(true);
    expect(looser("on_ai_silent", "continue_plan", "hold")).toBe(false);
    expect(looser("allowed_skills", ["goto"], ["goto", "orbit"])).toBe(true);
    expect(looser("allowed_skills", ["goto", "orbit"], ["goto"])).toBe(false);
    expect(looser("home", HOME, north(4))).toBe(false);
    expect(looser("pilot_position", HOME, north(6))).toBe(true);
    expect(looser("unknown_thing", 1, 2)).toBeNull();
  });

  it("the diff lists every changed field, unknown ones too, never the server's own", () => {
    const a = charter({ version: 1 } as Partial<MissionCharter>);
    const b = {
      ...charter({ geofence_buffer_m: 50, version: 2 } as Partial<MissionCharter>),
      new_field: 1,
    } as MissionCharter;
    const rows = charterDiff(a, b);
    expect(rows.map((r) => r.field)).toEqual(["geofence_buffer_m", "new_field"]);
    expect(rows.find((r) => r.field === "new_field")?.known).toBe(false);
  });
});
