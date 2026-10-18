import { describe, expect, it } from "vitest";
import type { Annotation, Command, StateSnapshot } from "../../api/types";
import {
  activePath,
  formatLength,
  leavesArea,
  lengthM,
  nextPathLabel,
  orderPath,
  pathFlownData,
  pathPlanData,
  pathProblem,
  pathsData,
} from "./pathData";

const HOME = { lat: -35.363261, lon: 149.16523 };
const M = 1 / 111_320; // degrees per metre north
const east = (m: number) => m / (111_320 * Math.cos((HOME.lat * Math.PI) / 180));
const at = (e: number, n: number): [number, number] => [HOME.lon + east(e), HOME.lat + n * M];
const square = (half: number) => ({
  type: "Polygon",
  coordinates: [
    [at(-half, -half), at(half, -half), at(half, half), at(-half, half), at(-half, -half)],
  ],
});

// An L-shaped 100 m line: 100 m north, then 100 m east.
const L = [at(0, 0), at(0, 100), at(100, 100)];

function cmd(status: string, points = L, id = "c1"): Command {
  return {
    id,
    skill: "follow_path",
    status,
    params: { points: points.map(([lon, lat]) => ({ lat, lon })), alt_m: 30, label: "pa-1" },
    created_at: "2026-10-18T12:00:00Z",
  } as unknown as Command;
}

describe("drawn paths", () => {
  it("measures and names the line", () => {
    expect(lengthM(L)).toBeCloseTo(200, -0.5);
    expect(formatLength(420.4)).toBe("420 m");
    expect(formatLength(1_234)).toBe("1,2 km");
    const anns = [{ label: "pa-1" }, { label: "pa-3" }, { label: "P2" }] as Annotation[];
    expect(nextPathLabel(anns)).toBe("pa-4");
    expect(nextPathLabel([])).toBe("pa-1");
  });

  it("shows each drawn path dashed with its name and length at the start", () => {
    const a = {
      id: "a1",
      kind: "path",
      label: "pa-1",
      geometry: { type: "LineString", coordinates: L },
      length_m: 200,
    } as unknown as Annotation;
    const f = pathsData([a, { ...a, kind: "area" } as Annotation]).features;
    expect(f.map((x) => x.properties.kind)).toEqual(["line", "start"]);
    expect(f[0]?.properties.label).toBe("pa-1 · 200 m");
  });

  it("refuses a path that leaves the charter, is too long or too short", () => {
    const area = square(150);
    expect(pathProblem(L, area)).toBeNull();
    expect(pathProblem(L, null)).toBeNull(); // no charter yet: the gate decides
    expect(pathProblem([at(0, 0)], area)).toEqual({ key: "draw.pathTooShort" });
    expect(pathProblem([at(0, 0), at(0, 200)], area)).toEqual({ key: "draw.pathOutside" });
    // Both ends inside, the leg cuts across a corner of the concave area.
    const notch = {
      type: "Polygon",
      coordinates: [
        [
          at(-100, -100),
          at(100, -100),
          at(100, 0),
          at(0, 0),
          at(0, 100),
          at(-100, 100),
          at(-100, -100),
        ],
      ],
    };
    expect(leavesArea([at(50, -50), at(-50, 50)], notch)).toBe(false);
    expect(leavesArea([at(80, -20), at(-20, 80)], notch)).toBe(true);
    const zigzag = Array.from({ length: 20 }, (_, i) => at(i % 2 ? -140 : 140, -140 + 15 * i));
    expect(pathProblem(zigzag, area)).toMatchObject({ key: "draw.pathTooLong", max: "5 km" });
    const many = Array.from({ length: 51 }, (_, i) => at(i, 0));
    expect(pathProblem(many, area)).toEqual({ key: "draw.pathTooManyPoints", max: 50 });
  });
});

describe("the path being flown", () => {
  it("reads a follow_path order's points and nothing else", () => {
    expect(orderPath(cmd("running"))).toHaveLength(3);
    expect(orderPath({ ...cmd("running"), skill: "goto" } as Command)).toBeNull();
  });

  it("a pending card shows the plan dashed, nothing flown, 0 %", () => {
    const p = activePath(null, [cmd("pending_approval")], null, null, null);
    expect(p?.running).toBe(false);
    expect(p?.pct).toBe(0);
    expect(pathPlanData(p).features.map((f) => f.properties.n ?? f.properties.kind)).toEqual([
      "line",
      "1",
      "2",
      "3",
    ]);
    expect(pathFlownData(p, null).features).toEqual([]);
  });

  it("progress comes from the drone's place on the line, or the onboard snapshot", () => {
    const running = cmd("running");
    const drone = { lon: at(0, 50)[0], lat: at(0, 50)[1] };
    const p = activePath(running, [], running, drone, null);
    expect(p?.pct).toBeCloseTo(25, 0);
    const flown = pathFlownData(p, drone).features[0]?.geometry.coordinates as number[][];
    expect(flown).toHaveLength(2);
    expect(flown[1]?.[1]).toBeCloseTo(drone.lat, 6);
    const snap = { active_command_id: "c1", progress_pct: 60 } as StateSnapshot;
    expect(activePath(running, [], running, drone, snap)?.pct).toBe(60);
    // Past the corner: the flown line runs through it.
    const later = { lon: at(50, 100)[0], lat: at(50, 100)[1] };
    expect(
      pathFlownData(activePath(running, [], running, later, null), later).features[0]?.geometry
        .coordinates,
    ).toHaveLength(3);
  });

  it("after the end the whole line is flown, 100 %, until another move starts", () => {
    const done = cmd("done");
    const p = activePath(null, [], done, null, null);
    expect(p?.pct).toBe(100);
    expect(pathFlownData(p, null).features).toHaveLength(1);
    const goto = { ...cmd("running", L, "c2"), skill: "goto" } as Command;
    expect(activePath(goto, [], goto, null, null)).toBeNull();
  });
});
