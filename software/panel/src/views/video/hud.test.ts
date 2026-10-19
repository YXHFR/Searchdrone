import { describe, expect, it } from "vitest";
import type { Telemetry } from "../../api/types";
import { heading } from "../../header/instruments";
import { horizon, tapeTicks } from "./hud";

const tel = (attitude: Telemetry["attitude"]) => ({ attitude }) as Telemetry;

describe("heading tape", () => {
  it("wraps through north and names the cardinals", () => {
    const ticks = tapeTicks(355, 20);
    expect(ticks[0]).toMatchObject({ deg: 335, offset: -20 });
    expect(ticks.find((t) => t.deg === 0)).toMatchObject({ offset: 5, label: "N", major: true });
    expect(ticks.find((t) => t.deg === 5)?.label).toBeNull();
    expect(ticks.at(-1)).toMatchObject({ deg: 15, offset: 20 });
  });

  it("labels every 30 degrees in tens", () => {
    expect(tapeTicks(60, 0)).toEqual([{ deg: 60, offset: 0, label: "06", major: true }]);
    expect(tapeTicks(90, 0)[0]?.label).toBe("O");
  });

  it("formats the heading with its compass point", () => {
    expect(heading(72.4)).toBe("072° O");
    expect(heading(-1)).toBe("359° N");
  });
});

describe("horizon", () => {
  it("draws the drone's own attitude", () => {
    expect(horizon(tel({ roll_deg: 4, pitch_deg: -2, yaw_deg: 70 }), 200)).toEqual({
      roll: 4,
      pitch: -2,
    });
  });

  it("is left out without an attitude or with old data", () => {
    expect(horizon(tel(null), 200)).toBeNull();
    expect(horizon(tel(undefined), 200)).toBeNull();
    expect(horizon(tel({ roll_deg: 4, pitch_deg: -2, yaw_deg: 70 }), 9_000)).toBeNull();
    expect(horizon(null, null)).toBeNull();
  });
});
