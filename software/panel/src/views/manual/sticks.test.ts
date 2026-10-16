import { describe, expect, it } from "vitest";
import {
  axesFromGamepad,
  axesFromSticks,
  clamp1,
  deadzone,
  NEUTRAL,
  stickFromPointer,
} from "./sticks";

describe("manual control helpers", () => {
  it("clamps and drops NaN", () => {
    expect(clamp1(3)).toBe(1);
    expect(clamp1(-3)).toBe(-1);
    expect(clamp1(Number.NaN)).toBe(0);
  });

  it("applies a deadzone and keeps full travel", () => {
    expect(deadzone(0.05)).toBe(0);
    expect(deadzone(-0.05)).toBe(0);
    expect(deadzone(1)).toBe(1);
    expect(deadzone(-1)).toBe(-1);
    expect(deadzone(0.54)).toBeCloseTo(0.5, 6);
  });

  it("maps pointer offsets to a clamped stick (y up)", () => {
    expect(stickFromPointer(0, -50, 50)).toEqual({ x: 0, y: 1 });
    const far = stickFromPointer(500, 500, 50);
    expect(Math.hypot(far.x, far.y)).toBeLessThanOrEqual(1 + 1e-9);
    expect(far.y).toBeLessThan(0);
    expect(stickFromPointer(10, 10, 0)).toEqual({ x: 0, y: 0 });
  });

  it("released sticks give neutral (hold position)", () => {
    expect(axesFromSticks({ x: 0, y: 0 }, { x: 0, y: 0 })).toEqual(NEUTRAL);
    expect(axesFromGamepad([0.02, -0.03, 0.01, 0])).toEqual({
      roll: 0,
      pitch: 0,
      yaw: 0,
      throttle: 0,
    });
  });

  it("reads a standard gamepad (mode 2)", () => {
    const a = axesFromGamepad([1, -1, -1, 1]);
    expect(a).toEqual({ yaw: 1, throttle: 1, roll: -1, pitch: -1 });
  });
});
