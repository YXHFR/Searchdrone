import { afterEach, describe, expect, it, vi } from "vitest";
import { computeBottom } from "./useVisualViewportBottom";

function mockTouch(coarse: boolean, vv: { height: number; offsetTop: number } | undefined): void {
  vi.stubGlobal("matchMedia", (q: string) => ({ matches: coarse && q === "(pointer: coarse)" }));
  Object.defineProperty(window, "visualViewport", { configurable: true, value: vv });
  Object.defineProperty(window, "innerHeight", { configurable: true, value: 800 });
}

afterEach(() => vi.unstubAllGlobals());

describe("computeBottom (safety bar above the virtual keyboard)", () => {
  it("lifts the bar by the keyboard height on touch screens", () => {
    mockTouch(true, { height: 400, offsetTop: 0 });
    expect(computeBottom()).toBe(400);
  });
  it("is 0 when the keyboard closes", () => {
    mockTouch(true, { height: 800, offsetTop: 0 });
    expect(computeBottom()).toBe(0);
  });
  it("is 0 without visualViewport or without a touch screen", () => {
    mockTouch(true, undefined);
    expect(computeBottom()).toBe(0);
    mockTouch(false, { height: 400, offsetTop: 0 });
    expect(computeBottom()).toBe(0);
  });
});
