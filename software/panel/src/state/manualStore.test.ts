import { describe, expect, it } from "vitest";
import type { ManualControlState } from "../api/types";
import { manualView, TAKEOVER_GRACE_MS } from "./manualStore";

const held = (id: string): ManualControlState => ({
  active: true,
  takeover_id: id,
  by: "operator:demo",
  since: "2026-01-01T00:00:00Z",
  input_age_ms: 50,
});
const nobody: ManualControlState = {
  active: false,
  takeover_id: null,
  by: null,
  since: null,
  input_age_ms: null,
};

describe("manual control state", () => {
  it("we steer while the ground holds our take-over", () => {
    expect(manualView(held("a1"), "a1", 0, 10_000)).toEqual({
      steering: true,
      otherActive: false,
      oursEnded: false,
    });
  });

  it("right after the take-over a stale 'nobody' does not end it", () => {
    expect(manualView(nobody, "a1", 1_000, 1_000 + TAKEOVER_GRACE_MS - 1).steering).toBe(true);
  });

  it("the ground ending it (STOP, FLY HOME, landing) ends ours", () => {
    expect(manualView(nobody, "a1", 0, 10_000)).toEqual({
      steering: false,
      otherActive: false,
      oursEnded: true,
    });
  });

  it("someone else's take-over (or one from before a reload) is not ours", () => {
    expect(manualView(held("b2"), null, 0, 10_000)).toEqual({
      steering: false,
      otherActive: true,
      oursEnded: false,
    });
    expect(manualView(held("b2"), "a1", 0, 10_000).oursEnded).toBe(true);
  });

  it("nothing known yet: no take-over shown", () => {
    expect(manualView(null, null, 0, 10_000)).toEqual({
      steering: false,
      otherActive: false,
      oursEnded: false,
    });
  });
});
