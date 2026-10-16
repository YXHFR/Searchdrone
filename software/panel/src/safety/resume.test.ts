import { describe, expect, it } from "vitest";
import type { StateSnapshot, VehicleLive } from "../api/types";
import { resumeView } from "./resume";

const live = (
  snapshot: Partial<StateSnapshot> | null,
  latch: VehicleLive["latch"] = null,
): VehicleLive =>
  ({
    vehicle_id: "sim-01",
    airborne: true,
    in_flight: true,
    latch,
    link: { mavlink: "up", sdlink: "up", onboard_heartbeat_age_ms: 100, rtt_ms: 10 },
    telemetry: null,
    snapshot: snapshot
      ? ({
          executor_state: "HELD",
          held_by: null,
          resume_available: false,
          ...snapshot,
        } as StateSnapshot)
      : null,
  }) as VehicleLive;

describe("resumeView", () => {
  it("HELD + operator_stop + resume_available → button", () => {
    expect(resumeView(live({ held_by: "operator_stop", resume_available: true }))).toBe("button");
  });
  it("other interruptions with resume_available also show the button", () => {
    expect(resumeView(live({ held_by: "onboard_restart", resume_available: true }))).toBe("button");
    expect(resumeView(live({ held_by: "external", resume_available: true }))).toBe("button");
  });
  it("return home started by the operator never offers Resume", () => {
    expect(resumeView(live({ held_by: "operator_home", resume_available: true }))).toBe(
      "home_active",
    );
  });
  it("no resume_available, other states or no vehicle → none", () => {
    expect(resumeView(live({ held_by: "fc_silent", resume_available: false }))).toBe("none");
    expect(resumeView(live({ executor_state: "SURVEY", resume_available: true }))).toBe("none");
    expect(resumeView(null)).toBe("none");
  });
  it("snapshot missing → no button, even with a stop latch", () => {
    const latch = {
      action: "stop",
      action_seq: 1,
      set_at: "",
      set_by: "",
      set_mono_ms: 0,
      vehicle_id: "sim-01",
    } as const;
    expect(resumeView(live(null, latch))).toBe("onboard_silent");
    expect(resumeView(live(null))).toBe("none");
  });
});
