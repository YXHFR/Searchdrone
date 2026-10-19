import { describe, expect, it } from "vitest";
import type { StateSnapshot } from "../api/types";
import { linkQuality, vehiclePill } from "./pills";

const snap = (p: Partial<StateSnapshot>): StateSnapshot =>
  ({ executor_state: "IDLE", held_by: null, ...p }) as StateSnapshot;

describe("vehiclePill", () => {
  it("covers the executor states", () => {
    expect(vehiclePill(null).key).toBe("pills.vehicle.unknown");
    expect(vehiclePill(snap({ executor_state: "SURVEY" })).key).toBe("pills.vehicle.survey");
    expect(vehiclePill(snap({ executor_state: "HELD", held_by: "operator_stop" })).key).toBe(
      "pills.vehicle.stopped",
    );
    expect(vehiclePill(snap({ executor_state: "HELD", held_by: "operator_home" })).tone).toBe(
      "orange",
    );
    expect(vehiclePill(snap({ executor_state: "HELD", held_by: "fc_silent" })).key).toBe(
      "pills.vehicle.humanTookOver",
    );
    expect(vehiclePill(snap({ executor_state: "FAILSAFE" })).tone).toBe("red");
    expect(vehiclePill(snap({ executor_state: "HOLD", hold_reason: "find_hold" })).key).toBe(
      "pills.vehicle.findHold",
    );
  });
});

describe("linkQuality", () => {
  const up = { mavlink: "up", sdlink: "up", onboard_heartbeat_age_ms: 100, rtt_ms: 20 } as const;
  it("takes the worst of server and telemetry age", () => {
    expect(linkQuality(up, 500)).toBe("good");
    expect(linkQuality(up, 2_000)).toBe("weak");
    expect(linkQuality(up, 3_500)).toBe("lost");
    expect(linkQuality({ ...up, sdlink: "degraded" }, 100)).toBe("weak");
    expect(linkQuality({ ...up, mavlink: "lost" }, 100)).toBe("lost");
    expect(linkQuality(null, 100)).toBe("lost");
    expect(linkQuality(up, null)).toBe("lost");
  });
});
