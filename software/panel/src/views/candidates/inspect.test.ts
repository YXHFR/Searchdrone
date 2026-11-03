// @vitest-environment node
import { beforeEach, describe, expect, it } from "vitest";
import type { Inspection } from "../../api/types";
import { openLook, useInspectStore } from "../../state/inspectStore";
import { inspectBlock, leftMs, lookKind, operationMode, stripFloor } from "./inspect";

const K3 = { candidate_id: "c3", state: "ready" as const };

function look(over: Partial<Inspection> = {}): Inspection {
  return {
    command_id: "cmd1",
    candidate_id: "c3",
    mission_id: "M1",
    seq: 3,
    started_at: "2026-11-03T10:00:00Z",
    started_by: "operator:demo",
    operation_mode: "exercise",
    overflight: true,
    phase: "flying",
    ...over,
  };
}

describe("inspectBlock", () => {
  const ok = { isPilot: true, card: K3, look: null, airborne: true, sdlink: "up" as const };

  it("lets the pilot press it on a ready or assessing card in the air with a link", () => {
    expect(inspectBlock(ok)).toBeNull();
    expect(inspectBlock({ ...ok, card: { ...K3, state: "assessing" } })).toBeNull();
    expect(inspectBlock({ ...ok, sdlink: "degraded" })).toBeNull();
  });

  it("names why it is grey, the pilot's role first", () => {
    expect(inspectBlock({ ...ok, isPilot: false, sdlink: "lost" })).toBe("notPilot");
    expect(inspectBlock({ ...ok, look: look({ candidate_id: "c4" }) })).toBe("busy");
    expect(inspectBlock({ ...ok, card: { ...K3, state: "new" } })).toBe("notOpen");
    expect(inspectBlock({ ...ok, card: { ...K3, state: "confirmed" } })).toBe("notOpen");
    expect(inspectBlock({ ...ok, sdlink: "lost" })).toBe("noLink");
    expect(inspectBlock({ ...ok, sdlink: null })).toBe("noLink");
    expect(inspectBlock({ ...ok, airborne: false })).toBe("notFlying");
  });
});

describe("the charter's mode", () => {
  it("reads an unset mode as live, and no charter as none", () => {
    expect(operationMode(null)).toBeNull();
    expect(operationMode({ operation_mode: null })).toBe("live");
    expect(operationMode({ operation_mode: "exercise" })).toBe("exercise");
  });

  it("flies over only with the ground's stamp", () => {
    expect(lookKind({ operation_mode: "exercise", overflight_allowed: true })).toBe("over");
    expect(lookKind({ operation_mode: "live", overflight_allowed: false })).toBe("side");
    expect(lookKind(null)).toBe("side");
  });
});

describe("the strip and the clocks", () => {
  const p = {
    command_id: "cmd1",
    candidate_id: "c3",
    step: 1,
    phase: "step" as const,
  };

  it("shows the lowest floor when known, else the automatic one", () => {
    expect(stripFloor(null)).toBeNull();
    expect(stripFloor({ ...p, floor_auto_above_ground_m: 30.2 })).toBe(30);
    expect(
      stripFloor({ ...p, floor_auto_above_ground_m: 30, floor_lowest_above_ground_m: 20.6 }),
    ).toBe(21);
  });

  it("counts down to an ISO time and never below zero", () => {
    const now = Date.parse("2026-11-03T10:00:00Z");
    expect(leftMs("2026-11-03T10:01:43Z", now)).toBe(103_000);
    expect(leftMs("2026-11-03T09:59:00Z", now)).toBe(0);
    expect(leftMs(null, now)).toBeNull();
    expect(leftMs("not a time", now)).toBeNull();
  });
});

describe("inspectStore", () => {
  beforeEach(() => useInspectStore.getState().reset("M1"));

  it("keeps one open look and tells the phase it had before", () => {
    const s = useInspectStore.getState();
    expect(s.upsert(look())).toBeNull();
    expect(s.upsert(look({ phase: "lower_offer" }))?.phase).toBe("flying");
    expect(openLook(useInspectStore.getState().looks)?.phase).toBe("lower_offer");
  });

  it("never goes back from the wait or from the end", () => {
    const s = useInspectStore.getState();
    s.upsert(look({ phase: "await" }));
    s.upsert(look({ phase: "flying" }));
    expect(useInspectStore.getState().looks.cmd1?.phase).toBe("await");
    s.upsert(look({ phase: "ended", ended_at: "2026-11-03T10:03:00Z", outcome: "timeout" }));
    s.setLooks([look({ phase: "await" })]);
    expect(useInspectStore.getState().looks.cmd1?.phase).toBe("ended");
    expect(openLook(useInspectStore.getState().looks)).toBeNull();
  });

  it("lets Gå lägre come and go while the drone flies", () => {
    const s = useInspectStore.getState();
    s.upsert(look({ phase: "lower_offer" }));
    s.upsert(look({ phase: "flying" }));
    expect(useInspectStore.getState().looks.cmd1?.phase).toBe("flying");
  });
});
