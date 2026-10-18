import { describe, expect, it } from "vitest";
import type { Charter, Telemetry, VehicleLive } from "../api/types";
import { clearanceAvailable } from "./clearance";

const charter = { status: "approved", install_status: "installed" } as Charter;
const vehicle = (executor_state: string | null, airborne: boolean): VehicleLive =>
  ({
    airborne,
    snapshot: executor_state === null ? null : { executor_state },
  }) as unknown as VehicleLive;
const tel = (armed: boolean) => ({ armed }) as Telemetry;

describe("Fritt runt drönaren is offered only on the ground", () => {
  it("pilot, approved and installed charter, on the ground by every signal", () => {
    expect(clearanceAvailable(true, charter, vehicle("PREFLIGHT", false), tel(false))).toBe(true);
    expect(clearanceAvailable(true, charter, vehicle("LANDED", false), tel(false))).toBe(true);
  });

  it("never for an observer or without an installed, approved charter", () => {
    const v = vehicle("PREFLIGHT", false);
    expect(clearanceAvailable(false, charter, v, tel(false))).toBe(false);
    expect(clearanceAvailable(true, null, v, tel(false))).toBe(false);
    for (const c of [
      { status: "proposed", install_status: "installed" },
      { status: "approved", install_status: "pending" },
      { status: "approved", install_status: "failed" },
    ])
      expect(clearanceAvailable(true, c as Charter, v, tel(false))).toBe(false);
  });

  it("never when any signal says the vehicle is not on the ground", () => {
    expect(clearanceAvailable(true, charter, vehicle("HOLD", false), tel(false))).toBe(false);
    expect(clearanceAvailable(true, charter, vehicle(null, false), tel(false))).toBe(false);
    expect(clearanceAvailable(true, charter, vehicle("PREFLIGHT", true), tel(false))).toBe(false);
    expect(clearanceAvailable(true, charter, null, tel(false))).toBe(false);
    expect(clearanceAvailable(true, charter, vehicle("PREFLIGHT", false), tel(true))).toBe(false);
    expect(clearanceAvailable(true, charter, vehicle("PREFLIGHT", false), null)).toBe(false);
  });
});
