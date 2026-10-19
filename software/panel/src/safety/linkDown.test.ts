import i18next from "i18next";
import { describe, expect, it } from "vitest";
import type { LinkLossPlan, VehicleLive } from "../api/types";
import { findLabel, linkDownBanner, wallClock } from "./linkDown";

const T0 = Date.parse("2026-10-19T10:00:00Z");
const at = (s: number) => new Date(T0 + s * 1000).toISOString();

const normal: LinkLossPlan = {
  mode: "normal",
  lost_at_utc: at(0),
  hold_at_utc: at(20),
  home_at_utc: at(60),
};
const find: LinkLossPlan = {
  mode: "find",
  keepout_id: "ko-0190-abcd",
  candidate_id: "c1",
  candidate_seq: 3,
  hover_lat: 59.288,
  hover_lon: 18.16,
  lost_at_utc: at(0),
  home_at_utc: at(120),
  hover_until_utc: at(120),
};

const live = (plan: LinkLossPlan | null, mavlink: "up" | "lost" = "up"): VehicleLive =>
  ({
    vehicle_id: "sim-01",
    airborne: true,
    in_flight: true,
    latch: null,
    link: { mavlink, sdlink: "lost", onboard_heartbeat_age_ms: null, rtt_ms: null },
    telemetry: null,
    snapshot: null,
    link_loss_plan: plan,
  }) as VehicleLive;

const text = (v: ReturnType<typeof linkDownBanner>) => (v ? i18next.t(v.key, v.vars) : "");

describe("linkDownBanner", () => {
  it("is null without a plan", () => {
    expect(linkDownBanner(null, T0)).toBeNull();
    expect(linkDownBanner(live(null), T0)).toBeNull();
  });

  it("SD-link down, MAVLink up: SDLINK_LOST, yellow, hover then home", () => {
    const v = linkDownBanner(live(normal), T0 + 12_400);
    expect(v).toMatchObject({ code: "SDLINK_LOST", tone: "yellow" });
    expect(v?.vars.s).toBe(12);
    expect(v?.next).toEqual({ key: "linkDown.next.hold", leftS: 8 });
    expect(text(v)).toBe(
      `Datalänken till drönaren är borta sedan 12 s. Drönaren gör klart steget, svävar från ${wallClock(T0 + 20_000)} och flyger hem ${wallClock(T0 + 60_000)}. STOPP och FLYG HEM fungerar via flygkontrollen.`,
    );
    expect(linkDownBanner(live(normal), T0 + 30_000)?.next).toEqual({
      key: "linkDown.next.home",
      leftS: 30,
    });
    expect(linkDownBanner(live(normal), T0 + 61_000)?.next).toBeNull();
  });

  it("both links down: LINK_LOST_ALL, orange", () => {
    const v = linkDownBanner(live(normal, "lost"), T0 + 5_000);
    expect(v).toMatchObject({ code: "LINK_LOST_ALL", tone: "orange" });
    expect(text(v)).toBe(
      `Ingen kontakt med drönaren sedan 5 s. Väntat: den flyger hem senast ${wallClock(T0 + 60_000)}. Ta över med RC om det behövs.`,
    );
  });

  it("find mode, MAVLink up: FIND_HOLD with K<n>, counting down to home", () => {
    const v = linkDownBanner(live(find), T0 + 30_000);
    expect(v).toMatchObject({ code: "FIND_HOLD", tone: "orange" });
    expect(v?.next).toEqual({ key: "linkDown.next.home", leftS: 90 });
    expect(text(v)).toBe(
      `Datalänken är borta vid fynd K3. Drönaren svävar vid fyndet till ${wallClock(T0 + 120_000)} och flyger sedan hem. Positionen är sparad. STOPP och FLYG HEM fungerar.`,
    );
  });

  it("find mode, both links down: FIND_HOLD_ALL", () => {
    const v = linkDownBanner(live(find, "lost"), T0 + 45_000);
    expect(v).toMatchObject({ code: "FIND_HOLD_ALL", tone: "orange" });
    expect(text(v)).toBe(
      `Ingen kontakt med drönaren sedan 45 s. Väntat: den svävar vid fynd K3 till ca ${wallClock(T0 + 120_000)} och flyger sedan hem. Positionen är sparad. Ta över med RC om det behövs.`,
    );
  });

  it("shows times as HH:MM:SS and the find by number, else by keep-out id", () => {
    expect(wallClock(T0)).toMatch(/^\d\d:\d\d:\d\d$/);
    expect(findLabel(7, "ko-1")).toBe("K7");
    expect(findLabel(null, "ko-0190-abcd")).toBe("abcd");
    expect(findLabel(null, null)).toBe("–");
  });
});
