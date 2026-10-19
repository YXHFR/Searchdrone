import { describe, expect, it } from "vitest";
import type { Telemetry } from "../api/types";
import {
  compass,
  distance,
  gpsText,
  heading,
  instrumentCells,
  type LinkStatus,
} from "./instruments";

const tel: Telemetry = {
  t: "2026-01-01T00:00:00Z",
  time_boot_ms: 1,
  lat: -35.36,
  lon: 149.16,
  alt_amsl_m: 600,
  alt_rel_home_m: 40.4,
  heading_deg: 72,
  groundspeed_ms: 6.24,
  climb_ms: 0,
  battery_pct: 58,
  battery_v: 15.2,
  flight_mode: "GUIDED",
  armed: true,
  landed_state: "in_air",
  gps_fix: 3,
  gps_sats: 14,
  home: null,
  dist_to_pilot_m: 210,
};
const up: LinkStatus = { mavlink: "up", sdlink: "up", rtt_ms: 38, onboard_heartbeat_age_ms: 100 };

const byId = (cells: ReturnType<typeof instrumentCells>) =>
  Object.fromEntries(cells.map((c) => [c.id, c]));

describe("instrument cells", () => {
  it("formats live values in Swedish units", () => {
    const c = byId(instrumentCells({ telemetry: tel, ageMs: 200, link: up, rtlPct: 15 }));
    expect(c.battery?.value).toBe("58 %");
    expect(c.battery?.bar).toEqual({ pct: 58, rtlPct: 15 });
    expect(c.alt?.value).toBe("40 m");
    expect(c.speed?.value).toBe("6,2 m/s");
    expect(c.heading?.value).toBe("072° O");
    expect(c.fromYou?.value).toBe("210 m");
    expect(c.link?.value).toBe("38 ms");
    expect(c.link?.bars).toBe(4);
    expect(c.gps?.value).toBe("3D · 14");
    expect(c.age).toBeUndefined();
  });

  it("never estimates time left", () => {
    const c = byId(instrumentCells({ telemetry: tel, ageMs: 200, link: up, rtlPct: 15 }));
    expect(c.timeLeft?.value).toBe("–");
  });

  it("greys every value and leads with the age when the data is old", () => {
    const cells = instrumentCells({ telemetry: tel, ageMs: 13_400, link: up, rtlPct: 15 });
    expect(cells[0]).toMatchObject({ id: "age", value: "13 s" });
    const c = byId(cells);
    expect(c.alt?.tone).toBe("stale");
    expect(c.alt?.value).toBe("40 m");
    expect(c.link?.tone).toBe("bad");
    expect(c.link?.bars).toBe(0);
    expect(c.link?.valueKey).toBe("strip.linkShort.lost");
  });

  it("shows dashes, not zeros, before the first telemetry", () => {
    const c = byId(instrumentCells({ telemetry: null, ageMs: null, link: null, rtlPct: 15 }));
    expect(c.alt?.value).toBe("–");
    expect(c.battery?.value).toBe("–");
    expect(c.battery?.bar).toBeUndefined();
    expect(c.age).toBeUndefined();
  });

  it("warns at low battery and alarms at the return-home level", () => {
    const low = byId(
      instrumentCells({ telemetry: { ...tel, battery_pct: 25 }, ageMs: 0, link: up, rtlPct: 15 }),
    );
    expect(low.battery?.tone).toBe("warn");
    const rtl = byId(
      instrumentCells({ telemetry: { ...tel, battery_pct: 15 }, ageMs: 0, link: up, rtlPct: 15 }),
    );
    expect(rtl.battery?.tone).toBe("bad");
  });
});

describe("formatting", () => {
  it("names the compass points", () => {
    expect([0, 44, 46, 90, 180, 270, 337, 359].map(compass)).toEqual([
      "N",
      "NO",
      "NO",
      "O",
      "S",
      "V",
      "NV",
      "N",
    ]);
    expect(heading(-5)).toBe("355° N");
  });

  it("switches to km at one kilometre", () => {
    expect(distance(999)).toBe("999 m");
    expect(distance(1250)).toBe("1,3 km");
  });

  it("has no position without a 2D fix", () => {
    expect(gpsText(1, 4)).toBe("– · 4");
    expect(gpsText(6, 20)).toBe("RTK · 20");
  });
});
