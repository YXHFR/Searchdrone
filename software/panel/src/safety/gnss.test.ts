import i18next from "i18next";
import { describe, expect, it } from "vitest";
import { type AlertView, alertView, findHoldEndedView, isSticky } from "./gnss";
import { wallClock } from "./linkDown";

const T0 = Date.parse("2026-10-19T10:00:00Z");
const say = (v: AlertView | null) => (v ? i18next.t(v.key, v.vars) : null);

describe("alertView", () => {
  it("GPS_WEAK: yellow, with the satellites", () => {
    const v = alertView("GPS_WEAK", { satellites: 5, hdop: 2.4 }, T0);
    expect(v?.tone).toBe("yellow");
    expect(say(v)).toBe(
      "GPS är svag (5 satelliter). Drönaren flyger vidare men kan bli osäker på var den är.",
    );
  });

  it("GPS_LOST: red and sticky", () => {
    const v = alertView("GPS_LOST", { dr_timeout_s: 30 }, T0);
    expect(v && isSticky(v)).toBe(true);
    expect(say(v)).toBe(
      "GPS är borta. Drönaren gissar sin position och flyger mot hem i högst 30 s, sedan landar den. Ta över med RC om du kan.",
    );
  });

  it("DEAD_RECKONING: red, seconds without GPS and left", () => {
    const v = alertView(
      "DEAD_RECKONING",
      { dr_timeout_s: 30, started_utc: new Date(T0 - 12_000).toISOString() },
      T0,
    );
    expect(v?.tone).toBe("red");
    expect(say(v)).toBe(
      "Drönaren flyger utan GPS sedan 12 s (dödräkning: den gissar sin position utifrån fart och vind). Den landar senast om 18 s.",
    );
  });

  it("GPS_GLITCH: orange", () => {
    const v = alertView("GPS_GLITCH", {}, T0);
    expect(v?.tone).toBe("orange");
    expect(say(v)).toBe("GPS hoppade. Flygkontrollen litar inte på GPS just nu.");
  });

  it("GPS_SPOOF: only the half that fits the kind", () => {
    const jump = say(alertView("GPS_SPOOF", { kind: "jump", value_m: 412.6 }, T0));
    expect(jump).toContain("(positionen hoppade 413 m)");
    expect(jump).not.toContain("höjden");
    const alt = alertView("GPS_SPOOF", { kind: "altitude", value_m: 80 }, T0);
    expect(alt?.tone).toBe("red");
    expect(say(alt)).toBe(
      "GPS verkar förfalskad (höjden skiljer 80 m). Drönaren slutar lita på GPS, flyger mot hem en kort stund och landar sedan. Ta över med RC.",
    );
  });

  it("GPS_OK: info, with the RC switch after a spoof", () => {
    expect(alertView("GPS_OK", { after_spoof: false }, T0)?.tone).toBe("info");
    expect(say(alertView("GPS_OK", { after_spoof: false }, T0))).toBe("GPS fungerar igen.");
    expect(say(alertView("GPS_OK", { after_spoof: true }, T0))).toBe(
      "GPS fungerar igen, men drönaren använder den inte förrän du byter tillbaka med RC-brytaren.",
    );
  });

  it("the link back and the link-down codes from the alert's details", () => {
    expect(say(alertView("LINK_BACK_HOLD", {}, T0))).toBe(
      "Länken är tillbaka. Drönaren svävar och väntar på order.",
    );
    expect(alertView("LINK_BACK_HOME", {}, T0)?.tone).toBe("yellow");
    const fh = alertView(
      "FIND_HOLD",
      {
        lost_at_utc: new Date(T0).toISOString(),
        home_at_utc: new Date(T0 + 60_000).toISOString(),
        hover_until_utc: new Date(T0 + 60_000).toISOString(),
        candidate_seq: 2,
      },
      T0,
    );
    expect(fh?.tone).toBe("orange");
    expect(say(fh)).toContain("vid fynd K2");
    expect(alertView("FIND_HOLD", {}, T0)).toBeNull(); // no times: the ground's own text
  });

  it("is null for codes the panel has no text for", () => {
    expect(alertView("BATTERY_RTL", {}, T0)).toBeNull();
  });

  it("only red ones are sticky", () => {
    for (const [code, d] of [
      ["GPS_WEAK", { satellites: 4 }],
      ["GPS_GLITCH", {}],
      ["GPS_OK", {}],
      ["LINK_BACK_HOME", {}],
    ] as const) {
      const v = alertView(code, d, T0);
      expect(v && isSticky(v)).toBe(false);
    }
  });
});

describe("findHoldEndedView", () => {
  const line = (reason: string, seq: number | null = 3) => {
    const v = findHoldEndedView({ keepout_id: "ko-9f2e", reason, hovered_s: 95 }, T0, seq);
    return i18next.t(v.key, {
      ...v.vars,
      reason: v.reasonKey ? i18next.t(v.reasonKey) : reason,
    });
  };

  it("says why it flew home, for every reason", () => {
    const why: Record<string, string> = {
      timeout: "tiden var slut",
      battery: "batteriet",
      limit: "en gräns",
      gnss: "GPS",
      human: "du eller RC tog över",
      fc_failsafe: "du eller RC tog över",
    };
    for (const [reason, sv] of Object.entries(why))
      expect(line(reason)).toBe(
        `Utan länk: svävade vid K3 i 1:35 och flög hem ${wallClock(T0)} (${sv}).`,
      );
  });

  it("with the link back it did not fly home", () => {
    expect(line("link_restored")).toBe(
      `Utan länk: svävade vid K3 i 1:35, sedan kom länken tillbaka ${wallClock(T0)}.`,
    );
  });

  it("names the find by its keep-out id when the number is not known", () => {
    expect(line("timeout", null)).toContain("svävade vid 9f2e i");
  });
});
