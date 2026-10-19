import i18n from "i18next";
import { describe, expect, it } from "vitest";
import { sentFeedback } from "./SafetyButton";

const at = new Date(2026, 0, 1, 14, 5).getTime();

describe("sentFeedback", () => {
  it("prefixes a status that follows a sent request with the time it was sent", () => {
    expect(sentFeedback(i18n.t, "sent", "väntar på drönaren", at)).toBe(
      "Skickat 14:05 · väntar på drönaren",
    );
    expect(sentFeedback(i18n.t, "unconfirmed", "kvittens saknas", at)).toBe(
      "Skickat 14:05 · kvittens saknas",
    );
  });

  it("leaves other states and unpressed buttons as they are", () => {
    expect(sentFeedback(i18n.t, "failed", "nekad", at)).toBe("nekad");
    expect(sentFeedback(i18n.t, "sent", "väntar på drönaren", null)).toBe("väntar på drönaren");
    expect(sentFeedback(i18n.t, "sent", null, at)).toBeNull();
  });
});
