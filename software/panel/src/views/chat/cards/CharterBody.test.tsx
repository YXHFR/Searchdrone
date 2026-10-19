import { render, screen } from "@testing-library/react";
import { describe, expect, it } from "vitest";
import { charterBody } from "../../../../e2e/fixtures";
import type { MissionCharter } from "../../../api/types";
import { CharterKeyFacts } from "./CharterBody";

const fact = (k: string): string =>
  screen.getByTestId("charter-facts").querySelector(`[data-fact="${k}"] dd`)?.textContent ?? "";

describe("CharterKeyFacts", () => {
  it("shows the area, where it lies from home, the heights, the speed and the return level", () => {
    render(<CharterKeyFacts c={charterBody()} />);
    expect(fact("area")).toMatch(/^1(,0)? ha · mitt 180 m .+ om hem$/);
    expect(fact("alt")).toBe("15–60 m");
    expect(fact("speed")).toBe("≤ 8 m/s");
    expect(fact("rtl")).toBe("15 % batteri");
  });

  it("shows a dash for a value the charter does not carry", () => {
    const c = { ...charterBody(), max_speed_ms: undefined } as unknown as MissionCharter;
    render(<CharterKeyFacts c={c} />);
    expect(fact("speed")).toBe("≤ – m/s");
  });
});
