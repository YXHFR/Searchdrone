import { describe, expect, it } from "vitest";
import { resolveTheme } from "./theme";

describe("resolveTheme", () => {
  it("follows the system only in auto", () => {
    expect(resolveTheme("auto", true)).toBe("light");
    expect(resolveTheme("auto", false)).toBe("dark");
    expect(resolveTheme("dark", true)).toBe("dark");
    expect(resolveTheme("light", false)).toBe("light");
  });
});
