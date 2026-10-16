import { describe, expect, it } from "vitest";
import { UUIDV7_RE, uuidv7 } from "./uuidv7";

describe("uuidv7", () => {
  it("has version 7 and the RFC variant", () => {
    expect(uuidv7()).toMatch(UUIDV7_RE);
  });

  it("encodes the timestamp in the first 48 bits", () => {
    const id = uuidv7(1_760_000_000_000);
    expect(Number.parseInt(id.replace(/-/g, "").slice(0, 12), 16)).toBeGreaterThanOrEqual(
      1_760_000_000_000,
    );
  });

  it("is strictly increasing for 1000 ids", () => {
    const ids = Array.from({ length: 1000 }, () => uuidv7());
    for (let i = 1; i < ids.length; i++) expect((ids[i] ?? "") > (ids[i - 1] ?? "")).toBe(true);
    expect(new Set(ids).size).toBe(1000);
  });
});
