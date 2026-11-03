import { describe, expect, it } from "vitest";
import type { FloorPackInfo } from "../../../api/types";
import { defaultFloorPack, floorPackChoice, inForceGone, NO_PACK } from "./floorPack";

const SHA = "ab".repeat(32);
const pack = (id: string, sha = SHA): FloorPackInfo => ({
  floor_pack_id: id,
  floor_pack_sha256: sha,
  attribution: "Testdata (CC0)",
  origin: { lat: -35.36, lon: 149.17 },
  size_east_m: 2000,
  size_north_m: 1500,
});

describe("the floor pack on the charter card", () => {
  it("picks nothing for the pilot on a first charter", () => {
    expect(defaultFloorPack([pack("a")], null)).toBe(NO_PACK);
    expect(defaultFloorPack([pack("a")], { floor_pack_id: null, floor_pack_sha256: null })).toBe(
      NO_PACK,
    );
  });

  it("keeps the pack in force while the ground has it unchanged", () => {
    const inForce = { floor_pack_id: "a", floor_pack_sha256: SHA };
    expect(defaultFloorPack([pack("b"), pack("a")], inForce)).toBe("a");
    expect(inForceGone([pack("a")], inForce)).toBeNull();
  });

  it("falls back to none when the pack in force is gone or written anew", () => {
    const inForce = { floor_pack_id: "a", floor_pack_sha256: SHA };
    expect(defaultFloorPack([pack("b")], inForce)).toBe(NO_PACK);
    expect(inForceGone([pack("b")], inForce)).toBe("a");
    expect(defaultFloorPack([pack("a", "cd".repeat(32))], inForce)).toBe(NO_PACK);
    expect(inForceGone([pack("a", "cd".repeat(32))], inForce)).toBe("a");
    expect(inForceGone([], null)).toBeNull();
  });

  it("sends the pack with the hash the list showed, or neither", () => {
    expect(floorPackChoice("a", [pack("a")])).toEqual({
      floor_pack_id: "a",
      floor_pack_sha256: SHA,
    });
    expect(floorPackChoice(NO_PACK, [pack("a")])).toEqual({
      floor_pack_id: null,
      floor_pack_sha256: null,
    });
    expect(floorPackChoice("gone", [pack("a")])).toBeNull();
  });
});
