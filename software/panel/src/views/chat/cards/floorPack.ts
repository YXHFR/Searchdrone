// The height-floor map pack on the charter card: which one is picked to begin with and what
// the approval sends. Pure, so the rules are tested without React.
import type { FloorPackChoice, FloorPackInfo, MissionCharter } from "../../../api/types";

/** The radio value for "Ingen höjdkarta". */
export const NO_PACK = "";

type InForce = Pick<MissionCharter, "floor_pack_id" | "floor_pack_sha256"> | null | undefined;

/** The ground still has the pack in force, unchanged. */
function listed(packs: readonly FloorPackInfo[], c: InForce): FloorPackInfo | undefined {
  return packs.find(
    (p) => p.floor_pack_id === c?.floor_pack_id && p.floor_pack_sha256 === c?.floor_pack_sha256,
  );
}

/** A change keeps the pack in force while the ground still has it unchanged; otherwise
 * (and on a first charter) nothing is picked for the pilot: "Ingen höjdkarta", so a closer
 * look never comes down by a map the pilot did not choose. */
export function defaultFloorPack(packs: readonly FloorPackInfo[], inForce: InForce): string {
  return listed(packs, inForce)?.floor_pack_id ?? NO_PACK;
}

/** The pack in force is no longer on the ground as it was (removed or written anew). */
export function inForceGone(packs: readonly FloorPackInfo[], inForce: InForce): string | null {
  const id = inForce?.floor_pack_id;
  return id && !listed(packs, inForce) ? id : null;
}

/** What the approval sends for the radio picked: the pack with the hash the list showed,
 * or neither. Null when the pick is not in the list (nothing is sent then). */
export function floorPackChoice(
  picked: string,
  packs: readonly FloorPackInfo[],
): FloorPackChoice | null {
  if (picked === NO_PACK) return { floor_pack_id: null, floor_pack_sha256: null };
  const p = packs.find((x) => x.floor_pack_id === picked);
  return p ? { floor_pack_id: p.floor_pack_id, floor_pack_sha256: p.floor_pack_sha256 } : null;
}
