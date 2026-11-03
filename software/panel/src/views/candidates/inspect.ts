// "Titta närmare": when the button may be pressed, and the look's progress in words. Pure,
// so the rules are tested without React.
import type { CandidateCard, Inspection, InspectProgress, MissionCharter } from "../../api/types";

export type OperationMode = "sim" | "exercise" | "live";

/** Why Titta närmare is grey, or null when the pilot may press it. In the order the
 * pilot can do something about it. */
export type InspectBlock = "notPilot" | "notOpen" | "busy" | "noLink" | "notFlying";

export function inspectBlock(o: {
  isPilot: boolean;
  card: Pick<CandidateCard, "candidate_id" | "state">;
  look: Inspection | null;
  airborne: boolean;
  sdlink: "up" | "degraded" | "lost" | null;
}): InspectBlock | null {
  if (!o.isPilot) return "notPilot";
  if (o.look && o.look.candidate_id !== o.card.candidate_id) return "busy";
  if (o.card.state !== "ready" && o.card.state !== "assessing") return "notOpen";
  if (o.sdlink !== "up" && o.sdlink !== "degraded") return "noLink";
  if (!o.airborne) return "notFlying";
  return null;
}

/** The charter's mode as the ground reads it: unset is live, the strictest. */
export function operationMode(c: Pick<MissionCharter, "operation_mode"> | null | undefined) {
  if (!c) return null;
  return (c.operation_mode ?? "live") as OperationMode;
}

/** What the look will do, for the question before it starts. */
export type LookKind = "over" | "side";

export function lookKind(
  c: Pick<MissionCharter, "operation_mode" | "overflight_allowed"> | null | undefined,
): LookKind {
  return c?.overflight_allowed ? "over" : "side";
}

/** Whole metres, or null. */
export function metres(v: number | null | undefined): number | null {
  return typeof v === "number" && Number.isFinite(v) ? Math.round(v) : null;
}

/** The floor in force on the strip: the lowest one when the drone knows it, else the
 * automatic one. */
export function stripFloor(p: InspectProgress | null | undefined): number | null {
  if (!p) return null;
  return metres(p.floor_lowest_above_ground_m) ?? metres(p.floor_auto_above_ground_m);
}

/** Milliseconds left until an ISO time (never negative), or null. */
export function leftMs(until: string | null | undefined, now: number): number | null {
  if (!until) return null;
  const at = Date.parse(until);
  return Number.isFinite(at) ? Math.max(0, at - now) : null;
}
