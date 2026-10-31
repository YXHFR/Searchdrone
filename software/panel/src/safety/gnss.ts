// Alerts the panel words itself (GPS, the link coming back, the link-down codes, the keep-out
// list on board or not, the keep-out hover) and the log
// line for a find-hold that ended without the link. Pure: keys and values, no i18next.
import type { KeepoutHoldField } from "@searchdrone/schemas";
import { mmss } from "../lib/useNow";
import {
  findLabel,
  type LinkDownCode,
  linkDownView,
  type PlanTimes,
  parseUtc,
  wallClock,
} from "./linkDown";

export type AlertTone = "yellow" | "orange" | "red" | "info";

export interface AlertView {
  key: string;
  vars: Record<string, string | number>;
  tone: AlertTone;
}

/** Red ones stay on screen until the person has acknowledged them. */
export const isSticky = (v: AlertView): boolean => v.tone === "red";

const num = (x: unknown): number | null => (typeof x === "number" && Number.isFinite(x) ? x : null);
const round = (x: unknown): number => Math.round(num(x) ?? 0);
// The KEEPOUT_HOLD details the ground passes on (sd_schemas.api.KeepoutHoldField).
const INSIDE: KeepoutHoldField = "inside";
const CLIMB_TO_M: KeepoutHoldField = "climb_to_m";

/** The link-down codes: the stage banner shows them for as long as they last. */
export const LINK_DOWN_CODES: ReadonlySet<string> = new Set<LinkDownCode>([
  "SDLINK_LOST",
  "LINK_LOST_ALL",
  "FIND_HOLD",
  "FIND_HOLD_ALL",
]);

/** sv.json key, values and tone for an alert, or null when the panel has no text of its own. */
export function alertView(
  code: string,
  details: Record<string, unknown>,
  nowMs: number,
): AlertView | null {
  if (LINK_DOWN_CODES.has(code)) {
    if (typeof details.lost_at_utc !== "string" || typeof details.home_at_utc !== "string")
      return null;
    const v = linkDownView(code as LinkDownCode, details as unknown as PlanTimes, nowMs);
    return { key: v.key, vars: v.vars, tone: v.tone };
  }
  switch (code) {
    case "LINK_BACK_HOLD":
    case "LINK_BACK_HOME":
      return { key: `linkDown.${code}`, vars: {}, tone: "yellow" };
    case "KEEPOUTS_NOT_INSTALLED":
      return { key: "keepouts.NOT_INSTALLED", vars: {}, tone: "red" };
    case "KEEPOUTS_INSTALLED":
      return { key: "keepouts.INSTALLED", vars: {}, tone: "info" };
    case "KEEPOUT_HOLD":
      if (details[INSIDE] !== true) return { key: "keepouts.HOLD", vars: {}, tone: "orange" };
      if (num(details[CLIMB_TO_M]) !== null)
        return {
          key: "keepouts.HOLD_CLIMB",
          vars: { alt: round(details[CLIMB_TO_M]) },
          tone: "red",
        };
      return { key: "keepouts.HOLD_INSIDE", vars: {}, tone: "red" };
    case "GPS_WEAK":
      return { key: "gnss.GPS_WEAK", vars: { sats: round(details.satellites) }, tone: "yellow" };
    case "GPS_LOST":
      return { key: "gnss.GPS_LOST", vars: { dr: round(details.dr_timeout_s) }, tone: "red" };
    case "DEAD_RECKONING": {
      const started = parseUtc(details.started_utc as string | undefined) ?? nowMs;
      const s = Math.max(0, Math.floor((nowMs - started) / 1000));
      const left = Math.max(0, round(details.dr_timeout_s) - s);
      return { key: "gnss.DEAD_RECKONING", vars: { s, left }, tone: "red" };
    }
    case "GPS_GLITCH":
      return { key: "gnss.GPS_GLITCH", vars: {}, tone: "orange" };
    case "GPS_SPOOF":
      return {
        key: details.kind === "altitude" ? "gnss.GPS_SPOOF_ALTITUDE" : "gnss.GPS_SPOOF_JUMP",
        vars: { m: round(details.value_m) },
        tone: "red",
      };
    case "GPS_OK":
      return {
        key: details.after_spoof === true ? "gnss.GPS_OK_AFTER_SPOOF" : "gnss.GPS_OK",
        vars: {},
        tone: "info",
      };
    default:
      return null;
  }
}

export type FindHoldReason =
  | "timeout"
  | "battery"
  | "limit"
  | "gnss"
  | "human"
  | "fc_failsafe"
  | "link_restored";

const REASONS = new Set<string>([
  "timeout",
  "battery",
  "limit",
  "gnss",
  "human",
  "fc_failsafe",
  "link_restored",
]);

/**
 * The event log line for executor.find_hold_ended. `atMs` is when it ended; `seq` the
 * candidate's number when the panel still knows it from the plan. With the link back the
 * drone did not fly home, so that reason gets its own line.
 */
export function findHoldEndedView(
  payload: { keepout_id?: unknown; reason?: unknown; hovered_s?: unknown },
  atMs: number,
  seq: number | null = null,
): AlertView & { reasonKey: string | null } {
  const reason = typeof payload.reason === "string" ? payload.reason : "";
  const known = REASONS.has(reason);
  const vars = {
    k: findLabel(seq, typeof payload.keepout_id === "string" ? payload.keepout_id : null),
    dur: mmss((num(payload.hovered_s) ?? 0) * 1000),
    time: wallClock(atMs),
  };
  if (reason === "link_restored")
    return { key: "linkDown.findHoldEndedBack", vars, tone: "info", reasonKey: null };
  return {
    key: "linkDown.findHoldEnded",
    vars,
    tone: "info",
    reasonKey: known ? `linkDown.reason.${reason}` : null,
  };
}
