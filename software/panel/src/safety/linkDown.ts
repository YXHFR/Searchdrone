import type { LinkLossPlan, VehicleLive } from "../api/types";

export type LinkDownCode = "SDLINK_LOST" | "LINK_LOST_ALL" | "FIND_HOLD" | "FIND_HOLD_ALL";

/** The plan fields the banner and the link alerts read (the alerts carry the same ones). */
export type PlanTimes = Pick<LinkLossPlan, "lost_at_utc" | "home_at_utc"> &
  Partial<Pick<LinkLossPlan, "hold_at_utc" | "hover_until_utc" | "candidate_seq" | "keepout_id">>;

export interface LinkDownView {
  code: LinkDownCode;
  key: `linkDown.${LinkDownCode}`;
  tone: "yellow" | "orange";
  /** s: whole seconds since the link went; hold, home, until: wall clock; k: the find. */
  vars: { s: number; hold: string; home: string; until: string; k: string };
  /** What comes next and how many seconds are left, for the countdown. */
  next: { key: "linkDown.next.hold" | "linkDown.next.home"; leftS: number } | null;
}

const clock = new Intl.DateTimeFormat("sv-SE", {
  hour: "2-digit",
  minute: "2-digit",
  second: "2-digit",
});

/** HH:MM:SS in local time, or "–" for a missing or unreadable time. */
export function wallClock(ms: number | null): string {
  return ms === null ? "–" : clock.format(new Date(ms));
}

/** Epoch ms of an ISO time, or null. */
export function parseUtc(iso: string | null | undefined): number | null {
  if (!iso) return null;
  const ms = Date.parse(iso);
  return Number.isFinite(ms) ? ms : null;
}

/** "K<seq>" for the candidate, else the end of the keep-out id (ids share their start). */
export function findLabel(seq: number | null | undefined, keepoutId: string | null | undefined) {
  if (typeof seq === "number") return `K${seq}`;
  return keepoutId ? keepoutId.slice(-4) : "–";
}

const TONE: Record<LinkDownCode, LinkDownView["tone"]> = {
  SDLINK_LOST: "yellow",
  LINK_LOST_ALL: "orange",
  FIND_HOLD: "orange",
  FIND_HOLD_ALL: "orange",
};

/** The text and countdown for one link code and plan at `nowMs`. */
export function linkDownView(code: LinkDownCode, plan: PlanTimes, nowMs: number): LinkDownView {
  const lost = parseUtc(plan.lost_at_utc);
  const home = parseUtc(plan.home_at_utc);
  const hold = parseUtc(plan.hold_at_utc) ?? home;
  const until = parseUtc(plan.hover_until_utc) ?? home;
  const find = code === "FIND_HOLD" || code === "FIND_HOLD_ALL";
  const left = (at: number | null): number | null =>
    at !== null && at > nowMs ? Math.ceil((at - nowMs) / 1000) : null;
  let next: LinkDownView["next"] = null;
  // Normal: it hovers first, then flies home. Find: it hovers by the find until it flies home.
  const toHold = find ? null : left(parseUtc(plan.hold_at_utc));
  const toHome = left(find ? until : home);
  if (toHold !== null) next = { key: "linkDown.next.hold", leftS: toHold };
  else if (toHome !== null) next = { key: "linkDown.next.home", leftS: toHome };
  return {
    code,
    key: `linkDown.${code}`,
    tone: TONE[code],
    vars: {
      s: lost === null ? 0 : Math.max(0, Math.floor((nowMs - lost) / 1000)),
      hold: wallClock(hold),
      home: wallClock(home),
      until: wallClock(until),
      k: findLabel(plan.candidate_seq, plan.keepout_id),
    },
    next,
  };
}

/**
 * The banner while SD-link is down in the air (the ground sets link_loss_plan only then).
 * With MAVLink up too the ground still reaches the flight controller; with both down it
 * only knows what the drone is expected to do.
 */
export function linkDownBanner(v: VehicleLive | null, nowMs: number): LinkDownView | null {
  const plan = v?.link_loss_plan;
  if (!v || !plan) return null;
  const all = v.link.mavlink === "lost";
  const code: LinkDownCode =
    plan.mode === "find"
      ? all
        ? "FIND_HOLD_ALL"
        : "FIND_HOLD"
      : all
        ? "LINK_LOST_ALL"
        : "SDLINK_LOST";
  return linkDownView(code, plan, nowMs);
}
