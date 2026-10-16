import type { StateSnapshot, VehicleLive } from "../api/types";

export type Tone = "grey" | "blue" | "green" | "yellow" | "orange" | "red";
export type LinkQuality = "good" | "weak" | "lost";

/** i18n key and colour for the vehicle pill (executor state + who holds it). */
export function vehiclePill(snap: StateSnapshot | null | undefined): {
  key: string;
  tone: Tone;
  raw?: string;
} {
  if (!snap) return { key: "pills.vehicle.unknown", tone: "grey" };
  switch (snap.executor_state) {
    case "IDLE":
    case "PREFLIGHT":
    case "LANDED":
      return { key: "pills.vehicle.ground", tone: "grey" };
    case "TAKEOFF":
      return { key: "pills.vehicle.takeoff", tone: "blue" };
    case "TRANSIT":
      return { key: "pills.vehicle.transit", tone: "green" };
    case "SURVEY":
      return { key: "pills.vehicle.survey", tone: "green" };
    case "HOLD":
      return { key: "pills.vehicle.hold", tone: "yellow" };
    case "HELD":
      if (snap.held_by === "operator_stop") return { key: "pills.vehicle.stopped", tone: "yellow" };
      if (snap.held_by === "operator_home")
        return { key: "pills.vehicle.homeByYou", tone: "orange" };
      if (snap.held_by === "operator_manual")
        return { key: "pills.vehicle.manual", tone: "yellow" };
      return { key: "pills.vehicle.humanTookOver", tone: "yellow" };
    case "RTL":
      return { key: "pills.vehicle.rtl", tone: "orange" };
    case "LAND":
      return { key: "pills.vehicle.land", tone: "orange" };
    case "FAILSAFE":
      return { key: "pills.vehicle.failsafe", tone: "red" };
    default:
      return { key: "pills.vehicle.raw", tone: "grey", raw: String(snap.executor_state) };
  }
}

const RANK: Record<LinkQuality, number> = { good: 0, weak: 1, lost: 2 };

/** Worst of the server's link report and the age of the last telemetry. */
export function linkQuality(
  link: VehicleLive["link"] | null | undefined,
  telemetryAgeMs: number | null,
): LinkQuality {
  const fromAge: LinkQuality =
    telemetryAgeMs === null
      ? "lost"
      : telemetryAgeMs < 1_500
        ? "good"
        : telemetryAgeMs <= 3_000
          ? "weak"
          : "lost";
  let fromServer: LinkQuality = "lost";
  if (link) {
    const parts = [link.mavlink, link.sdlink];
    fromServer = parts.every((p) => p === "up")
      ? "good"
      : parts.some((p) => p === "lost")
        ? "lost"
        : "weak";
  }
  return RANK[fromAge] >= RANK[fromServer] ? fromAge : fromServer;
}

export const STALE_MS = 3_000;
