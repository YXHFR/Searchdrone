// The top strip's instrument cells as plain data, so every rule (units, greying, the link
// bars) is tested without React. Values are already formatted; labels are i18n keys.
import type { Telemetry, VehicleLive } from "../api/types";
import { type LinkQuality, linkQuality, STALE_MS } from "./pills";

export type LinkStatus = NonNullable<VehicleLive["link"]>;
export type CellTone = "normal" | "warn" | "bad" | "stale";
export type CellId =
  | "age"
  | "battery"
  | "timeLeft"
  | "alt"
  | "speed"
  | "heading"
  | "fromYou"
  | "link"
  | "gps";

export interface Cell {
  id: CellId;
  labelKey: string;
  value: string;
  tone: CellTone;
  /** Battery level and the return-home mark, both 0..100. */
  bar?: { pct: number; rtlPct: number };
  /** Link strength, 0 (lost) to 4. */
  bars?: number;
  /** Extra words for screen readers and the tooltip, as an i18n key. */
  hintKey?: string;
  /** Short words shown when there is no number, as an i18n key. */
  valueKey?: string;
}

export interface InstrumentInput {
  telemetry: Telemetry | null;
  /** Milliseconds since the last telemetry; null when none has arrived. */
  ageMs: number | null;
  link: LinkStatus | null;
  rtlPct: number;
}

const DASH = "–";
const one = new Intl.NumberFormat("sv-SE", { maximumFractionDigits: 1, minimumFractionDigits: 1 });
const whole = new Intl.NumberFormat("sv-SE", { maximumFractionDigits: 0 });

/** 8-point compass in Swedish: N, NO, O, SO, S, SV, V, NV. */
export function compass(deg: number): string {
  const names = ["N", "NO", "O", "SO", "S", "SV", "V", "NV"];
  return names[Math.round((((deg % 360) + 360) % 360) / 45) % 8] ?? "N";
}

export function heading(deg: number): string {
  const d = Math.round(((deg % 360) + 360) % 360) % 360;
  return `${String(d).padStart(3, "0")}° ${compass(d)}`;
}

export function distance(m: number): string {
  if (m < 1000) return `${whole.format(Math.round(m))} m`;
  return `${one.format(m / 1000)} km`;
}

/** Fix type as MAVLink GPS_FIX_TYPE: below 2 there is no position. */
export function gpsText(fix: number, sats: number): string {
  const kind =
    fix >= 6
      ? "RTK"
      : fix === 5
        ? "RTK~"
        : fix === 4
          ? "DGPS"
          : fix === 3
            ? "3D"
            : fix === 2
              ? "2D"
              : null;
  return kind ? `${kind} · ${sats}` : `${DASH} · ${sats}`;
}

const BARS: Record<LinkQuality, number> = { good: 4, weak: 2, lost: 0 };

function linkBars(q: LinkQuality, rttMs: number | null): number {
  if (q !== "good" || rttMs === null) return BARS[q];
  return rttMs <= 150 ? 4 : 3;
}

export function isStale(ageMs: number | null): boolean {
  return ageMs === null || ageMs > STALE_MS;
}

/** Every cell, in strip order. While stale, an age cell comes first and values grey out. */
export function instrumentCells(i: InstrumentInput): Cell[] {
  const t = i.telemetry;
  const stale = isStale(i.ageMs);
  const tone = (live: CellTone): CellTone => (stale ? "stale" : live);
  const cells: Cell[] = [];
  if (stale && t && i.ageMs !== null) {
    cells.push({
      id: "age",
      labelKey: "strip.age",
      value: `${Math.floor(i.ageMs / 1000)} s`,
      tone: "warn",
    });
  }
  const pct = t?.battery_pct ?? null;
  cells.push({
    id: "battery",
    labelKey: "strip.battery",
    value: pct === null ? DASH : `${Math.round(pct)} %`,
    tone: tone(pct === null ? "normal" : pct <= i.rtlPct ? "bad" : pct < 30 ? "warn" : "normal"),
    ...(pct === null ? {} : { bar: { pct: Math.max(0, Math.min(100, pct)), rtlPct: i.rtlPct } }),
  });
  // No time-left estimate in the panel: only the drone may say this, and it does not yet.
  cells.push({
    id: "timeLeft",
    labelKey: "strip.timeLeft",
    value: DASH,
    tone: tone("normal"),
    hintKey: "strip.timeLeftNone",
  });
  cells.push({
    id: "alt",
    labelKey: "strip.alt",
    value: t ? `${whole.format(Math.round(t.alt_rel_home_m))} m` : DASH,
    tone: tone("normal"),
  });
  cells.push({
    id: "speed",
    labelKey: "strip.speed",
    value: t ? `${one.format(t.groundspeed_ms)} m/s` : DASH,
    tone: tone("normal"),
  });
  cells.push({
    id: "heading",
    labelKey: "strip.heading",
    value: t ? heading(t.heading_deg) : DASH,
    tone: tone("normal"),
  });
  const away = t?.dist_to_pilot_m ?? null;
  cells.push({
    id: "fromYou",
    labelKey: "strip.fromYou",
    value: away === null ? DASH : distance(away),
    tone: tone("normal"),
  });
  const q = linkQuality(i.link, i.ageMs);
  const rtt = i.link?.rtt_ms ?? null;
  cells.push({
    id: "link",
    labelKey: "strip.link",
    value: q === "lost" ? "" : rtt === null ? "" : `${Math.round(rtt)} ms`,
    tone: q === "lost" ? "bad" : q === "weak" ? "warn" : "normal",
    bars: linkBars(q, rtt),
    hintKey: `strip.linkQ.${q}`,
    valueKey: `strip.linkShort.${q}`,
  });
  cells.push({
    id: "gps",
    labelKey: "strip.gps",
    value: t ? gpsText(t.gps_fix, t.gps_sats) : DASH,
    tone: tone(t && t.gps_fix < 3 ? "warn" : "normal"),
  });
  return cells;
}
