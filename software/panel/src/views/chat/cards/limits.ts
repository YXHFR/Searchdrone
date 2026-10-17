// Limit checks for charter cards and the limits form, always against the server's numbers
// (PanelConfig.limits_absolute = the outer cap, limits_default = the standard values). The
// panel never hard-codes a limit. The server and the safety gate refuse anyway; this only
// keeps "Godkänn" and "Spara" off and explains why.
import type { MissionCharter, PanelConfig, SafetyLimits } from "../../../api/types";
import { areaM2, distanceM, type LatLon } from "../../../lib/geo";

export type Problem =
  | { key: "limits.aboveCap"; field: string; cap: number }
  | { key: "limits.belowFloor"; field: string; floor: number }
  | { key: "limits.minNotBelowMax"; field: string }
  | { key: "limits.areaTooBig"; field: string; cap: number }
  | { key: "limits.pilotFarFromHome"; field: string; m: number }
  | { key: "limits.homeFarFromVehicle"; field: string; m: number };

const NEAR_M = 10;

type Abs = PanelConfig["limits_absolute"];

/** Mission limits form: the four fields a pilot can change. */
export function limitProblems(l: SafetyLimits, abs: Abs): Problem[] {
  const out: Problem[] = [];
  const above = (field: keyof SafetyLimits & keyof Abs) => {
    const v = l[field];
    const cap = abs[field];
    if (v !== undefined && cap !== undefined && v > cap)
      out.push({ key: "limits.aboveCap", field, cap });
  };
  above("max_alt_agl_m");
  above("max_speed_ms");
  above("max_range_from_pilot_m");
  const floor = abs.min_battery_rtl_pct;
  if (l.battery_rtl_pct !== undefined && floor !== undefined && l.battery_rtl_pct < floor)
    out.push({ key: "limits.belowFloor", field: "battery_rtl_pct", floor });
  return out;
}

/** Charter card: everything in the limit table, plus home and pilot position. */
export function charterProblems(
  c: MissionCharter,
  abs: Abs,
  telemetryHome: LatLon | null | undefined,
): Problem[] {
  const out = limitProblems(
    {
      max_alt_agl_m: c.alt_max_agl_m,
      max_speed_ms: c.max_speed_ms,
      max_range_from_pilot_m: c.max_range_from_pilot_m,
      battery_rtl_pct: c.battery_rtl_pct,
    } as SafetyLimits,
    abs,
  );
  if (c.alt_min_agl_m !== undefined) {
    if (abs.min_alt_agl_m !== undefined && c.alt_min_agl_m < abs.min_alt_agl_m)
      out.push({ key: "limits.belowFloor", field: "alt_min_agl_m", floor: abs.min_alt_agl_m });
    if (c.alt_max_agl_m !== undefined && c.alt_min_agl_m >= c.alt_max_agl_m)
      out.push({ key: "limits.minNotBelowMax", field: "alt_min_agl_m" });
  }
  if (abs.max_area_km2 !== undefined && areaM2(c.area) / 1e6 > abs.max_area_km2)
    out.push({ key: "limits.areaTooBig", field: "area", cap: abs.max_area_km2 });
  const pilot = distanceM(c.pilot_position, c.home);
  if (pilot > NEAR_M)
    out.push({ key: "limits.pilotFarFromHome", field: "pilot_position", m: Math.round(pilot) });
  if (telemetryHome) {
    const d = distanceM(c.home, telemetryHome);
    if (d > NEAR_M) out.push({ key: "limits.homeFarFromVehicle", field: "home", m: Math.round(d) });
  }
  return out;
}

const SILENT_ORDER = ["hold", "continue_plan", "return_home"];

/** Is the change from `before` to `after` looser for this field? (null = cannot tell.) */
export function looser(field: string, before: unknown, after: unknown): boolean | null {
  const n = (x: unknown) => (typeof x === "number" ? x : null);
  switch (field) {
    case "alt_max_agl_m":
    case "max_speed_ms":
    case "max_range_from_pilot_m":
    case "geofence_buffer_m":
    case "ai_silent_timeout_s": {
      const a = n(before);
      const b = n(after);
      return a === null || b === null ? null : b > a;
    }
    case "alt_min_agl_m":
    case "battery_rtl_pct": {
      const a = n(before);
      const b = n(after);
      return a === null || b === null ? null : b < a;
    }
    case "on_ai_silent":
      return SILENT_ORDER.indexOf(String(after)) > SILENT_ORDER.indexOf(String(before));
    case "allowed_skills": {
      const a = new Set(Array.isArray(before) ? before.map(String) : []);
      return Array.isArray(after) && after.some((s) => !a.has(String(s)));
    }
    case "home":
    case "pilot_position": {
      const a = before as LatLon | null;
      const b = after as LatLon | null;
      return a && b ? distanceM(a, b) > 5 : null;
    }
    case "area":
      return areaM2(after) > areaM2(before) + 1;
    default:
      return null;
  }
}

/** Fields the server sets itself; they always differ between versions and say nothing. */
const SERVER_FIELDS = new Set([
  "charter_id",
  "version",
  "approved_at",
  "approved_by",
  "fence_hash",
  "mission_id",
  "vehicle_id",
  "autonomy_mode",
]);

export interface DiffRow {
  field: string;
  before: unknown;
  after: unknown;
  looser: boolean | null;
  known: boolean;
}

const KNOWN = new Set([
  "ai_silent_timeout_s",
  "allowed_skills",
  "alt_max_agl_m",
  "alt_min_agl_m",
  "area",
  "battery_rtl_pct",
  "default_alt_agl_m",
  "geofence_buffer_m",
  "home",
  "max_range_from_pilot_m",
  "max_speed_ms",
  "notes",
  "on_ai_silent",
  "pilot_position",
  "sectors",
  "no_personal_data",
]);

/** Every field that differs, known or not (a generic diff). */
export function charterDiff(before: MissionCharter, after: MissionCharter): DiffRow[] {
  const a = before as unknown as Record<string, unknown>;
  const b = after as unknown as Record<string, unknown>;
  const fields = [...new Set([...Object.keys(a), ...Object.keys(b)])].sort();
  return fields
    .filter((f) => !SERVER_FIELDS.has(f) && JSON.stringify(a[f]) !== JSON.stringify(b[f]))
    .map((f) => ({
      field: f,
      before: a[f],
      after: b[f],
      looser: KNOWN.has(f) ? looser(f, a[f], b[f]) : null,
      known: KNOWN.has(f),
    }));
}

/** "ändrat från standard" for a charter value against limits_default. */
export function changedFromDefault(
  field: keyof SafetyLimits,
  value: number | undefined,
  defaults: SafetyLimits,
): "tighter" | "looser" | null {
  const d = defaults[field];
  if (value === undefined || d === undefined || value === d) return null;
  const isLooser = looser(field, d, value);
  return isLooser ? "looser" : "tighter";
}
