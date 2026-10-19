// The left column's mission summary as plain data: sector rows and how much of the area has
// been swept. "Swept" is the camera footprint the ground station reports per sector, never a
// panel estimate from the planned lines.
import { areaM2 } from "../lib/geo";
import type { SectorFeature } from "../state/missionStore";

export type SectorStatus = "pending" | "active" | "done" | "skipped";
const STATUSES: readonly SectorStatus[] = ["pending", "active", "done", "skipped"];

export interface SectorRow {
  id: string;
  name: string;
  status: SectorStatus;
  /** 0..100, whole percent. */
  pct: number;
}

export interface Coverage {
  sweptHa: number;
  totalHa: number;
}

function status(raw: string): SectorStatus {
  return (STATUSES as readonly string[]).includes(raw) ? (raw as SectorStatus) : "pending";
}

export function sectorRows(sectors: SectorFeature[]): SectorRow[] {
  return sectors.map((s) => ({
    id: s.properties.sector_id,
    name: s.properties.name,
    status: status(s.properties.status),
    pct: Math.round(Math.max(0, Math.min(1, s.properties.progress)) * 100),
  }));
}

/** Swept and total hectares; null until there are sectors with an area. */
export function coverage(sectors: SectorFeature[]): Coverage | null {
  let total = 0;
  let swept = 0;
  for (const s of sectors) {
    const a = s.properties.area_m2;
    total += typeof a === "number" && Number.isFinite(a) ? a : areaM2(s.geometry);
    swept += areaM2(s.properties.covered);
  }
  if (total <= 0) return null;
  return { sweptHa: Math.min(swept, total) / 10_000, totalHa: total / 10_000 };
}

export function doneCount(rows: SectorRow[]): number {
  return rows.filter((r) => r.status === "done").length;
}
