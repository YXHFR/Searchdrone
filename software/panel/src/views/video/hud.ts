// The video HUD as plain data: the heading tape's ticks and what the horizon may show. The
// horizon needs a fresh attitude from the drone; without one it is left out, never guessed.
import type { Telemetry } from "../../api/types";
import { isStale } from "../../header/instruments";

export interface Tick {
  deg: number;
  /** Degrees right of the centre (negative: left). */
  offset: number;
  label: string | null;
  major: boolean;
}

const CARDINAL: Record<number, string> = { 0: "N", 90: "O", 180: "S", 270: "V" };

/** Ticks every 5°, labelled every 30° (N, O, S, V at the cardinals), within ±span. */
export function tapeTicks(heading: number, span: number): Tick[] {
  const h = ((heading % 360) + 360) % 360;
  const out: Tick[] = [];
  for (let d = Math.ceil((h - span) / 5) * 5; d <= h + span; d += 5) {
    const deg = ((d % 360) + 360) % 360;
    const labelled = deg % 30 === 0;
    out.push({
      deg,
      offset: d - h,
      label: labelled ? (CARDINAL[deg] ?? String(deg / 10).padStart(2, "0")) : null,
      major: deg % 10 === 0,
    });
  }
  return out;
}

export interface Horizon {
  roll: number;
  pitch: number;
}

/** Roll and pitch to draw, or null when the drone has not sent a fresh attitude. */
export function horizon(tel: Telemetry | null, ageMs: number | null): Horizon | null {
  const a = tel?.attitude;
  if (!a || isStale(ageMs)) return null;
  return { roll: a.roll_deg, pitch: a.pitch_deg };
}
