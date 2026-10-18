// Drawn paths (pa-1, pa-2 …) and the path the drone follows: pure functions from store state
// to GeoJSON, plus the checks the panel makes before it saves a path. No MapLibre here.
import type { Annotation, Command, StateSnapshot } from "../../api/types";
import { distanceM, type LatLon } from "../../lib/geo";
import type { FC, Feature } from "./mapData";

export const PATH_SOURCE_IDS = ["paths", "path-plan", "path-flown"] as const;
export type PathSourceId = (typeof PATH_SOURCE_IDS)[number];

/** The longest path the gate accepts (follow_path's PATH_TOO_LONG); the gate decides. */
export const PATH_MAX_LENGTH_M = 5_000;
export const PATH_MAX_POINTS = 50;

type Pos = [number, number]; // [lon, lat]

const fc = (features: Feature[]): FC => ({ type: "FeatureCollection", features });
const ll = ([lon, lat]: Pos): LatLon => ({ lat, lon });
const line = (coordinates: Pos[], properties: Record<string, unknown> = {}): Feature => ({
  type: "Feature",
  geometry: { type: "LineString", coordinates },
  properties,
});
const nfKm = new Intl.NumberFormat("sv-SE", { maximumFractionDigits: 1 });

/** Length of a line in metres. */
export function lengthM(coords: Pos[]): number {
  let sum = 0;
  for (let i = 1; i < coords.length; i += 1)
    sum += distanceM(ll(coords[i - 1] as Pos), ll(coords[i] as Pos));
  return sum;
}

/** "420 m" or "1,2 km". */
export function formatLength(m: number): string {
  return m < 1_000 ? `${Math.round(m)} m` : `${nfKm.format(m / 1_000)} km`;
}

function coordsOf(geometry: unknown): Pos[] | null {
  const g = geometry as { type?: string; coordinates?: unknown } | null;
  if (g?.type !== "LineString" || !Array.isArray(g.coordinates)) return null;
  return g.coordinates as Pos[];
}

/** The pilot's drawn paths, dashed, with "pa-1 · 420 m" at their start. */
export function pathsData(anns: Annotation[]): FC {
  const out: Feature[] = [];
  for (const a of anns) {
    const coords = a.kind === "path" ? coordsOf(a.geometry) : null;
    if (!coords || coords.length < 2) continue;
    const m = (a as { length_m?: number | null }).length_m ?? lengthM(coords);
    out.push(line(coords, { id: a.id, kind: "line", label: `${a.label} · ${formatLength(m)}` }));
    out.push({
      type: "Feature",
      geometry: { type: "Point", coordinates: coords[0] },
      properties: { id: a.id, kind: "start", label: a.label },
    });
  }
  return fc(out);
}

/** Next free path name: pa-1, pa-2 … */
export function nextPathLabel(anns: Annotation[]): string {
  let max = 0;
  for (const a of anns) {
    const m = /^pa-(\d+)$/.exec(a.label);
    if (m?.[1]) max = Math.max(max, Number(m[1]));
  }
  return `pa-${max + 1}`;
}

// --- checks before saving --------------------------------------------------------------

type Ring = Pos[];

function inside(p: Pos, ring: Ring): boolean {
  let hit = false;
  for (let i = 0, j = ring.length - 1; i < ring.length; j = i, i += 1) {
    const [xi, yi] = ring[i] as Pos;
    const [xj, yj] = ring[j] as Pos;
    if (yi > p[1] !== yj > p[1] && p[0] < ((xj - xi) * (p[1] - yi)) / (yj - yi) + xi) hit = !hit;
  }
  return hit;
}

function crosses(a: Pos, b: Pos, c: Pos, d: Pos): boolean {
  const o = (p: Pos, q: Pos, r: Pos): number =>
    (q[0] - p[0]) * (r[1] - p[1]) - (q[1] - p[1]) * (r[0] - p[0]);
  return o(c, d, a) * o(c, d, b) < 0 && o(a, b, c) * o(a, b, d) < 0;
}

/** Does any part of the line leave the polygon (a corner outside, or a leg across an edge)? */
export function leavesArea(coords: Pos[], area: unknown): boolean {
  const g = area as { type?: string; coordinates?: Ring[] } | null;
  const ring = g?.type === "Polygon" ? g.coordinates?.[0] : undefined;
  if (!ring || ring.length < 4) return false; // no charter: the gate decides
  if (coords.some((p) => !inside(p, ring))) return true;
  for (let i = 1; i < coords.length; i += 1)
    for (let j = 1; j < ring.length; j += 1)
      if (crosses(coords[i - 1] as Pos, coords[i] as Pos, ring[j - 1] as Pos, ring[j] as Pos))
        return true;
  return false;
}

export type PathProblem =
  | { key: "draw.pathTooShort" }
  | { key: "draw.pathTooManyPoints"; max: number }
  | { key: "draw.pathTooLong"; length: string; max: string }
  | { key: "draw.pathOutside" };

/** Why the panel will not save this path (the gate checks everything again), or null. */
export function pathProblem(coords: Pos[], charterArea: unknown): PathProblem | null {
  if (coords.length < 2) return { key: "draw.pathTooShort" };
  if (coords.length > PATH_MAX_POINTS)
    return { key: "draw.pathTooManyPoints", max: PATH_MAX_POINTS };
  const m = lengthM(coords);
  if (m > PATH_MAX_LENGTH_M)
    return {
      key: "draw.pathTooLong",
      length: formatLength(m),
      max: formatLength(PATH_MAX_LENGTH_M),
    };
  if (leavesArea(coords, charterArea)) return { key: "draw.pathOutside" };
  return null;
}

// --- the path being flown --------------------------------------------------------------

/** The points of a follow_path order, or null for any other order. */
export function orderPath(c: Command | null | undefined): Pos[] | null {
  if (!c || c.skill !== "follow_path") return null;
  const pts = (c.params as { points?: unknown }).points;
  if (!Array.isArray(pts)) return null;
  const out: Pos[] = [];
  for (const p of pts as { lat?: unknown; lon?: unknown }[])
    if (typeof p.lat === "number" && typeof p.lon === "number") out.push([p.lon, p.lat]);
  return out.length >= 2 ? out : null;
}

/** Where along the line the drone is: the leg it is on and the share of the length. */
function project(coords: Pos[], at: LatLon): { leg: number; along: Pos; pct: number } {
  const lat0 = (at.lat * Math.PI) / 180;
  const k = Math.cos(lat0);
  const xy = (p: Pos): [number, number] => [p[0] * k, p[1]];
  const total = lengthM(coords);
  let best = { d: Number.POSITIVE_INFINITY, leg: 0, t: 0 };
  for (let i = 1; i < coords.length; i += 1) {
    const [ax, ay] = xy(coords[i - 1] as Pos);
    const [bx, by] = xy(coords[i] as Pos);
    const [px, py] = xy([at.lon, at.lat]);
    const len2 = (bx - ax) ** 2 + (by - ay) ** 2;
    const t =
      len2 === 0
        ? 0
        : Math.max(0, Math.min(1, ((px - ax) * (bx - ax) + (py - ay) * (by - ay)) / len2));
    const d = (ax + t * (bx - ax) - px) ** 2 + (ay + t * (by - ay) - py) ** 2;
    if (d < best.d) best = { d, leg: i, t };
  }
  const a = coords[best.leg - 1] as Pos;
  const b = coords[best.leg] as Pos;
  const along: Pos = [a[0] + best.t * (b[0] - a[0]), a[1] + best.t * (b[1] - a[1])];
  const done = lengthM([...coords.slice(0, best.leg), along]);
  return { leg: best.leg, along, pct: total > 0 ? (100 * done) / total : 0 };
}

/** Orders that never reached the vehicle: they cannot be what a resume continues. */
const NEVER_SENT = new Set([
  "pending_approval",
  "rejected_by_gate",
  "rejected_by_operator",
  "approval_expired",
  "rejected_onboard",
  "expired",
]);

/** Orders that leave the vehicle's resume point as it was: a resume after them continues
 * the order before them. */
const KEEPS_RESUME_POINT = new Set(["resume", "hold"]);

/** The follow_path order a running resume continues: the last order sent before it (holds
 * aside), when that was a path. Null for any other active order. */
export function resumedPath(active: Command | null, commands: Command[]): Command | null {
  if (active?.skill !== "resume") return null;
  let last: Command | null = null;
  for (const c of commands) {
    if (KEEPS_RESUME_POINT.has(c.skill) || NEVER_SENT.has(c.status)) continue;
    if (c.created_at >= active.created_at) continue;
    if (!last || c.created_at > last.created_at) last = c;
  }
  return orderPath(last) ? last : null;
}

/** The path flown to its end by the last finished follow_path or resume order (a resume
 * that flew the rest of one is shown as done). Other orders are not considered. */
export function finishedPath(commands: Command[]): Command | null {
  let last: Command | null = null;
  for (const c of commands) {
    if (c.status !== "done" || (c.skill !== "follow_path" && c.skill !== "resume")) continue;
    if (!last || c.updated_at > last.updated_at) last = c;
  }
  if (last?.skill !== "resume") return last;
  const path = resumedPath(last, commands);
  return path ? { ...path, status: "done" } : null;
}

export interface ActivePath {
  command: Command;
  coords: Pos[];
  pct: number; // 0..100
  running: boolean;
}

/** The path to show: the running follow_path order (or the one a running resume
 * continues), else one waiting on a card, else the last order when it was a path flown to
 * its end (the drone hovers there). */
export function activePath(
  active: Command | null,
  pending: Command[],
  latest: Command | null,
  drone: LatLon | null,
  snapshot: StateSnapshot | null | undefined,
  resumed: Command | null = null,
): ActivePath | null {
  const running = orderPath(active) ? active : resumed;
  const finished = latest?.status === "done" && orderPath(latest) ? latest : null;
  const pick = running ?? pending.find((c) => orderPath(c) !== null) ?? finished;
  const coords = orderPath(pick);
  if (!pick || !coords) return null;
  let pct = 0;
  if (running) {
    const onboard =
      snapshot?.active_command_id === pick.id && typeof snapshot.progress_pct === "number"
        ? snapshot.progress_pct
        : null;
    pct = onboard ?? (drone ? project(coords, drone).pct : 0);
  } else if (pick.status === "done") pct = 100;
  return { command: pick, coords, pct: Math.max(0, Math.min(100, pct)), running: running !== null };
}

/** Planned: the whole line, dashed, with numbered points. */
export function pathPlanData(p: ActivePath | null): FC {
  if (!p) return fc([]);
  const label = (p.command.params as { label?: unknown }).label;
  return fc([
    line(p.coords, { kind: "line", label: typeof label === "string" ? label : "" }),
    ...p.coords.map(
      (c, i): Feature => ({
        type: "Feature",
        geometry: { type: "Point", coordinates: c },
        properties: { kind: "point", n: String(i + 1) },
      }),
    ),
  ]);
}

/** Flown: the line up to the drone's place on it, solid. */
export function pathFlownData(p: ActivePath | null, drone: LatLon | null): FC {
  if (p && p.pct >= 100) return fc([line(p.coords)]);
  if (!p || !p.running || !drone) return fc([]);
  const { leg, along } = project(p.coords, drone);
  const flown = [...p.coords.slice(0, leg), along];
  return flown.length >= 2 ? fc([line(flown)]) : fc([]);
}
