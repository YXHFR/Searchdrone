// Pure functions: store state → the GeoJSON each map layer shows. No MapLibre here, so this
// is unit-tested without WebGL and stays out of the map chunk's way.
import type { Annotation, Approval, Charter, Command, Telemetry } from "../../api/types";
import { areaM2, circleRing } from "../../lib/geo";
import type { SectorFeature, Track } from "../../state/missionStore";

export interface Feature {
  type: "Feature";
  geometry: { type: string; coordinates: unknown };
  properties: Record<string, unknown>;
}
export interface FC {
  type: "FeatureCollection";
  features: Feature[];
}

export const SOURCE_IDS = [
  "charter-area",
  "charter-preview",
  "sectors",
  "swept",
  "pilot-range",
  "home",
  "target",
  "track",
  "areas",
  "pins",
  "draw",
  "drone",
] as const;
export type SourceId = (typeof SOURCE_IDS)[number];

export const EMPTY: FC = { type: "FeatureCollection", features: [] };
const fc = (features: Feature[]): FC => ({ type: "FeatureCollection", features });
const point = (lon: number, lat: number, properties: Record<string, unknown> = {}): Feature => ({
  type: "Feature",
  geometry: { type: "Point", coordinates: [lon, lat] },
  properties,
});

const isPolygon = (g: unknown): g is { type: string; coordinates: unknown } => {
  const t = (g as { type?: string } | null)?.type;
  return t === "Polygon" || t === "MultiPolygon";
};

/** The drone: one point, grey and marked stale when the telemetry is older than 3 s. */
export function droneData(tel: Telemetry | null, ageMs: number | null): FC {
  if (!tel || !Number.isFinite(tel.lat) || !Number.isFinite(tel.lon)) return EMPTY;
  const stale = ageMs === null || ageMs > 3_000;
  return fc([
    point(tel.lon, tel.lat, {
      heading: tel.heading_deg ?? 0,
      alt: Math.round(tel.alt_rel_home_m ?? 0),
      stale,
      age_s: ageMs === null ? null : Math.round(ageMs / 1000),
    }),
  ]);
}

/** The approved charter's area (fill + edge). */
export function charterAreaData(c: Charter | null): FC {
  if (!c || !isPolygon(c.body.area)) return EMPTY;
  return fc([{ type: "Feature", geometry: c.body.area, properties: { version: c.version } }]);
}

/** Charters waiting on a card (new or changed), drawn dashed. */
export function charterPreviewData(approvals: Approval[], charters: Record<string, Charter>): FC {
  const out: Feature[] = [];
  for (const a of approvals) {
    if (a.kind !== "charter" || a.status !== "pending") continue;
    const c = charters[a.subject_id];
    if (c && isPolygon(c.body.area))
      out.push({ type: "Feature", geometry: c.body.area, properties: { version: c.version } });
  }
  return fc(out);
}

export function sectorsData(sectors: SectorFeature[]): FC {
  return fc(
    sectors
      .filter((s) => isPolygon(s.geometry))
      .map((s) => ({
        type: "Feature",
        geometry: s.geometry,
        properties: {
          sector_id: s.properties.sector_id,
          name: s.properties.name,
          status: s.properties.status,
          progress: s.properties.progress,
          label: `${s.properties.name} · ${Math.round(s.properties.progress * 100)} %`,
        },
      })),
  );
}

/** What the camera has actually covered, as the ground station reports it per sector. */
export function sweptData(sectors: SectorFeature[]): FC {
  return fc(
    sectors
      .filter((s) => isPolygon(s.properties.covered))
      .map((s) => ({
        type: "Feature",
        geometry: s.properties.covered as Feature["geometry"],
        properties: { sector_id: s.properties.sector_id },
      })),
  );
}

/** Pilot position plus the range circle (active charter, else the one waiting for a click). */
export function pilotRangeData(c: Charter | null): FC {
  if (!c) return EMPTY;
  const p = c.body.pilot_position;
  const r = c.body.max_range_from_pilot_m;
  const out: Feature[] = [point(p.lon, p.lat, { kind: "pilot" })];
  if (typeof r === "number" && r > 0)
    out.push({
      type: "Feature",
      geometry: { type: "Polygon", coordinates: [circleRing(p, r)] },
      properties: { kind: "range", radius_m: r },
    });
  return fc(out);
}

export function homeData(c: Charter | null): FC {
  return c ? fc([point(c.body.home.lon, c.body.home.lat)]) : EMPTY;
}

/** The current target: the running order's point or area, else a pending order card's. */
export function targetData(active: Command | null, pending: Command[]): FC {
  const pick = active ?? pending[0] ?? null;
  if (!pick) return EMPTY;
  const p = pick.params as { lat?: unknown; lon?: unknown; polygon?: unknown };
  if (typeof p.lat === "number" && typeof p.lon === "number")
    return fc([point(p.lon, p.lat, { skill: pick.skill })]);
  if (isPolygon(p.polygon))
    return fc([{ type: "Feature", geometry: p.polygon, properties: { skill: pick.skill } }]);
  return EMPTY;
}

export function trackData(track: Track): FC {
  if (track.length < 2) return EMPTY;
  return fc([
    { type: "Feature", geometry: { type: "LineString", coordinates: track }, properties: {} },
  ]);
}

const nfHa = new Intl.NumberFormat("sv-SE", { maximumFractionDigits: 1 });

export function areasData(anns: Annotation[]): FC {
  return fc(
    anns
      .filter((a) => a.kind === "area" && isPolygon(a.geometry))
      .map((a) => ({
        type: "Feature",
        geometry: a.geometry as Feature["geometry"],
        properties: {
          id: a.id,
          label: `${a.label} · ${nfHa.format((a.area_m2 ?? areaM2(a.geometry)) / 10_000)} ha`,
        },
      })),
  );
}

export function pinsData(anns: Annotation[]): FC {
  return fc(
    anns
      .filter((a) => a.kind === "pin" && (a.geometry as { type?: string }).type === "Point")
      .map((a) => ({
        type: "Feature",
        geometry: a.geometry as Feature["geometry"],
        properties: { id: a.id, label: a.label },
      })),
  );
}

/** The drawing in progress: the vertices so far and the line or polygon through them. */
export function drawData(points: [number, number][], closed: boolean): FC {
  const out: Feature[] = points.map(([lon, lat]) => point(lon, lat, { kind: "vertex" }));
  if (points.length >= 2) {
    const first = points[0] as [number, number];
    out.push(
      closed && points.length >= 3
        ? {
            type: "Feature",
            geometry: { type: "Polygon", coordinates: [[...points, first]] },
            properties: { kind: "shape" },
          }
        : {
            type: "Feature",
            geometry: { type: "LineString", coordinates: points },
            properties: { kind: "shape" },
          },
    );
  }
  return fc(out);
}

/** Axis-aligned rectangle from two opposite corners, as a closed ring. */
export function rectangle(a: [number, number], b: [number, number]): [number, number][] {
  return [a, [b[0], a[1]], b, [a[0], b[1]]];
}

/** Next free label: P1, P2 … for pins, ar-1, ar-2 … for areas. */
export function nextLabel(anns: Annotation[], kind: "pin" | "area"): string {
  const re = kind === "pin" ? /^P(\d+)$/ : /^ar-(\d+)$/;
  let max = 0;
  for (const a of anns) {
    const m = re.exec(a.label);
    if (m?.[1]) max = Math.max(max, Number(m[1]));
  }
  return kind === "pin" ? `P${max + 1}` : `ar-${max + 1}`;
}

/** Bounds [[w, s], [e, n]] of a polygon's outer ring, for "Visa hela ramen". */
export function bounds(geom: unknown): [[number, number], [number, number]] | null {
  const g = geom as { type?: string; coordinates?: number[][][] | number[][][][] } | null;
  if (!g?.coordinates) return null;
  const ring = (
    g.type === "MultiPolygon"
      ? (g.coordinates as number[][][][]).flat(1)
      : (g.coordinates as number[][][])
  ).flat(1);
  if (ring.length === 0) return null;
  const lons = ring.map((p) => p[0] ?? 0);
  const lats = ring.map((p) => p[1] ?? 0);
  return [
    [Math.min(...lons), Math.min(...lats)],
    [Math.max(...lons), Math.max(...lats)],
  ];
}
