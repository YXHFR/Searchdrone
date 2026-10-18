// Small geometry helpers for areas and distances of a few kilometres (charter areas, pins,
// the pilot circle). Accurate to well under a percent at that size; no extra library.
const R = 6_371_008.8; // mean Earth radius (m)
const rad = (d: number): number => (d * Math.PI) / 180;

export interface LatLon {
  lat: number;
  lon: number;
}

/** Great-circle distance in metres. */
export function distanceM(a: LatLon, b: LatLon): number {
  const dLat = rad(b.lat - a.lat);
  const dLon = rad(b.lon - a.lon);
  const h =
    Math.sin(dLat / 2) ** 2 + Math.cos(rad(a.lat)) * Math.cos(rad(b.lat)) * Math.sin(dLon / 2) ** 2;
  return 2 * R * Math.asin(Math.min(1, Math.sqrt(h)));
}

type Ring = number[][];

function rings(geom: unknown): Ring[][] {
  const g = geom as { type?: string; coordinates?: unknown } | null;
  if (!g || !Array.isArray(g.coordinates)) return [];
  if (g.type === "Polygon") return [g.coordinates as Ring[]];
  if (g.type === "MultiPolygon") return g.coordinates as Ring[][];
  return [];
}

/** Signed area of one ring (m²), on a local equirectangular projection around its first point. */
function ringArea(ring: Ring): number {
  const first = ring[0];
  if (!first || ring.length < 4) return 0;
  const lat0 = rad(first[1] ?? 0);
  const xy = ring.map((p) => [rad(p[0] ?? 0) * R * Math.cos(lat0), rad(p[1] ?? 0) * R]);
  let sum = 0;
  for (let i = 0; i < xy.length - 1; i += 1) {
    const [x1, y1] = xy[i] as number[];
    const [x2, y2] = xy[i + 1] as number[];
    sum += (x1 ?? 0) * (y2 ?? 0) - (x2 ?? 0) * (y1 ?? 0);
  }
  return sum / 2;
}

/** Area of a GeoJSON Polygon or MultiPolygon in m² (holes subtracted). */
export function areaM2(geom: unknown): number {
  let total = 0;
  for (const poly of rings(geom)) {
    poly.forEach((ring, i) => {
      const a = Math.abs(ringArea(ring));
      total += i === 0 ? a : -a;
    });
  }
  return total;
}

/** Centre of the outer rings' vertices (good enough to place a label or measure from home). */
export function centroid(geom: unknown): LatLon | null {
  const pts = rings(geom).flatMap((poly) => (poly[0] ?? []).slice(0, -1));
  if (pts.length === 0) return null;
  const lon = pts.reduce((s, p) => s + (p[0] ?? 0), 0) / pts.length;
  const lat = pts.reduce((s, p) => s + (p[1] ?? 0), 0) / pts.length;
  return { lat, lon };
}

/** Width (east-west) and height (north-south) of the bounding box in metres. */
export function extentM(geom: unknown): { w: number; h: number } | null {
  const pts = rings(geom).flatMap((poly) => poly[0] ?? []);
  if (pts.length === 0) return null;
  const lons = pts.map((p) => p[0] ?? 0);
  const lats = pts.map((p) => p[1] ?? 0);
  const minLat = Math.min(...lats);
  const maxLat = Math.max(...lats);
  const midLat = (minLat + maxLat) / 2;
  return {
    w: distanceM({ lat: midLat, lon: Math.min(...lons) }, { lat: midLat, lon: Math.max(...lons) }),
    h: distanceM({ lat: minLat, lon: lons[0] ?? 0 }, { lat: maxLat, lon: lons[0] ?? 0 }),
  };
}

/** Does any pair of non-adjacent edges of the ring cross? (`[lon, lat]`, closed ring.) */
export function selfIntersects(ring: Ring): boolean {
  const n = ring.length - 1;
  const seg = (i: number): [number[], number[]] => [ring[i] ?? [], ring[(i + 1) % n] ?? []];
  const cross = (o: number[], a: number[], b: number[]): number =>
    ((a[0] ?? 0) - (o[0] ?? 0)) * ((b[1] ?? 0) - (o[1] ?? 0)) -
    ((a[1] ?? 0) - (o[1] ?? 0)) * ((b[0] ?? 0) - (o[0] ?? 0));
  for (let i = 0; i < n; i += 1) {
    for (let j = i + 1; j < n; j += 1) {
      if (j === i + 1 || (i === 0 && j === n - 1)) continue;
      const [p1, p2] = seg(i);
      const [p3, p4] = seg(j);
      const d1 = cross(p3, p4, p1);
      const d2 = cross(p3, p4, p2);
      const d3 = cross(p1, p2, p3);
      const d4 = cross(p1, p2, p4);
      if (d1 * d2 < 0 && d3 * d4 < 0) return true;
    }
  }
  return false;
}

/** A circle as a closed GeoJSON ring of `steps` points around `c` with radius `m`. */
export function circleRing(c: LatLon, m: number, steps = 64): number[][] {
  const out: number[][] = [];
  for (let i = 0; i <= steps; i += 1) {
    const a = (2 * Math.PI * i) / steps;
    const dLat = (m * Math.cos(a)) / R;
    const dLon = (m * Math.sin(a)) / (R * Math.cos(rad(c.lat)));
    out.push([c.lon + (dLon * 180) / Math.PI, c.lat + (dLat * 180) / Math.PI]);
  }
  return out;
}
