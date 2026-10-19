// The proposal mini map's projection: fit every point into a box, north up, metres square.
export type Ring = number[][];

export function outerRing(geom: unknown): Ring | null {
  const g = geom as { type?: string; coordinates?: unknown } | null;
  if (g?.type === "Polygon") return ((g.coordinates as Ring[])?.[0] ?? null) as Ring | null;
  if (g?.type === "MultiPolygon")
    return ((g.coordinates as Ring[][])?.[0]?.[0] ?? null) as Ring | null;
  return null;
}

export interface Fit {
  x(lon: number): number;
  y(lat: number): number;
  /** Metres per pixel. */
  mpp: number;
  /** A round scale-bar length that is at least 40 px long. */
  scaleM: number;
}

const M_PER_DEG = 111_320;
const SCALES = [10, 20, 50, 100, 200, 500, 1000, 2000, 5000];

/** Null when there is nothing to draw. Never zooms in past 0.5 m per pixel. */
export function fit(points: number[][], w: number, h: number, pad: number): Fit | null {
  if (points.length === 0) return null;
  const lat0 = ((points.reduce((a, p) => a + (p[1] ?? 0), 0) / points.length) * Math.PI) / 180;
  const mx = (lon: number): number => lon * M_PER_DEG * Math.cos(lat0);
  const my = (lat: number): number => lat * M_PER_DEG;
  const xs = points.map((p) => mx(p[0] ?? 0));
  const ys = points.map((p) => my(p[1] ?? 0));
  const [minX, maxX, minY, maxY] = [
    Math.min(...xs),
    Math.max(...xs),
    Math.min(...ys),
    Math.max(...ys),
  ];
  const mpp = Math.max((maxX - minX) / (w - 2 * pad), (maxY - minY) / (h - 2 * pad), 0.5);
  const cx = (minX + maxX) / 2;
  const cy = (minY + maxY) / 2;
  return {
    x: (lon) => w / 2 + (mx(lon) - cx) / mpp,
    y: (lat) => h / 2 - (my(lat) - cy) / mpp,
    mpp,
    scaleM: SCALES.find((m) => m / mpp >= 40) ?? 5000,
  };
}
