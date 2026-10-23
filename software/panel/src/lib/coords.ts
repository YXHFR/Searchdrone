// A position in the three forms a ground team asks for: decimal degrees (WGS 84), degrees
// minutes seconds, and SWEREF 99 TM (EPSG:3006, the Swedish national grid). No library: the
// grid is Gauss-Krüger on GRS 80 (Lantmäteriet's formulas), good to well under a millimetre.

/** "59.288420, 18.159530" (6 decimals: about 0.1 m). */
export function decimalText(lat: number, lon: number): string {
  return `${lat.toFixed(6)}, ${lon.toFixed(6)}`;
}

function dmsPart(v: number, pos: string, neg: string): string {
  const abs = Math.abs(v);
  let d = Math.floor(abs);
  let m = Math.floor((abs - d) * 60);
  let s = Math.round(((abs - d) * 60 - m) * 600) / 10;
  if (s >= 60) {
    s = 0;
    m += 1;
  }
  if (m >= 60) {
    m = 0;
    d += 1;
  }
  return `${d}°${String(m).padStart(2, "0")}′${s.toFixed(1).padStart(4, "0")}″${v < 0 ? neg : pos}`;
}

/** "59°17′18.3″N 18°09′34.3″E" (a tenth of a second: about 3 m). */
export function dmsText(lat: number, lon: number): string {
  return `${dmsPart(lat, "N", "S")} ${dmsPart(lon, "E", "W")}`;
}

const A = 6_378_137;
const F = 1 / 298.257222101;
const LON0 = 15;
const K0 = 0.9996;
const FE = 500_000;

/** SWEREF 99 TM northing and easting in metres. */
export function sweref99tm(lat: number, lon: number): { n: number; e: number } {
  const e2 = F * (2 - F);
  const n = F / (2 - F);
  const aRoof = (A / (1 + n)) * (1 + n ** 2 / 4 + n ** 4 / 64);
  const ca = e2;
  const cb = (5 * e2 ** 2 - e2 ** 3) / 6;
  const cc = (104 * e2 ** 3 - 45 * e2 ** 4) / 120;
  const cd = (1237 * e2 ** 4) / 1260;
  const b1 = n / 2 - (2 * n ** 2) / 3 + (5 * n ** 3) / 16 + (41 * n ** 4) / 180;
  const b2 = (13 * n ** 2) / 48 - (3 * n ** 3) / 5 + (557 * n ** 4) / 1440;
  const b3 = (61 * n ** 3) / 240 - (103 * n ** 4) / 140;
  const b4 = (49561 * n ** 4) / 161280;
  const phi = (lat * Math.PI) / 180;
  const dl = ((lon - LON0) * Math.PI) / 180;
  const sin2 = Math.sin(phi) ** 2;
  const phiStar =
    phi - Math.sin(phi) * Math.cos(phi) * (ca + cb * sin2 + cc * sin2 ** 2 + cd * sin2 ** 3);
  const xi = Math.atan2(Math.tan(phiStar), Math.cos(dl));
  const eta = Math.atanh(Math.cos(phiStar) * Math.sin(dl));
  const sum = (f: (x: number) => number, g: (x: number) => number): number =>
    b1 * f(2 * xi) * g(2 * eta) +
    b2 * f(4 * xi) * g(4 * eta) +
    b3 * f(6 * xi) * g(6 * eta) +
    b4 * f(8 * xi) * g(8 * eta);
  return {
    n: K0 * aRoof * (xi + sum(Math.sin, Math.cosh)),
    e: K0 * aRoof * (eta + sum(Math.cos, Math.sinh)) + FE,
  };
}

/** Roughly Sweden's extent: the grid means nothing far outside it. */
export function inSweden(lat: number, lon: number): boolean {
  return lat >= 55 && lat <= 69.5 && lon >= 10.5 && lon <= 24.5;
}

/** "N 6576436 E 679958" (whole metres, as the grid is read out); null outside Sweden. */
export function swerefText(lat: number, lon: number): string | null {
  if (!inSweden(lat, lon)) return null;
  const { n, e } = sweref99tm(lat, lon);
  return `N ${Math.round(n)} E ${Math.round(e)}`;
}
