// A made-up aerial photo for the tests and screenshots: forest, fields, open ground and a
// gravel road drawn from noise in code, packed as a raster PMTiles archive. No real imagery.
// Every pixel is computed from its position in metres, so tiles and zoom levels line up.
import { crc32, deflateSync, gzipSync } from "node:zlib";
import { zxyToTileId } from "pmtiles";

const HOME = { lat: -35.363261, lon: 149.16523 };
const SPAN = { lat: 0.01, lon: 0.012 };
const MIN_ZOOM = 12;
const MAX_ZOOM = 16;
const SIZE = 256;
const EARTH = 40_075_016.686;
const HOME_X = (HOME.lon / 360) * EARTH;
const HOME_Y = (Math.asinh(Math.tan((HOME.lat * Math.PI) / 180)) / (2 * Math.PI)) * EARTH;
export const IMAGERY_ATTRIBUTION = "Syntetisk testbild, genererad i koden";
/** MapConfig.imagery for the fake ground, as sd-ground sends it for a photo pack. */
export const IMAGERY = {
  pack_url: "/tiles/imagery.pmtiles",
  maxzoom: 16,
  attribution: IMAGERY_ATTRIBUTION,
};

function hash(cx: number, cy: number): number {
  let h = Math.imul(cx, 374_761_393) ^ Math.imul(cy, 668_265_263);
  h = Math.imul(h ^ (h >>> 13), 1_274_126_177);
  return ((h ^ (h >>> 16)) >>> 0) / 4_294_967_296;
}

function noise(x: number, y: number): number {
  const cx = Math.floor(x);
  const cy = Math.floor(y);
  const fx = x - cx;
  const fy = y - cy;
  const sx = fx * fx * (3 - 2 * fx);
  const sy = fy * fy * (3 - 2 * fy);
  const top = hash(cx, cy) + (hash(cx + 1, cy) - hash(cx, cy)) * sx;
  const bottom = hash(cx, cy + 1) + (hash(cx + 1, cy + 1) - hash(cx, cy + 1)) * sx;
  return top + (bottom - top) * sy;
}

function fbm(x: number, y: number, octaves: number): number {
  let sum = 0;
  let amp = 0.5;
  let norm = 0;
  for (let i = 0; i < octaves; i++) {
    sum += noise(x, y) * amp;
    norm += amp;
    x = x * 2.03 + 17.1;
    y = y * 2.03 - 9.7;
    amp *= 0.5;
  }
  return sum / norm;
}

type Rgb = [number, number, number];
const CROPS: Rgb[] = [
  [146, 140, 86],
  [104, 124, 62],
  [166, 156, 110],
  [122, 134, 76],
  [88, 110, 56],
  [150, 128, 92],
];
const ROT = 0.35;

const mix = (a: Rgb, b: Rgb, t: number): Rgb => [
  a[0] + (b[0] - a[0]) * t,
  a[1] + (b[1] - a[1]) * t,
  a[2] + (b[2] - a[2]) * t,
];
const smooth = (e0: number, e1: number, v: number) => {
  const t = Math.min(1, Math.max(0, (v - e0) / (e1 - e0)));
  return t * t * (3 - 2 * t);
};

/** Forest: dark crowns with gaps, conifer and broadleaf in drifts. */
function forest(x: number, y: number): Rgb {
  const kind = fbm(x / 240, y / 240, 2);
  const base = mix([26, 44, 30], [56, 76, 40], smooth(0.35, 0.65, kind));
  const crown = noise(x / 2.4, y / 2.4) * noise(x / 6.5 + 3, y / 6.5 - 5);
  const shade = 0.35 + crown * 1.5;
  return [base[0] * shade, base[1] * shade, base[2] * shade];
}

/** Farmland: straight parcels in a tilted grid, crop rows, darker hedges along the edges. */
function fields(x: number, y: number): Rgb {
  const u = x * Math.cos(ROT) + y * Math.sin(ROT);
  const v = -x * Math.sin(ROT) + y * Math.cos(ROT);
  const cu = Math.floor(u / 170);
  const depth = 70 + hash(cu, 7) * 160;
  const shift = hash(cu, 11) * depth;
  const cv = Math.floor((v + shift) / depth);
  const crop = CROPS[Math.floor(hash(cu, cv) * CROPS.length)] as Rgb;
  const across = hash(cv, cu) < 0.5;
  const rows = Math.sin((across ? u : v) / 1.8) * 5;
  const patchy = (fbm(x / 40, y / 40, 3) - 0.5) * 22;
  const vv = v + shift;
  const edge = Math.min(u - cu * 170, (cu + 1) * 170 - u, vv - cv * depth, (cv + 1) * depth - vv);
  const c: Rgb = [crop[0] + rows + patchy, crop[1] + rows + patchy, crop[2] + rows + patchy];
  return edge < 2.2 ? mix(c, forest(x, y), 0.8) : c;
}

/** The ground colour at a point, in metres east and north in web mercator. */
function ground(x: number, y: number): Rgb {
  const ex = x - HOME_X;
  const road = Math.abs(
    y - HOME_Y - 380 * Math.sin(ex / 520) - 120 * Math.sin(ex / 140 + 1.3) - 160,
  );
  const land = fbm(x / 600, y / 600, 5);
  let c = mix(fields(x, y), forest(x, y), smooth(0.53, 0.47, land));
  const bare = smooth(0.7, 0.74, land);
  if (bare > 0) {
    const rock = fbm(x / 25, y / 25, 3);
    c = mix(c, [128 + rock * 40, 122 + rock * 34, 104 + rock * 28], bare);
  }
  if (road < 3) c = [182, 170, 140];
  else if (road < 4.5) c = mix(c, [182, 170, 140], 0.4);
  const light = 0.9 + fbm(x / 900, y / 900, 2) * 0.2;
  const grain = (noise(x / 1.3, y / 1.3) - 0.5) * 12;
  return [c[0] * light + grain, c[1] * light + grain, c[2] * light + grain];
}

function png(rgb: Uint8ClampedArray): Buffer {
  const rows = Buffer.alloc((SIZE * 3 + 1) * SIZE);
  for (let r = 0; r < SIZE; r++)
    rows.set(rgb.subarray(r * SIZE * 3, (r + 1) * SIZE * 3), r * (SIZE * 3 + 1) + 1);
  const chunk = (type: string, data: Buffer) => {
    const head = Buffer.alloc(8);
    head.writeUInt32BE(data.length, 0);
    head.write(type, 4, "ascii");
    const crc = Buffer.alloc(4);
    crc.writeUInt32BE(crc32(Buffer.concat([head.subarray(4), data])), 0);
    return Buffer.concat([head, data, crc]);
  };
  const ihdr = Buffer.alloc(13);
  ihdr.writeUInt32BE(SIZE, 0);
  ihdr.writeUInt32BE(SIZE, 4);
  ihdr.set([8, 2, 0, 0, 0], 8);
  return Buffer.concat([
    Buffer.from([0x89, 0x50, 0x4e, 0x47, 0x0d, 0x0a, 0x1a, 0x0a]),
    chunk("IHDR", ihdr),
    chunk("IDAT", deflateSync(rows, { level: 6 })),
    chunk("IEND", Buffer.alloc(0)),
  ]);
}

function tile(z: number, tx: number, ty: number): Buffer {
  const rgb = new Uint8ClampedArray(SIZE * SIZE * 3);
  const scale = EARTH / 2 ** z / SIZE;
  for (let py = 0; py < SIZE; py++) {
    const y = EARTH / 2 - (ty * SIZE + py + 0.5) * scale;
    for (let px = 0; px < SIZE; px++) {
      const x = (tx * SIZE + px + 0.5) * scale - EARTH / 2;
      const [r, g, b] = ground(x, y);
      const i = (py * SIZE + px) * 3;
      rgb[i] = r;
      rgb[i + 1] = g;
      rgb[i + 2] = b;
    }
  }
  return png(rgb);
}

const tileX = (z: number, lon: number) => Math.floor(((lon + 180) / 360) * 2 ** z);
const tileY = (z: number, lat: number) => {
  const r = (lat * Math.PI) / 180;
  return Math.floor(((1 - Math.asinh(Math.tan(r)) / Math.PI) / 2) * 2 ** z);
};

function varints(values: number[]): Buffer {
  const out: number[] = [];
  for (let v of values) {
    while (v >= 0x80) {
      out.push((v % 0x80) | 0x80);
      v = Math.floor(v / 0x80);
    }
    out.push(v);
  }
  return Buffer.from(out);
}

/** A PMTiles v3 archive: header, root directory, metadata, then PNG tiles in id order. */
function pack(): Buffer {
  const lon0 = HOME.lon - SPAN.lon;
  const lon1 = HOME.lon + SPAN.lon;
  const lat0 = HOME.lat - SPAN.lat;
  const lat1 = HOME.lat + SPAN.lat;
  const tiles: { id: number; data: Buffer }[] = [];
  for (let z = MIN_ZOOM; z <= MAX_ZOOM; z++)
    for (let x = tileX(z, lon0); x <= tileX(z, lon1); x++)
      for (let y = tileY(z, lat1); y <= tileY(z, lat0); y++)
        tiles.push({ id: zxyToTileId(z, x, y), data: tile(z, x, y) });
  tiles.sort((a, b) => a.id - b.id);

  let offset = 0;
  const ids: number[] = [];
  let last = 0;
  for (const t of tiles) {
    ids.push(t.id - last);
    last = t.id;
  }
  const lengths = tiles.map((t) => t.data.length);
  const offsets = tiles.map((t, i) => {
    const o = offset;
    offset += t.data.length;
    return i === 0 ? o + 1 : 0;
  });
  const dir = gzipSync(
    varints([tiles.length, ...ids, ...tiles.map(() => 1), ...lengths, ...offsets]),
  );
  const meta = gzipSync(
    JSON.stringify({ name: "searchdrone-test-imagery", attribution: IMAGERY_ATTRIBUTION }),
  );
  const data = Buffer.concat(tiles.map((t) => t.data));

  const h = Buffer.alloc(127);
  h.write("PMTiles", 0, "ascii");
  h[7] = 3;
  const u64 = (at: number, v: number) => h.writeBigUInt64LE(BigInt(v), at);
  u64(8, 127);
  u64(16, dir.length);
  u64(24, 127 + dir.length);
  u64(32, meta.length);
  u64(40, 0);
  u64(48, 0);
  u64(56, 127 + dir.length + meta.length);
  u64(64, data.length);
  u64(72, tiles.length);
  u64(80, tiles.length);
  u64(88, tiles.length);
  h.set([1, 2, 1, 2, MIN_ZOOM, MAX_ZOOM], 96); // clustered, gzip directories, raw PNG tiles
  const e7 = (v: number) => Math.round(v * 1e7);
  h.writeInt32LE(e7(lon0), 102);
  h.writeInt32LE(e7(lat0), 106);
  h.writeInt32LE(e7(lon1), 110);
  h.writeInt32LE(e7(lat1), 114);
  h[118] = 14;
  h.writeInt32LE(e7(HOME.lon), 119);
  h.writeInt32LE(e7(HOME.lat), 123);
  return Buffer.concat([h, dir, meta, data]);
}

let cached: Buffer | null = null;

/** Built on first use and kept for the worker's lifetime (about a second of work). */
export function syntheticImagery(): Buffer {
  cached ??= pack();
  return cached;
}
