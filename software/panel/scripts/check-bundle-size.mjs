// Main bundle = the entry chunk plus all its static imports, gzip level 9. Limits: main
// ≤ 350 kB, all JS ≤ 1 200 kB. MapLibre must stay out of the main bundle (lazy chunk).
import { readdirSync, readFileSync } from "node:fs";
import { join } from "node:path";
import { gzipSync } from "node:zlib";

const DIST = process.argv[2] ?? "dist";
const MAIN_LIMIT = 350 * 1024;
const TOTAL_LIMIT = 1200 * 1024;
const manifest = JSON.parse(readFileSync(join(DIST, ".vite/manifest.json"), "utf8"));
const gz = (file) => gzipSync(readFileSync(join(DIST, file)), { level: 9 }).length;

const entry = Object.keys(manifest).find((k) => manifest[k].isEntry);
const seen = new Set();
const walk = (key) => {
  const chunk = manifest[key];
  if (!chunk || seen.has(chunk.file)) return;
  seen.add(chunk.file);
  for (const imp of chunk.imports ?? []) walk(imp);
};
walk(entry);

const main = [...seen].reduce((a, f) => a + gz(f), 0);
const allJs = readdirSync(join(DIST, "assets"))
  .filter((f) => f.endsWith(".js"))
  .reduce((a, f) => a + gz(join("assets", f)), 0);
const mapInMain = [...seen].some((f) => readFileSync(join(DIST, f), "utf8").includes("maplibregl"));

const kb = (n) => `${(n / 1024).toFixed(1)} kB`;
console.log(
  `check-bundle-size: main ${kb(main)} (limit ${kb(MAIN_LIMIT)}), all JS ${kb(allJs)} (limit ${kb(TOTAL_LIMIT)})`,
);
if (main > MAIN_LIMIT || allJs > TOTAL_LIMIT || mapInMain) {
  if (mapInMain) console.error("MapLibre is in the main bundle");
  process.exit(1);
}
