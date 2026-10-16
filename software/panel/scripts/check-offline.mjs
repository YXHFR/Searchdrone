// The panel must work without internet: no external URLs in the production build, and no
// test hooks. URLs in the allow-list are text inside libraries (XML namespaces, links in
// error messages); none of them is ever fetched. Exit 1 on any other finding.
import { readdirSync, readFileSync, statSync } from "node:fs";
import { join } from "node:path";

const DIST = process.argv[2] ?? "dist";
const ALLOWED = [
  "http://www.w3.org/", // SVG/XML/MathML namespaces
  "https://react.dev/errors/", // React error decoder link in messages
  "https://react.i18next.com/", // react-i18next warning text
  "https://maplibre.org/", // MapLibre attribution text and error messages
  "https://maplibre.invalid/", // MapLibre placeholder base URL, never resolved
  "https://wiki.openstreetmap.org/wiki/This_map_requires_WebGL", // MapLibre no-WebGL message
  "https://github.com/mapbox/mapbox-gl-js/issues/", // MapLibre comment link in an error text
  "https://github.com/protomaps/PMTiles/tree/main/js", // PMTiles Leaflet error text (unused)
  "https://github.com/maplibre/maplibre-gl-js/blob/", // licence link in the worker's header comment
];
const FORBIDDEN = ["__sd", 'name="sd-build"', "navigator.geolocation", '"crash"'];

function files(dir) {
  return readdirSync(dir).flatMap((name) => {
    const p = join(dir, name);
    return statSync(p).isDirectory() ? files(p) : [p];
  });
}

const problems = [];
for (const file of files(DIST).filter((f) => /\.(js|css|html)$/.test(f))) {
  const text = readFileSync(file, "utf8");
  // http(s) and ws(s) URLs, and protocol-relative "//host/..." string literals.
  for (const m of text.matchAll(
    /(?:https?|wss?):\/\/[^\s"'`)<>]+|(?<=["'`])\/\/[a-z0-9-]+\.[a-z0-9.-]+[^\s"'`)<>]*/gi,
  )) {
    if (!ALLOWED.some((a) => m[0].startsWith(a))) problems.push(`${file}: external URL ${m[0]}`);
  }
  for (const f of FORBIDDEN) if (text.includes(f)) problems.push(`${file}: forbidden text ${f}`);
}
if (problems.length) {
  console.error(problems.join("\n"));
  process.exit(1);
}
console.log(`check-offline: ok (${DIST})`);
