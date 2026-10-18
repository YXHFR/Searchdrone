// Every t("key") used in src/ exists in sv.json, and no JSX text node holds letters outside t().
import { readdirSync, readFileSync, statSync } from "node:fs";
import { join } from "node:path";

const SRC = "src";
const sv = JSON.parse(readFileSync(join(SRC, "i18n/sv.json"), "utf8"));
const has = (key) =>
  key.split(".").reduce((o, k) => (o && typeof o === "object" ? o[k] : undefined), sv) !==
  undefined;

function files(dir) {
  return readdirSync(dir).flatMap((name) => {
    const p = join(dir, name);
    return statSync(p).isDirectory() ? files(p) : [p];
  });
}

const problems = [];
for (const file of files(SRC).filter((f) => /\.tsx?$/.test(f) && !/\.test\.tsx?$/.test(f))) {
  const text = readFileSync(file, "utf8");
  // t("a.b"), and plain "x.y.z" literals that look like i18n keys passed around as data
  // (not `case "chat.message":`, which is a live-event type).
  for (const m of text.matchAll(/\bt\(\s*"([a-zA-Z0-9_.]+)"/g))
    if (!has(m[1])) problems.push(`${file}: missing key ${m[1]}`);
  for (const m of text.matchAll(
    /(?<!case )"((?:safety|pills|orders|login|layout|views|status|resume|chat|manual)\.[a-zA-Z0-9_.]+)"/g,
  )) {
    if (!has(m[1])) problems.push(`${file}: missing key ${m[1]}`);
  }
  if (file.endsWith(".tsx")) {
    for (const m of text.matchAll(/>\s*([^<>{}\n]*[A-Za-zÅÄÖåäö]{2,}[^<>{}\n]*)\s*</g)) {
      // Skip code that only looks like text (comparisons, arrows, calls).
      if (!/^\s*\/\//.test(m[1]) && !/[=?;(&]/.test(m[1])) {
        problems.push(`${file}: JSX text outside t(): ${m[1].trim()}`);
      }
    }
  }
}
if (problems.length) {
  console.error(problems.join("\n"));
  process.exit(1);
}
console.log("check-i18n: ok");
