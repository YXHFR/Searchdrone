// Screenshots of the built status page at phone and desktop size, and a check that the
// demo film plays in Chromium.
//
//   pnpm --filter @searchdrone/panel exec node ../../tools/status_page/screenshot.mjs \
//     <site dir> <screenshot dir>
//
// Runs in software/panel (pnpm --filter ... exec), whose Playwright it uses. Exits 1 when
// the page has a film that does not play.

import { mkdirSync, readFileSync } from "node:fs";
import { createServer } from "node:http";
import { createRequire } from "node:module";
import { extname, join, resolve, sep } from "node:path";

const require = createRequire(join(process.cwd(), "package.json"));
const { chromium } = require("@playwright/test");

const [site, shots] = process.argv.slice(2).map((p) => resolve(process.env.INIT_CWD ?? ".", p));
if (!site || !shots) {
  console.error("användning: screenshot.mjs <sajt> <bilder>");
  process.exit(2);
}
mkdirSync(shots, { recursive: true });

const TYPES = {
  ".html": "text/html; charset=utf-8",
  ".json": "application/json",
  ".mp4": "video/mp4",
  ".webm": "video/webm",
};

// A static server with byte ranges, as GitHub Pages serves the film.
const server = createServer((req, res) => {
  const path = decodeURIComponent(new URL(req.url, "http://x").pathname);
  const file = resolve(site, `.${path.endsWith("/") ? `${path}index.html` : path}`);
  let body = null;
  if (file.startsWith(site + sep)) {
    try {
      body = readFileSync(file); // a folder or a missing file throws
    } catch {
      body = null;
    }
  }
  if (body === null) {
    res.writeHead(404).end();
    return;
  }
  const type = TYPES[extname(file)] ?? "application/octet-stream";
  const m = /^bytes=(\d+)-(\d*)$/.exec(req.headers.range ?? "");
  if (m) {
    const start = Number(m[1]);
    const end = m[2] ? Math.min(Number(m[2]), body.length - 1) : body.length - 1;
    res.writeHead(206, {
      "content-type": type,
      "content-range": `bytes ${start}-${end}/${body.length}`,
      "accept-ranges": "bytes",
    });
    res.end(body.subarray(start, end + 1));
    return;
  }
  res.writeHead(200, { "content-type": type, "accept-ranges": "bytes" }).end(body);
});
await new Promise((ok) => server.listen(0, "127.0.0.1", ok));
const url = `http://127.0.0.1:${server.address().port}/`;

const browser = await chromium.launch(
  process.env.SD_CHROMIUM_PATH ? { executablePath: process.env.SD_CHROMIUM_PATH } : {},
);
let code = 0;
try {
  const sizes = [
    ["mobil", { width: 375, height: 812 }, true],
    ["dator", { width: 1280, height: 800 }, false],
  ];
  for (const [name, viewport, isMobile] of sizes) {
    const page = await browser.newPage({ viewport, isMobile, hasTouch: isMobile });
    await page.goto(url);
    const overflow = await page.evaluate(
      () => document.documentElement.scrollWidth > document.documentElement.clientWidth,
    );
    if (overflow) {
      console.error(`${name}: sidan är bredare än skärmen`);
      code = 1;
    }
    await page.screenshot({ path: join(shots, `status-${name}.png`), fullPage: true });
    console.log(`${name}: ${join(shots, `status-${name}.png`)}`);
    await page.close();
  }

  const page = await browser.newPage({ viewport: { width: 1280, height: 800 } });
  await page.goto(url);
  if ((await page.locator("video").count()) > 0) {
    const played = await page.evaluate(async () => {
      const v = document.querySelector("video");
      await v.play();
      const until = Date.now() + 15_000;
      while (v.currentTime < 1 && Date.now() < until) {
        await new Promise((r) => setTimeout(r, 200));
      }
      return { src: v.currentSrc, time: v.currentTime, size: [v.videoWidth, v.videoHeight] };
    });
    console.log(`filmen: ${played.src} spelad till ${played.time.toFixed(1)} s, ${played.size}`);
    if (played.time < 1 || played.size[0] === 0) {
      console.error("filmen spelas inte upp i Chromium");
      code = 1;
    }
    await page.screenshot({ path: join(shots, "status-film.png") });
  } else {
    console.log("ingen film på sidan");
  }
} finally {
  await browser.close();
  server.close();
}
process.exit(code);
