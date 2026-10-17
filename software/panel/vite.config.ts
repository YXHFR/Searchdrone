/// <reference types="vitest/config" />
import type { IncomingMessage } from "node:http";
import react from "@vitejs/plugin-react";
import { defineConfig, type Plugin, type ProxyOptions } from "vite";
// The same file sd-ground reads, so dev preview and the server send identical headers.
import securityHeaders from "../common/schemas/security_headers.json" with { type: "json" };

const TARGET = process.env.SD_GROUND_URL ?? "http://127.0.0.1:8080";
const DEV_ORIGIN = "http://127.0.0.1:5173";

interface HeaderSetter {
  setHeader(name: string, value: string): void;
}

// Rewrite Origin ONLY when it comes from the dev server itself. Any other Origin passes
// unchanged so sd-ground answers 403 ORIGIN_NOT_ALLOWED (WS: 4403). A missing Origin stays missing.
export function fixOrigin(proxyReq: HeaderSetter, req: IncomingMessage): void {
  if (req.headers.origin === DEV_ORIGIN) proxyReq.setHeader("origin", TARGET);
}

const proxy: ProxyOptions = {
  target: TARGET,
  changeOrigin: true,
  configure: (p) => {
    p.on("proxyReq", (proxyReq, req) => fixOrigin(proxyReq, req));
    p.on("proxyReqWs", (proxyReq, req) => fixOrigin(proxyReq, req));
  },
};

// The test build (mode "test") gets window.__sd hooks and a marker sd-ground refuses outside sim.
function testBuildMarker(enabled: boolean): Plugin {
  return {
    name: "sd-test-build-marker",
    transformIndexHtml: () =>
      enabled
        ? [{ tag: "meta", attrs: { name: "sd-build", content: "test" }, injectTo: "head" }]
        : [],
  };
}

export default defineConfig(({ mode }) => {
  const testBuild = mode === "test";
  return {
    plugins: [react(), testBuildMarker(testBuild)],
    define: {
      "import.meta.env.VITE_SD_TEST_HOOKS": JSON.stringify(testBuild ? "1" : "0"),
    },
    server: {
      host: "127.0.0.1",
      port: 5173,
      strictPort: true,
      proxy: { "/api": proxy, "/ws": { ...proxy, ws: true }, "/tiles": proxy },
    },
    preview: { host: "127.0.0.1", port: 4173, strictPort: true, headers: securityHeaders },
    // MapLibre (~1 MB) is its own lazy chunk; the main bundle limit is checked by check-bundle-size.
    // MapLibre's worker is an ES module; it is bundled as a module worker (src/views/map).
    worker: { format: "es" },
    build: { manifest: true, target: "es2022", sourcemap: false, chunkSizeWarningLimit: 1200 },
    test: {
      environment: "jsdom",
      include: ["src/**/*.test.{ts,tsx}", "e2e/helpers/**/*.test.ts"],
      setupFiles: ["src/test/setup.ts"],
    },
  };
});
