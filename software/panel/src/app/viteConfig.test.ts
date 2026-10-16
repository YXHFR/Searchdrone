// @vitest-environment node
import { describe, expect, it } from "vitest";
import securityHeaders from "../../../common/schemas/security_headers.json" with { type: "json" };
import configFn, { fixOrigin } from "../../vite.config";

type Cfg = {
  server: { host: string; allowedHosts?: unknown; proxy: Record<string, unknown> };
  preview: { host: string; headers: Record<string, string> };
};

const load = (mode: string): Cfg =>
  (configFn as unknown as (e: { mode: string; command: string }) => Cfg)({
    mode,
    command: "serve",
  });

describe("vite.config", () => {
  it("binds dev and preview to loopback only and never sets allowedHosts", () => {
    const c = load("development");
    expect(c.server.host).toBe("127.0.0.1");
    expect(c.preview.host).toBe("127.0.0.1");
    expect(c.server.allowedHosts).toBeUndefined();
  });

  it("proxies exactly /api, /ws and /tiles", () => {
    expect(Object.keys(load("development").server.proxy).sort()).toEqual(["/api", "/tiles", "/ws"]);
  });

  it("preview sends the shared security headers", () => {
    expect(load("production").preview.headers).toEqual(securityHeaders);
  });

  it("rewrites Origin only for the dev server's own origin", () => {
    const set: Record<string, string> = {};
    const req = (origin?: string) => ({ headers: origin ? { origin } : {} }) as never;
    fixOrigin({ setHeader: (k, v) => (set[k] = v) }, req("http://127.0.0.1:5173"));
    expect(set.origin).toBe("http://127.0.0.1:8080");
    const other: Record<string, string> = {};
    fixOrigin({ setHeader: (k, v) => (other[k] = v) }, req("http://127.0.0.1:3000"));
    fixOrigin({ setHeader: (k, v) => (other[k] = v) }, req());
    expect(other).toEqual({});
  });
});
