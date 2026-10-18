// Fake sd-ground for the mock end-to-end tests: page.route for /api/v1/** and /tiles/**
// (the synthetic test pack with HTTP Range, plus a stand-in sprite), page.routeWebSocket for
// /ws/v1/live. Every test also checks, when it ends, that the panel
// made no request to another host, read no geolocation, sent the CSRF header on every
// changing request, never sent Idempotency-Key, and called no path the fake does not know.
import { existsSync, readFileSync } from "node:fs";
import { fileURLToPath } from "node:url";
import {
  test as base,
  expect,
  type Page,
  type Request,
  type Route,
  type WebSocketRoute,
} from "@playwright/test";
import type {
  Annotation,
  Approval,
  Charter,
  Clearance,
  Command,
  Mission,
  PanelConfig,
  Session,
} from "@searchdrone/schemas";
import {
  dispatched,
  MISSION,
  mission,
  outcome,
  panelConfig,
  pilotSession,
  systemStatus,
  VEHICLE,
  vehicleLive,
} from "../fixtures";

export interface SeenRequest {
  method: string;
  path: string;
  headers: Record<string, string>;
  body: unknown;
}

type Handler = (
  req: SeenRequest,
) => { status: number; body?: unknown; headers?: Record<string, string> } | "hang";

const API = "/api/v1";
const FIXTURE_PACK = fileURLToPath(new URL("../fixtures/fixture.pmtiles", import.meta.url));
// A 1×1 transparent PNG: the sprite sheet stand-in (the test pack has no icons).
const PNG_1PX = Buffer.from(
  "iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAYAAAAfFcSJAAAADUlEQVR4nGNgYGBgAAAABQABpfZFQAAAAABJRU5ErkJggg==",
  "base64",
);

/** An empty glyph range in the MapLibre glyph PBF format (the test pack has no names). */
function emptyGlyphs(fontstack: string, range: string): Buffer {
  const field = (tag: number, bytes: Buffer) =>
    Buffer.concat([Buffer.from([tag, bytes.length]), bytes]);
  const stack = Buffer.concat([
    field(0x0a, Buffer.from(fontstack)),
    field(0x12, Buffer.from(range)),
  ]);
  return field(0x0a, stack);
}

const SNAPSHOT = fileURLToPath(
  new URL("../../../ground/tests/contract/snapshots/panel.openapi.json", import.meta.url),
);

/** Contract guard: once the ground server's OpenAPI snapshot exists, unknown paths fail. */
function contractPaths(): RegExp[] | null {
  if (!existsSync(SNAPSHOT)) return null;
  const doc = JSON.parse(readFileSync(SNAPSHOT, "utf8")) as {
    paths: Record<string, Record<string, unknown>>;
  };
  return Object.entries(doc.paths).flatMap(([p, ops]) =>
    Object.keys(ops).map(
      (m) => new RegExp(`^${m.toUpperCase()} ${p.replace(/\{[^}]+\}/g, "[^/]+")}$`),
    ),
  );
}

export class FakeGround {
  readonly requests: SeenRequest[] = [];
  readonly external: string[] = [];
  readonly unknown: string[] = [];
  readonly cspErrors: string[] = [];
  /** Every /tiles/ request and its status (glyph and tile failures make a test fail). */
  readonly tiles: { path: string; status: number }[] = [];
  session: Session | null = pilotSession;
  config: PanelConfig = structuredClone(panelConfig);
  private overrides: { match: string; handler: Handler; times: number }[] = [];
  private sockets: WebSocketRoute[] = [];
  private seq = 0;
  private vehicleState: unknown = vehicleLive;
  private stateAnswered = 0;
  private actionCounter = 0;
  private readonly contract = contractPaths();
  /** Mission data the fake keeps, so the panel's resync and the live events agree. */
  mission: Mission = structuredClone(mission);
  charters: Charter[] = [];
  approvals: Approval[] = [];
  commands: Command[] = [];
  annotations: Annotation[] = [];
  clearance: Clearance | null = null;
  sectors: unknown = { type: "FeatureCollection", features: [] };
  track: number[][] = [];
  /** Messages the panel sent on its live sockets (subscribe, ping, manual). */
  readonly wsSent: Record<string, unknown>[] = [];

  constructor(
    readonly page: Page,
    private readonly origin: string,
  ) {}

  async install(): Promise<void> {
    const page = this.page;
    // Count every access to navigator.geolocation (must stay 0 in v1).
    await page.addInitScript(() => {
      const w = window as unknown as { __geoAccess: number };
      w.__geoAccess = 0;
      const orig = Object.getOwnPropertyDescriptor(Navigator.prototype, "geolocation");
      Object.defineProperty(Navigator.prototype, "geolocation", {
        configurable: true,
        get() {
          w.__geoAccess += 1;
          return orig?.get?.call(this);
        },
      });
    });
    page.on("console", (msg) => {
      if (/Content[- ]Security[- ]Policy/i.test(msg.text())) this.cspErrors.push(msg.text());
    });
    // Anything not on the test origin is recorded and blocked.
    await page.route(
      (url) => url.origin !== this.origin && url.protocol.startsWith("http"),
      (route) => {
        this.external.push(route.request().url());
        return route.abort();
      },
    );
    await page.route(`${this.origin}/api/v1/**`, (route) => this.handle(route));
    await page.route(`${this.origin}/tiles/**`, (route) => this.handleTiles(route));
    const host = new URL(this.origin).host;
    await page.routeWebSocket(
      (url) => url.host !== host,
      (ws) => {
        this.external.push(ws.url());
        return ws.close();
      },
    );
    await page.routeWebSocket(
      (url) => url.host === host && url.pathname === "/ws/v1/live",
      (ws) => this.onSocket(ws),
    );
  }

  // --- control from tests ------------------------------------------------------------

  setSession(s: Session | null): void {
    this.session = s;
  }

  setConfig(patch: Partial<PanelConfig>): void {
    this.config = { ...this.config, ...patch };
  }

  /** Answer the next `times` requests to `METHOD /path` with `handler`. */
  respond(match: string, handler: Handler, times = 1): void {
    this.overrides.push({ match, handler, times });
  }

  failNext(match: string, status: number, body: unknown = {}): void {
    this.respond(match, () => ({ status, body }));
  }

  hangNext(match: string, n: number): void {
    this.respond(match, () => "hang", n);
  }

  /** Wait until the page has its live connection open (pushLive only reaches open ones)
   *  and has been answered the vehicle state it loads on connect, so that answer cannot
   *  land after a pushed state. */
  async liveConnected(): Promise<void> {
    await expect.poll(() => this.sockets.length).toBeGreaterThan(0);
    await expect.poll(() => this.stateAnswered).toBeGreaterThan(0);
    await this.page.waitForTimeout(100); // the panel reads the answer asynchronously
  }

  /** The panel has finished its resync (the track is the last thing it loads). */
  async resynced(): Promise<void> {
    await expect
      .poll(() => this.calls(/^GET \/api\/v1\/missions\/[^/]+\/track/).length)
      .toBeGreaterThan(0);
    await this.page.waitForTimeout(100);
  }

  pushLive(topic: string, type: string, data: unknown): void {
    // Like the real ground: the REST state is the newest pushed one, so the panel's
    // resync on connect cannot bring back an older state.
    if (topic === "vehicle" && type === "vehicle.state") this.vehicleState = data;
    this.remember(type, data);
    // Periodic events (telemetry, vehicle.state) carry no seq, as on the real bus.
    const periodic = topic === "telemetry" || type === "vehicle.state";
    if (!periodic) this.seq += 1;
    const msg = JSON.stringify({
      seq: periodic ? null : this.seq,
      ts: new Date().toISOString(),
      topic,
      type,
      vehicle_id: VEHICLE,
      mission_id: MISSION,
      data,
    });
    for (const ws of this.sockets) ws.send(msg);
  }

  /** Keep the REST answers in step with what was pushed live. */
  private remember(type: string, data: unknown): void {
    const upsert = <T extends { id: string }>(list: T[], x: T): T[] => [
      ...list.filter((y) => y.id !== x.id),
      x,
    ];
    if (type === "charter.updated") this.charters = upsert(this.charters, data as Charter);
    if (type === "approval.requested" || type === "approval.updated")
      this.approvals = upsert(this.approvals, data as Approval);
    if (type === "command.created" || type === "command.updated")
      this.commands = upsert(this.commands, data as Command);
    if (type === "annotation.created")
      this.annotations = upsert(this.annotations, data as Annotation);
    if (type === "mission.updated") this.mission = data as Mission;
  }

  async dropSocket(): Promise<void> {
    const open = this.sockets;
    this.sockets = [];
    for (const ws of open) await ws.close({ code: 1006 }).catch(() => undefined);
  }

  /** Stop answering new sockets (connection stays down). */
  refuseSockets = false;

  posts(path: string): SeenRequest[] {
    return this.requests.filter((r) => r.method === "POST" && r.path === path);
  }

  calls(re: RegExp): SeenRequest[] {
    return this.requests.filter((r) => re.test(`${r.method} ${r.path}`));
  }

  async open(path = "/"): Promise<void> {
    await this.page.goto(path);
  }

  /** Checks that run after every test. */
  async verify(): Promise<void> {
    expect(this.external, "requests to other hosts").toEqual([]);
    expect(this.unknown, "paths the fake ground does not know").toEqual([]);
    expect(this.cspErrors, "CSP violations").toEqual([]);
    for (const r of this.requests.filter((x) => x.method !== "GET")) {
      expect(r.headers["x-sd-client"], `${r.method} ${r.path} X-SD-Client`).toBe("panel");
      // WHEP signalling is SDP, on the video paths only (the server's guard allows just that).
      const sdp = /^\/api\/v1\/video\/[^/]+\/main\/whep(\/[^/]+)?$/.test(r.path)
        ? r.method === "POST"
          ? "application/sdp"
          : r.method === "PATCH"
            ? "application/trickle-ice-sdpfrag"
            : null
        : null;
      expect(r.headers["content-type"], `${r.method} ${r.path} Content-Type`).toContain(
        sdp ?? "application/json",
      );
      expect(r.headers["idempotency-key"], `${r.method} ${r.path} Idempotency-Key`).toBeUndefined();
    }
    if (!this.page.isClosed()) {
      const geo = await this.page.evaluate(
        () => (window as unknown as { __geoAccess?: number }).__geoAccess ?? 0,
      );
      expect(geo, "navigator.geolocation accesses").toBe(0);
    }
  }

  // --- internals ----------------------------------------------------------------------

  private async handleTiles(route: Route): Promise<void> {
    const req = route.request();
    const path = decodeURIComponent(new URL(req.url()).pathname);
    const answer = async (status: number, body?: Buffer, headers: Record<string, string> = {}) => {
      this.tiles.push({ path, status });
      await route.fulfill(body ? { status, body, headers } : { status, headers });
    };
    if (path === "/tiles/fixture.pmtiles") {
      const data = readFileSync(FIXTURE_PACK);
      const range = /^bytes=(\d+)-(\d*)$/.exec((await req.allHeaders()).range ?? "");
      if (!range) return answer(200, data, { "Content-Type": "application/vnd.pmtiles" });
      const start = Number(range[1]);
      const end = Math.min(range[2] ? Number(range[2]) : data.length - 1, data.length - 1);
      return answer(206, data.subarray(start, end + 1), {
        "Content-Type": "application/vnd.pmtiles",
        "Content-Range": `bytes ${start}-${end}/${data.length}`,
        "Accept-Ranges": "bytes",
      });
    }
    const sprite = /^\/tiles\/assets\/sprites\/v4\/(dark|light)(@2x)?\.(json|png)$/.exec(path);
    if (sprite?.[3] === "json")
      return answer(200, Buffer.from("{}"), { "Content-Type": "application/json" });
    if (sprite) return answer(200, PNG_1PX, { "Content-Type": "image/png" });
    const glyphs = /^\/tiles\/assets\/fonts\/([A-Za-z ]{1,64})\/(\d{1,5}-\d{1,5})\.pbf$/.exec(path);
    if (glyphs?.[1] && glyphs[2])
      return answer(200, emptyGlyphs(glyphs[1], glyphs[2]), {
        "Content-Type": "application/x-protobuf",
      });
    return answer(404);
  }

  private onSocket(ws: WebSocketRoute): void {
    if (this.refuseSockets) {
      void ws.close({ code: 1006 });
      return;
    }
    this.sockets.push(ws);
    ws.onMessage((m) => {
      try {
        this.wsSent.push(JSON.parse(String(m)) as Record<string, unknown>);
      } catch {
        // not JSON: ignored like the real server
      }
    });
    ws.onClose(() => {
      this.sockets = this.sockets.filter((s) => s !== ws);
    });
    ws.send(
      JSON.stringify({
        seq: null,
        topic: "system",
        type: "hello",
        data: { server_epoch: "e1", version: "0.0.0", profile: "sim" },
      }),
    );
  }

  private async handle(route: Route): Promise<void> {
    const req: Request = route.request();
    const url = new URL(req.url());
    const path = url.pathname + url.search;
    let body: unknown = null;
    try {
      body = req.postDataJSON();
    } catch {
      body = req.postData();
    }
    const seen: SeenRequest = { method: req.method(), path, headers: await req.allHeaders(), body };
    this.requests.push(seen);

    const key = `${seen.method} ${url.pathname}`;
    const o = this.overrides.find((x) => x.match === key && x.times > 0);
    let res: ReturnType<Handler> | null = null;
    if (o) {
      o.times -= 1;
      res = o.handler(seen);
    } else {
      res = this.defaults(seen, url.pathname);
    }
    if (this.contract && !this.contract.some((re) => re.test(key)))
      this.unknown.push(`${key} (not in OpenAPI)`);
    if (res === "hang") return; // never answered; the panel's own timeout aborts it
    if (res === null) {
      this.unknown.push(key);
      await route.fulfill({
        status: 404,
        json: { code: "NOT_FOUND", detail_sv: "okänd väg", status: 404, title: "" },
      });
      return;
    }
    if (res.status === 204) {
      await route.fulfill({ status: 204, headers: res.headers ?? {} });
      return;
    }
    await route.fulfill({ status: res.status, json: res.body ?? {}, headers: res.headers ?? {} });
    if (/^GET .*\/vehicles\/[^/]+\/state$/.test(key)) this.stateAnswered += 1;
  }

  private defaults(req: SeenRequest, p: string): ReturnType<Handler> | null {
    const k = `${req.method} ${p}`;
    const v = `${API}/vehicles/${VEHICLE}`;
    if (k === `GET ${API}/session`) {
      return this.session
        ? { status: 200, body: this.session }
        : { status: 401, body: { code: "UNAUTHENTICATED" } };
    }
    if (k === `POST ${API}/session`) {
      const b = req.body as { pin?: string; login_code?: string };
      if (b.pin === "123456" || b.login_code === "abc") {
        this.session = this.session ?? pilotSession;
        return { status: 200, body: this.session };
      }
      return { status: 401, body: { code: "LOGIN_INVALID", detail_sv: "Fel kod" } };
    }
    if (k === `POST ${API}/session/logout`) {
      this.session = null;
      return { status: 204 };
    }
    if (k === `GET ${API}/panel/config`) return { status: 200, body: this.config };
    if (k === `GET ${API}/system/status`) return { status: 200, body: systemStatus };
    if (k === `GET ${v}/state`) return { status: 200, body: this.vehicleState };
    if (k === `POST ${v}/missions/open`) return { status: 200, body: this.mission };
    if (k === `GET ${API}/missions/active`) return { status: 200, body: this.mission };
    const ms = `${API}/missions/${MISSION}`;
    if (k === `GET ${ms}`) {
      return {
        status: 200,
        body: {
          mission: this.mission,
          charter: this.charters.find((c) => c.id === this.mission.active_charter_id) ?? null,
          clearance: this.clearance,
          latch: null,
          active_command: null,
          pending_approvals: this.approvals.filter((a) => a.status === "pending"),
          sectors: this.sectors,
        },
      };
    }
    if (k === `PATCH ${ms}`) {
      this.mission = { ...this.mission, ...(req.body as object) };
      return { status: 200, body: this.mission };
    }
    if (k === `GET ${ms}/charters`) return { status: 200, body: this.charters };
    if (k === `GET ${ms}/approvals`) {
      const pending = req.path.includes("status=pending");
      return {
        status: 200,
        body: this.approvals.filter((a) => !pending || a.status === "pending"),
      };
    }
    if (k === `GET ${ms}/commands`) return { status: 200, body: this.commands };
    if (k === `GET ${ms}/sectors`) return { status: 200, body: this.sectors };
    if (k === `GET ${ms}/annotations`) return { status: 200, body: this.annotations };
    if (k === `GET ${ms}/chat`) return { status: 200, body: { items: [], next_before: null } };
    if (k === `GET ${ms}/track`) {
      return {
        status: 200,
        body: {
          type: "Feature",
          geometry: { type: "LineString", coordinates: this.track },
          properties: {},
        },
      };
    }
    if (k === `POST ${ms}/annotations`) {
      const b = req.body as {
        kind: "pin" | "area";
        label: string;
        geometry: Record<string, unknown>;
      };
      const a: Annotation = {
        id: `an-${this.annotations.length + 1}`,
        mission_id: MISSION,
        kind: b.kind,
        label: b.label,
        geometry: b.geometry,
        area_m2: b.kind === "area" ? 10_000 : null,
        created_at: new Date().toISOString(),
        created_by: "operator:demo",
      };
      this.annotations = [...this.annotations, a];
      return { status: 201, body: a };
    }
    if (k === `POST ${ms}/clearance`) {
      const c: Clearance = {
        id: "CL1",
        mission_id: MISSION,
        vehicle_id: VEHICLE,
        charter_id: "C1",
        charter_version: 1,
        status: "active",
        confirmed_at: new Date().toISOString(),
        confirmed_by: "operator:demo",
        valid_until: new Date(Date.now() + 120_000).toISOString(),
      };
      this.clearance = c;
      return { status: 200, body: c };
    }
    if (k === `PUT ${ms}/limits`) {
      this.mission = { ...this.mission, limits: req.body as Mission["limits"] };
      return { status: 200, body: this.mission };
    }
    if (k === `PUT ${ms}/autonomy`) {
      this.mission = {
        ...this.mission,
        autonomy_mode: (req.body as { mode: Mission["autonomy_mode"] }).mode,
      };
      return { status: 200, body: this.mission };
    }
    if (k === `POST ${ms}/end`) {
      this.mission = { ...this.mission, status: "CLOSED" };
      return { status: 200, body: this.mission };
    }
    if (k === `POST ${ms}/commands`) {
      const b = req.body as {
        client_command_id: string;
        skill: string;
        params: Record<string, unknown>;
      };
      return {
        status: 201,
        body: {
          command: {
            ...this.commands[0],
            id: b.client_command_id,
            skill: b.skill,
            params: b.params,
            status: "dispatched",
            issuer_kind: "operator",
          },
          gate: {
            decision: "ALLOW",
            reason_codes: [],
            rule_ids: [],
            message_sv: "",
            details: {},
            side: "ground",
          },
        },
      };
    }
    const decision = /^POST \/api\/v1\/approvals\/([^/]+)\/decision$/.exec(k);
    if (decision?.[1]) {
      const a = this.approvals.find((x) => x.id === decision[1]);
      if (!a) return { status: 404, body: { code: "NOT_FOUND", detail_sv: "okänt" } };
      const b = req.body as { decision: "approve" | "reject" };
      const updated: Approval = {
        ...a,
        status: b.decision === "approve" ? "approved" : "rejected",
        decided_by: "operator:demo",
        decided_at: new Date().toISOString(),
      };
      this.approvals = this.approvals.map((x) => (x.id === a.id ? updated : x));
      return { status: 200, body: updated };
    }
    const annDel = /^DELETE \/api\/v1\/annotations\/([^/]+)$/.exec(k);
    if (annDel?.[1]) {
      this.annotations = this.annotations.filter((x) => x.id !== annDel[1]);
      return { status: 204 };
    }
    const action =
      /^POST \/api\/v1\/vehicles\/sim-01\/actions\/(stop|return_home|resume|takeover|handback)$/.exec(
        k,
      );
    if (action) {
      this.actionCounter += 1;
      return { status: 202, body: dispatched(`a${this.actionCounter}`) };
    }
    const outcomeGet = /^GET \/api\/v1\/vehicles\/sim-01\/actions\/([^/]+)$/.exec(k);
    if (outcomeGet?.[1]) return { status: 200, body: outcome(outcomeGet[1]) };
    const all = /^POST \/api\/v1\/actions\/(stop|return_home)_all$/.exec(k);
    if (all) {
      this.actionCounter += 1;
      return {
        status: 202,
        body: { actions: [{ ...dispatched(`a${this.actionCounter}`), vehicle_id: VEHICLE }] },
      };
    }
    if (k === `POST ${API}/missions/${MISSION}/chat`) {
      const b = req.body as { text: string; client_msg_id: string };
      return {
        status: 201,
        body: {
          id: `c-${b.client_msg_id}`,
          mission_id: MISSION,
          seq: 1,
          role: "operator",
          kind: "text",
          text: b.text,
          author_ref: "operator:demo",
          author_role: "pilot",
          created_at: new Date().toISOString(),
          delivery: "queued",
          handled_by: null,
          in_reply_to: null,
          parsed_intent: null,
          refs: {},
        },
      };
    }
    return null;
  }
}

export const test = base.extend<{ ground: FakeGround }>({
  ground: async ({ page, baseURL }, use) => {
    const g = new FakeGround(page, new URL(baseURL ?? "http://127.0.0.1:4173").origin);
    await g.install();
    await use(g);
    await g.verify();
  },
});

export { expect };
