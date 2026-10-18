// Every panel state the operator meets, photographed at each project's size in both themes.
// The fake ground supplies the data; the video is a made-up aerial frame (aerial-frame.jpg)
// sent over a real WebRTC connection, only in the states where the drone is in the air.
import { readFileSync } from "node:fs";
import { fileURLToPath } from "node:url";
import type { Page, Route } from "@playwright/test";
import type { MapConfig } from "@searchdrone/schemas";
import {
  approval,
  charter,
  chatMessage,
  HOME,
  panelConfig,
  snapshot,
  square,
  telemetry,
  vehicleLive,
} from "../fixtures";
import { expect, type FakeGround, test } from "../helpers/fakeGround";
import { layoutOf, showView } from "../helpers/safety";

const OUT = fileURLToPath(new URL("../../screens/", import.meta.url));
const FRAME = readFileSync(new URL("../fixtures/aerial-frame.jpg", import.meta.url));
const WHEP = "/api/v1/video/sim-01/main/whep";
const MAP: MapConfig = {
  pack_url: "/tiles/fixture.pmtiles",
  center: HOME,
  zoom: 16,
  maxzoom: 15,
  attribution: "© OpenStreetMap contributors, Protomaps",
  assets_url: "/tiles/assets",
};

// The search area north of home, in four sectors of 150 × 150 m.
const AREA = square(150, 0, 230);
const sector = (id: string, east: number, north: number, status: string, progress: number) => ({
  type: "Feature",
  geometry: square(75, east, north),
  properties: {
    sector_id: id,
    name: id,
    status,
    progress,
    area_m2: 22_500,
    covered:
      progress >= 1
        ? square(75, east, north)
        : progress > 0
          ? {
              type: "Polygon",
              coordinates: [
                square(75, east, north).coordinates[0]?.map(([lon, lat], i) =>
                  i === 2 || i === 3
                    ? [lon, (lat ?? 0) - (150 * (1 - progress)) / 111_320]
                    : [lon, lat],
                ),
              ],
            }
          : null,
  },
});
const SECTORS = {
  type: "FeatureCollection",
  features: [
    sector("A", -75, 155, "done", 1),
    sector("B", 75, 155, "active", 0.45),
    sector("C", -75, 305, "pending", 0),
    sector("D", 75, 305, "pending", 0),
  ],
};
const charterBodyArea = {
  area: AREA,
  notes: "Fyra sektorer, sydväst först.",
  sectors: SECTORS.features.map((f, i) => ({
    sector_id: f.properties.sector_id,
    name: f.properties.name,
    order_idx: i,
    area: f.geometry,
  })),
};
// Over sector B, at the edge of what is swept.
const AIRBORNE = telemetry({
  lat: HOME.lat + 150 / 111_320,
  lon: HOME.lon + 75 / (111_320 * Math.cos((HOME.lat * Math.PI) / 180)),
  alt_rel_home_m: 40,
  heading_deg: 72,
  groundspeed_ms: 6.2,
  battery_pct: 58,
  dist_to_pilot_m: 210,
  attitude: { roll_deg: -7, pitch_deg: -3, yaw_deg: 72 },
});

/** A WHEP mock that answers with a real stream of the aerial frame, drifting slowly. */
async function aerialVideo(page: Page): Promise<void> {
  await page.route("**/__screens/aerial-frame.jpg", (r) =>
    r.fulfill({ status: 200, body: FRAME, contentType: "image/jpeg" }),
  );
  await page.route(`**${WHEP}**`, async (route: Route) => {
    const req = route.request();
    if (req.method() === "DELETE") return route.fulfill({ status: 204 });
    if (req.method() !== "POST") return route.fulfill({ status: 404 });
    const sdp = await page.evaluate(async (offer) => {
      const img = new Image();
      img.src = "/__screens/aerial-frame.jpg";
      await img.decode();
      const canvas = document.createElement("canvas");
      canvas.width = 1280;
      canvas.height = 720;
      const ctx = canvas.getContext("2d");
      let n = 0;
      setInterval(() => {
        n += 1;
        ctx?.drawImage(img, -(n % 40), -(n % 24), 1320, 744);
      }, 66);
      const pc = new RTCPeerConnection({ iceServers: [] });
      (window as unknown as { __sender: RTCPeerConnection }).__sender = pc;
      for (const t of canvas.captureStream(15).getTracks()) pc.addTrack(t);
      await pc.setRemoteDescription({ type: "offer", sdp: offer });
      await pc.setLocalDescription(await pc.createAnswer());
      await new Promise((r) => setTimeout(r, 300));
      return pc.localDescription?.sdp ?? "";
    }, req.postData() ?? "");
    await route.fulfill({
      status: 201,
      body: sdp,
      headers: { "Content-Type": "application/sdp", Location: `${WHEP}/s1` },
    });
  });
}

/** The panel's own clock for the stores, so a value can be made old on purpose. */
async function ageTelemetry(page: Page, ms: number): Promise<void> {
  await page.evaluate((age) => {
    const s = (window as unknown as { __sd: { liveStore: { setState(p: unknown): void } } }).__sd;
    s.liveStore.setState({ lastTelemetryAt: Date.now() - age });
  }, ms);
}

async function mapReady(page: Page): Promise<void> {
  await page.waitForFunction(
    () =>
      (window as unknown as { __sd?: { map?: { loaded(): boolean } } }).__sd?.map?.loaded() ===
      true,
  );
  await page.waitForTimeout(600);
}

async function chatHistory(ground: FakeGround): Promise<void> {
  ground.pushLive("chat", "chat.brain_status", { state: "ready", typing: false });
  const lines: [string, "operator" | "brain", string][] = [
    ["c1", "operator", "Sök av rutan norr om hem, 300 × 300 m. Börja i sydväst."],
    [
      "c2",
      "brain",
      "Jag föreslår fyra sektorer på 150 × 150 m, 40 m höjd och 8 m/s. Det täcker 9 ha på ungefär 14 minuter.",
    ],
  ];
  for (const [i, [id, role, text]] of lines.entries())
    ground.pushLive(
      "chat",
      "chat.message",
      chatMessage({
        id,
        seq: i + 1,
        role,
        text,
        author_ref: role === "brain" ? "brain:commander" : "operator:demo",
        created_at: new Date(Date.now() - (3 - i) * 60_000).toISOString(),
      }),
    );
}

/** On the ground with a charter waiting for the hold-to-approve. */
async function proposal(page: Page, ground: FakeGround): Promise<void> {
  ground.mission = { ...ground.mission, status: "CHARTER_PROPOSED" };
  ground.charters = [charter({}, charterBodyArea)];
  ground.approvals = [approval({ summary_sv: "Sök av rutan 300 × 300 m norr om hem." })];
  // Nothing is flown or swept before the charter is approved, and no camera is up yet.
  ground.sectors = { type: "FeatureCollection", features: [] };
  ground.track = [];
  ground.setConfig({ video: null });
  await ground.open();
  await ground.liveConnected();
  await ground.resynced();
  await chatHistory(ground);
  ground.pushLive("vehicle", "vehicle.state", {
    ...vehicleLive,
    snapshot: snapshot({ executor_state: "PREFLIGHT" }),
  });
  keepReporting(
    page,
    ground,
    telemetry({
      lat: HOME.lat,
      lon: HOME.lon,
      armed: false,
      landed_state: "on_ground",
      alt_rel_home_m: 0,
      groundspeed_ms: 0,
      attitude: { roll_deg: 0, pitch_deg: 0, yaw_deg: 10 },
    }),
  );
  await mapReady(page);
  await showWholeArea(page);
  await showView(page, "chat");
  await expect(page.getByTestId("approval-AP1")).toBeVisible();
}

const heartbeats = new WeakMap<Page, ReturnType<typeof setInterval>>();

/** The drone keeps reporting, as a real one does, until a state cuts the link. */
function keepReporting(page: Page, ground: FakeGround, tel: ReturnType<typeof telemetry>): void {
  ground.pushLive("telemetry", "telemetry", tel);
  const beat = setInterval(() => ground.pushLive("telemetry", "telemetry", tel), 500);
  page.once("close", () => clearInterval(beat));
  heartbeats.set(page, beat);
}

/** In the air, searching sector B, with the video live. */
async function searching(page: Page, ground: FakeGround): Promise<void> {
  ground.mission = {
    ...ground.mission,
    status: "ACTIVE",
    active_charter_id: "C1",
    autonomy_mode: "WITHIN_CHARTER",
    activated_at: new Date(Date.now() - 6 * 60_000).toISOString(),
  };
  ground.charters = [charter({ status: "approved", install_status: "installed" }, charterBodyArea)];
  await aerialVideo(page);
  await ground.open();
  await ground.liveConnected();
  await ground.resynced();
  await chatHistory(ground);
  ground.pushLive(
    "chat",
    "chat.message",
    chatMessage({ id: "c3", seq: 3, text: "Sektor A klar. Fortsätter med B." }),
  );
  ground.pushLive("vehicle", "vehicle.state", {
    ...vehicleLive,
    airborne: true,
    in_flight: true,
    snapshot: snapshot({ executor_state: "SURVEY" }),
  });
  keepReporting(page, ground, AIRBORNE);
  ground.pushLive("alerts", "alert", {
    id: null,
    severity: "info",
    code: "SECTOR_DONE",
    message_sv: "Sektor A klar",
    data: {},
  });
  await expect(page.getByTestId("video-badge")).toContainText("WebRTC", { timeout: 15_000 });
  await mapReady(page);
  await showWholeArea(page);
}

/** What the operator presses to see the whole search area. */
async function showWholeArea(page: Page): Promise<void> {
  await page.getByRole("button", { name: "Hela området" }).click();
  await page.waitForTimeout(500);
}

interface State {
  name: string;
  run(page: Page, ground: FakeGround): Promise<void>;
}

const STATES: State[] = [
  { name: "01-proposal", run: proposal },
  {
    name: "02-proposal-holding",
    async run(page, ground) {
      await proposal(page, ground);
      const btn = page.getByTestId("approval-AP1").getByTestId("approve");
      await btn.scrollIntoViewIfNeeded();
      const box = await btn.boundingBox();
      if (!box) throw new Error("no approve button");
      await page.mouse.move(box.x + box.width / 2, box.y + box.height / 2);
      await page.mouse.down();
      await page.waitForTimeout(450);
    },
  },
  { name: "03-search-map", run: searching },
  {
    name: "04-search-video",
    async run(page, ground) {
      await searching(page, ground);
      await showView(page, "video");
      ground.pushLive("telemetry", "telemetry", AIRBORNE);
      await expect(page.getByTestId("hud")).toBeVisible();
    },
  },
  {
    name: "05-events",
    async run(page, ground) {
      await searching(page, ground);
      await showView(page, "events");
    },
  },
  {
    name: "06-link-lost",
    async run(page, ground) {
      await searching(page, ground);
      await showView(page, "video");
      clearInterval(heartbeats.get(page));
      await page.waitForTimeout(1_500);
      await page.evaluate(() => {
        const pc = (window as unknown as { __sender: RTCPeerConnection }).__sender;
        return Promise.all(pc.getSenders().map((s) => s.replaceTrack(null)));
      });
      ground.pushLive("vehicle", "vehicle.state", {
        ...vehicleLive,
        airborne: true,
        link: { mavlink: "lost", sdlink: "lost", onboard_heartbeat_age_ms: 14_000, rtt_ms: null },
        snapshot: snapshot({ executor_state: "SURVEY" }),
      });
      await expect(page.getByTestId("video-stale")).toBeVisible({ timeout: 10_000 });
      await ageTelemetry(page, 14_000);
      await page.waitForTimeout(600);
    },
  },
  {
    name: "07-ai-down",
    async run(page, ground) {
      await searching(page, ground);
      ground.pushLive("alerts", "alert", {
        id: null,
        severity: "critical",
        code: "BRAIN_DISCONNECTED",
        message_sv: "AI:n svarar inte",
        data: {},
      });
      await showView(page, "chat");
      await expect(page.getByTestId("ai-down-banner")).toBeVisible();
    },
  },
  {
    name: "08-stop-sent",
    async run(page, ground) {
      await searching(page, ground);
      await page.getByTestId("safety-stop").click();
      await expect(page.getByTestId("safety-stop")).toContainText(/skickat|kvitterat/i, {
        timeout: 5_000,
      });
    },
  },
  {
    name: "09-no-ground",
    async run(page, ground) {
      await searching(page, ground);
      clearInterval(heartbeats.get(page));
      ground.refuseSockets = true;
      await ground.dropSocket();
      await expect(page.getByTestId("connection-banner")).toBeVisible({ timeout: 5_000 });
    },
  },
  {
    name: "10-review",
    async run(page, ground) {
      await searching(page, ground);
      await showView(page, "candidates");
    },
  },
  {
    name: "11-status",
    async run(page, ground) {
      await searching(page, ground);
      await showView(page, "status");
    },
  },
];

for (const theme of ["dark", "light"] as const) {
  for (const state of STATES) {
    test(`${theme} ${state.name}`, async ({ page, ground }, info) => {
      await page.addInitScript((t) => localStorage.setItem("sd.panel.theme", t), theme);
      ground.setConfig({
        map: MAP,
        video: { whep_url: WHEP, hls_url: null },
        features: { ...panelConfig.features, order_buttons: true, manual_control: true },
      });
      ground.sectors = SECTORS;
      ground.track = [
        [HOME.lon, HOME.lat],
        ...(SECTORS.features[0]?.geometry.coordinates[0]?.slice(0, 4) ?? []),
      ];
      await state.run(page, ground);
      if ((await layoutOf(page)) === "phone" && state.name === "03-search-map")
        await page.getByTestId("sheet-handle").click();
      await page.screenshot({ path: `${OUT}${info.project.name}/${theme}-${state.name}.png` });
    });
  }
}
