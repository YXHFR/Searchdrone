// Drawing a path the drone follows: draw it on the map, ask the AI in the chat, approve the
// AI's card, then the planned path (dashed), the flown part (solid) and the progress in %.
// Screenshots of each step go to test-results/path-flow/ (uploaded by CI).
import type { Page } from "@playwright/test";
import type { Command, MapConfig } from "@searchdrone/schemas";
import { approval, charter, command, HOME, square, telemetry } from "../fixtures";
import { expect, test } from "../helpers/fakeGround";
import { expectSafetyVisible, isMobile } from "../helpers/safety";

const ANN = "/api/v1/missions/M1/annotations";
const MAP: MapConfig = {
  pack_url: "/tiles/fixture.pmtiles",
  center: HOME,
  zoom: 16,
  maxzoom: 15,
  attribution: "© OpenStreetMap contributors, Protomaps",
  assets_url: "/tiles/assets",
};
const CHAT = "/api/v1/missions/M1/chat";
// A trail north of home (east, north in metres), inside the charter below.
const TRAIL: [number, number][] = [
  [0, 20],
  [30, 50],
  [50, 85],
  [20, 110],
  [-30, 95],
];

interface MapHandle {
  loaded(): boolean;
  once(type: "idle"): Promise<unknown>;
  fitBounds(
    b: [[number, number], [number, number]],
    o: { padding: number; duration: number },
  ): void;
  fire(type: string, e: unknown): void;
  getSource(id: string):
    | {
        serialize(): { data: { features: { geometry: { type: string; coordinates: unknown } }[] } };
      }
    | undefined;
}

const lonlat = (east: number, north: number): [number, number] => [
  HOME.lon + east / (111_320 * Math.cos((HOME.lat * Math.PI) / 180)),
  HOME.lat + north / 111_320,
];

async function mapReady(page: Page): Promise<void> {
  await page.waitForFunction(
    () => (window as unknown as { __sd?: { map?: MapHandle } }).__sd?.map?.loaded() === true,
  );
  await page.evaluate(
    async (b) => {
      const map = (window as unknown as { __sd: { map: MapHandle } }).__sd.map;
      const idle = map.once("idle");
      map.fitBounds(b, { padding: 30, duration: 0 });
      await idle;
    },
    [lonlat(-45, 5), lonlat(65, 125)] as [[number, number], [number, number]],
  );
}

async function mapClick(page: Page, east: number, north: number): Promise<void> {
  const [lng, lat] = lonlat(east, north);
  await page.evaluate(
    ({ lng, lat }) =>
      (window as unknown as { __sd: { map: MapHandle } }).__sd.map.fire("click", {
        lngLat: { lng, lat },
      }),
    { lng, lat },
  );
}

async function feats(page: Page, id: string) {
  return page.evaluate(
    (id) =>
      (window as unknown as { __sd: { map: MapHandle } }).__sd.map.getSource(id)?.serialize().data
        .features ?? [],
    id,
  );
}

async function shot(page: Page, name: string): Promise<void> {
  await page.waitForTimeout(300); // let the map draw the last change
  await page.screenshot({ path: `test-results/path-flow/${test.info().project.name}-${name}.png` });
}

function setUp(ground: import("../helpers/fakeGround").FakeGround): void {
  ground.setConfig({ map: MAP });
  ground.mission = { ...ground.mission, status: "ACTIVE", active_charter_id: "C1" };
  ground.charters = [
    charter(
      { status: "approved", install_status: "installed" },
      { area: square(120, 0, 70), sectors: [], allowed_skills: ["goto", "follow_path", "hold"] },
    ),
  ];
}

const POINTS = TRAIL.map(([e, n]) => {
  const [lon, lat] = lonlat(e, n);
  return { lat, lon };
});

function pathOrder(status: Command["status"]) {
  return command({
    id: "K7",
    skill: "follow_path",
    status,
    approval_id: "AP7",
    params: { points: POINTS, alt_m: 30, label: "pa-1" },
    rationale_sv: "Följer stigen som piloten ritade.",
  });
}

test("draw a path, ask in the chat, approve: planned dashed, flown solid, progress in %", async ({
  page,
  ground,
}) => {
  setUp(ground);
  await ground.open();
  const webgl2 = await page.evaluate(
    () => document.createElement("canvas").getContext("webgl2") !== null,
  );
  test.skip(!webgl2, "WebGL2 saknas i denna webbläsare");
  const phone = isMobile(page);
  if (phone) await page.getByTestId("tab-map").click();
  await ground.liveConnected();
  await ground.resynced();
  ground.pushLive("telemetry", "telemetry", telemetry({ ...lonlatObj(0, 0), alt_rel_home_m: 30 }));
  await mapReady(page);

  // 1. Draw the path: five clicks along the trail, then Klar.
  await page.getByTestId("draw-path").click();
  for (const [e, n] of TRAIL) await mapClick(page, e, n);
  await expect(page.getByTestId("draw-bar")).toContainText("5 punkter");
  await shot(page, "1-drawing");
  await page.getByTestId("draw-done").click();
  await expect.poll(() => ground.posts(ANN).length).toBe(1);
  const body = ground.posts(ANN)[0]?.body as {
    kind: string;
    label: string;
    geometry: { type: string; coordinates: number[][] };
  };
  expect(body).toMatchObject({ kind: "path", label: "pa-1", geometry: { type: "LineString" } });
  expect(body.geometry.coordinates).toHaveLength(5);
  await expect(page.getByTestId("toasts")).toContainText("Väg pa-1 sparad (5 punkter");
  if (!phone) {
    await expect.poll(async () => (await feats(page, "paths")).length).toBe(2); // line + start
    await shot(page, "2-saved");
  }

  // 2. Ask the AI in the chat. The fake AI answers with a follow_path card.
  if (phone && !(await page.getByTestId("view-chat").isVisible()))
    await page.getByTestId("tab-chat").click();
  const box = page.getByRole("textbox", { name: "Meddelande till AI-befälhavaren" });
  await box.fill("Flyg vägen pa-1.");
  await box.press("Enter");
  await expect.poll(() => ground.posts(CHAT).length).toBe(1);
  ground.pushLive("commands", "command.created", pathOrder("pending_approval"));
  ground.pushLive(
    "approvals",
    "approval.requested",
    approval({
      id: "AP7",
      kind: "command",
      subject_id: "K7",
      summary_sv: "Flyg längs vägen pa-1 (5 punkter, 0,2 km) på 30 m",
    }),
  );
  const card = page.getByTestId("approval-AP7");
  await expect(card).toBeVisible();
  await expect(card.getByTestId("skill-path")).toContainText("Flyg längs vägen pa-1: 5 punkter");
  await expect(card).not.toContainText('"lat"'); // the points are not dumped as JSON
  await shot(page, "3-card");
  if (!phone) {
    await expect(page.getByTestId("path-progress")).toContainText("väntar på ditt godkännande");
    expect((await feats(page, "path-plan")).length).toBe(6); // the line and its 5 points
    expect(await feats(page, "path-flown")).toEqual([]);
  }

  // 3. Approve. The drone flies the path.
  await card.getByTestId("approve").click();
  await expect.poll(() => ground.posts("/api/v1/approvals/AP7/decision").length).toBe(1);
  expect(ground.posts("/api/v1/approvals/AP7/decision")[0]?.body).toMatchObject({
    decision: "approve",
  });
  ground.pushLive("commands", "command.updated", pathOrder("running"));
  ground.pushLive(
    "telemetry",
    "telemetry",
    telemetry({ ...lonlatObj(45, 77), alt_rel_home_m: 30 }),
  );
  if (phone) await page.getByTestId("tab-map").click();
  await expect(page.getByTestId("path-progress")).toContainText(/Väg pa-1: \d+ % flugen/);
  const pct = Number(
    /(\d+) %/.exec((await page.getByTestId("path-progress").textContent()) ?? "")?.[1],
  );
  expect(pct).toBeGreaterThan(30);
  expect(pct).toBeLessThan(60);
  await expect.poll(async () => (await feats(page, "path-flown")).length).toBe(1);
  const flown = (await feats(page, "path-flown"))[0]?.geometry.coordinates as number[][];
  expect(flown.length).toBe(3); // the first two points, then the drone's place on the line
  await expectSafetyVisible(page, "flying a path");
  await shot(page, "4-flying");

  // 4. At the end: done, the whole line flown, 100 %. The drone hovers there.
  ground.pushLive("commands", "command.updated", pathOrder("done"));
  ground.pushLive(
    "telemetry",
    "telemetry",
    telemetry({ ...lonlatObj(-30, 95), alt_rel_home_m: 30 }),
  );
  await expect(page.getByTestId("path-progress")).toContainText("Väg pa-1: 100 % flugen");
  await expect
    .poll(
      async () =>
        ((await feats(page, "path-flown"))[0]?.geometry.coordinates as unknown[] | undefined)
          ?.length,
    )
    .toBe(5);
  await shot(page, "5-done");
});

test("a path outside the search area is refused in the panel, in Swedish", async ({
  page,
  ground,
}) => {
  setUp(ground);
  await ground.open();
  const webgl2 = await page.evaluate(
    () => document.createElement("canvas").getContext("webgl2") !== null,
  );
  test.skip(!webgl2, "WebGL2 saknas i denna webbläsare");
  if (isMobile(page)) await page.getByTestId("tab-map").click();
  await ground.resynced();
  await mapReady(page);
  await page.getByTestId("draw-path").click();
  await mapClick(page, 0, 20);
  await mapClick(page, 200, 60); // 80 m past the charter's east edge
  await page.getByTestId("draw-done").click();
  await expect(page.getByTestId("draw-bar")).toContainText(
    "Vägen går utanför sökområdet. Rita den inom uppdragsramen.",
  );
  expect(ground.posts(ANN)).toEqual([]);
  await shot(page, "6-outside");
});

test("a path the battery cannot fly is refused by the gate with the explanation", async ({
  page,
  ground,
}) => {
  setUp(ground);
  await ground.open();
  await ground.liveConnected();
  await ground.resynced();
  const reason =
    "Batteriet räcker inte för hela vägen och flygningen hem: 4,9 km kräver ungefär 106 % " +
    "inklusive reserven på 15 %, men batteriet har 80 %. Rita en kortare väg eller byt batteri.";
  ground.pushLive("commands", "command.updated", {
    ...pathOrder("rejected_by_gate"),
    status_reason: "ROUTE_BATTERY",
    reason_codes: ["ROUTE_BATTERY"],
    message_sv: reason,
  });
  await expect(page.getByTestId("toasts")).toContainText(`Vägen nekades: ${reason}`);
  await shot(page, "7-battery");
});

function lonlatObj(east: number, north: number): { lat: number; lon: number } {
  const [lon, lat] = lonlat(east, north);
  return { lat, lon };
}
