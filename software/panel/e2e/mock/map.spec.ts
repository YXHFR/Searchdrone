// The offline map (B1-11) with every other host blocked by the fake ground station: the
// synthetic test pack is drawn at zoom 12, 15 and 18 (16-18 overzoomed from 15), with no
// tile or glyph errors and the OpenStreetMap attribution visible.
import type { MapConfig } from "@searchdrone/schemas";
import { approval, charter, command, HOME, square, telemetry } from "../fixtures";
import { expect, test } from "../helpers/fakeGround";
import { expectSafetyVisible, isMobile } from "../helpers/safety";

const MAP: MapConfig = {
  pack_url: "/tiles/fixture.pmtiles",
  center: HOME,
  zoom: 12,
  maxzoom: 15,
  attribution: "© OpenStreetMap contributors, Protomaps",
  assets_url: "/tiles/assets",
};

interface TestMap {
  loaded(): boolean;
  jumpTo(o: { center: [number, number]; zoom: number }): void;
  once(type: "idle"): Promise<unknown>;
  getZoom(): number;
  queryRenderedFeatures(): { sourceLayer?: string }[];
  getStyle(): { sources: Record<string, { maxzoom?: number }> };
}

test("the map draws the offline pack at zoom 12, 15 and 18", async ({ page, ground }) => {
  const errors: string[] = [];
  page.on("console", (m) => {
    // WebKit reports the viewport meta's interactive-widget key (for Chrome's keyboard
    // handling) as unknown; that is not a map error.
    if (m.type() === "error" && !/interactive-widget/.test(m.text())) errors.push(m.text());
  });
  page.on("pageerror", (e) => errors.push(e.message));
  ground.setConfig({ map: MAP });
  await ground.open();
  const webgl2 = await page.evaluate(
    () => document.createElement("canvas").getContext("webgl2") !== null,
  );
  test.skip(!webgl2, "WebGL2 saknas i denna webbläsare");
  if (isMobile(page)) await page.getByTestId("tab-map").click();
  await page.waitForFunction(
    () => (window as unknown as { __sd?: { map?: TestMap } }).__sd?.map?.loaded() === true,
  );

  for (const zoom of [12, 15, 18]) {
    const seen = await page.evaluate(
      async ({ zoom, center }) => {
        const map = (window as unknown as { __sd: { map: TestMap } }).__sd.map;
        const idle = map.once("idle");
        map.jumpTo({ center: [center.lon, center.lat], zoom });
        await idle;
        const features = map.queryRenderedFeatures();
        return {
          zoom: map.getZoom(),
          count: features.length,
          layers: [...new Set(features.map((f) => f.sourceLayer))].sort(),
        };
      },
      { zoom, center: HOME },
    );
    expect(seen.zoom).toBe(zoom);
    expect(seen.count, `rendered features at zoom ${zoom}`).toBeGreaterThan(0);
    expect(seen.layers).toContain("earth");
  }

  const maxzoom = await page.evaluate(
    () =>
      (window as unknown as { __sd: { map: TestMap } }).__sd.map.getStyle().sources.protomaps
        ?.maxzoom,
  );
  expect(maxzoom).toBe(15);
  await expect(page.locator(".maplibregl-ctrl-attrib")).toContainText("OpenStreetMap");

  // Tiles came from the pack with Range requests; nothing on /tiles/ failed (no glyph errors).
  expect(ground.tiles.some((t) => t.path === "/tiles/fixture.pmtiles" && t.status === 206)).toBe(
    true,
  );
  expect(ground.tiles.filter((t) => t.status >= 400)).toEqual([]);
  expect(errors).toEqual([]);
});

test("without a map pack the map view says so, requests no tiles and still draws the drone", async ({
  page,
  ground,
}) => {
  await ground.open();
  if (isMobile(page)) await page.getByTestId("tab-map").click();
  await expect(page.getByText("Ingen offlinekarta laddad")).toBeVisible();
  const webgl2 = await page.evaluate(
    () => document.createElement("canvas").getContext("webgl2") !== null,
  );
  if (webgl2) {
    await ground.liveConnected();
    await page.waitForFunction(
      () => (window as unknown as { __sd?: { map?: SourceMap } }).__sd?.map?.loaded() === true,
    );
    ground.pushLive("telemetry", "telemetry", telemetry());
    await expect.poll(async () => (await feats(page, "drone")).length).toBe(1);
  }
  expect(ground.tiles).toEqual([]);
});

test("STOPP and FLYG HEM stay visible and work over the real map", async ({ page, ground }) => {
  ground.setConfig({ map: MAP });
  await ground.open();
  const webgl2 = await page.evaluate(
    () => document.createElement("canvas").getContext("webgl2") !== null,
  );
  test.skip(!webgl2, "WebGL2 saknas i denna webbläsare");
  if (isMobile(page)) await page.getByTestId("tab-map").click();
  await page.waitForFunction(
    () => (window as unknown as { __sd?: { map?: TestMap } }).__sd?.map?.loaded() === true,
  );
  await expectSafetyVisible(page, "map loaded");
  if (!isMobile(page)) {
    await page.getByTestId("maximize-map").click();
    await page.waitForFunction(
      () => (window as unknown as { __sd?: { map?: TestMap } }).__sd?.map?.loaded() === true,
    );
    await expectSafetyVisible(page, "map full screen");
  }
  await page.getByTestId("safety-stop").click();
  await expect.poll(() => ground.posts("/api/v1/vehicles/sim-01/actions/stop").length).toBe(1);
});

interface SourceMap {
  loaded(): boolean;
  getSource(id: string):
    | {
        serialize(): { data: { features: { geometry: { type: string; coordinates: unknown } }[] } };
      }
    | undefined;
  project(ll: [number, number]): { x: number; y: number };
}

interface F {
  geometry: { type: string; coordinates: unknown };
}

/** The features currently set on one of the panel's own sources. */
async function feats(page: import("@playwright/test").Page, id: string): Promise<F[]> {
  return (await sources(page, [id]))[id] ?? [];
}

/** Zoom close (a drawn area must stay under the server's area cap) and wait until idle. */
async function zoomIn(page: import("@playwright/test").Page) {
  await page.evaluate(async (c) => {
    const map = (window as unknown as { __sd: { map: TestMap } }).__sd.map;
    const idle = map.once("idle");
    map.jumpTo({ center: [c.lon, c.lat], zoom: 17 });
    await idle;
  }, HOME);
}

/** A click on the map: a tap on touch screens (MapLibre listens to touch there). */
async function press(page: import("@playwright/test").Page, x: number, y: number) {
  if (isMobile(page)) await page.touchscreen.tap(x, y);
  else await page.mouse.click(x, y);
  await page.waitForTimeout(350); // two quick presses would be a double tap
}

/** A map click at `east`, `north` metres from home, as MapLibre reports it to the panel. */
async function mapClick(page: import("@playwright/test").Page, east: number, north: number) {
  await page.evaluate(
    ({ c, east, north }) => {
      const map = (window as unknown as { __sd: { map: { fire(t: string, e: unknown): void } } })
        .__sd.map;
      const lat = c.lat + north / 111_320;
      const lng = c.lon + east / (111_320 * Math.cos((c.lat * Math.PI) / 180));
      map.fire("click", { lngLat: { lng, lat } });
    },
    { c: HOME, east, north },
  );
}

/** The GeoJSON currently set on each of the panel's own sources. */
async function sources(page: import("@playwright/test").Page, ids: string[]) {
  return page.evaluate((list) => {
    const map = (window as unknown as { __sd: { map: SourceMap } }).__sd.map;
    return Object.fromEntries(
      list.map((id) => [id, map.getSource(id)?.serialize().data.features ?? []]),
    );
  }, ids);
}

test("every layer is fed: drone, charter, pilot range, home, sectors, target, track", async ({
  page,
  ground,
}) => {
  ground.setConfig({ map: MAP });
  ground.mission = { ...ground.mission, status: "ACTIVE", active_charter_id: "C1" };
  ground.charters = [charter({ status: "approved", install_status: "installed" })];
  ground.sectors = {
    type: "FeatureCollection",
    features: [
      {
        type: "Feature",
        geometry: square(50, 0, 180),
        properties: { sector_id: "S1", name: "S1", status: "active", progress: 0.4 },
      },
      {
        type: "Feature",
        geometry: square(50, 100, 180),
        properties: { sector_id: "S2", name: "S2", status: "pending", progress: 0 },
      },
    ],
  };
  ground.track = [
    [HOME.lon, HOME.lat],
    [HOME.lon, HOME.lat + 0.0004],
  ];
  await ground.open();
  const webgl2 = await page.evaluate(
    () => document.createElement("canvas").getContext("webgl2") !== null,
  );
  test.skip(!webgl2, "WebGL2 saknas i denna webbläsare");
  if (isMobile(page)) await page.getByTestId("tab-map").click();
  await ground.liveConnected();
  await ground.resynced();
  await page.waitForFunction(
    () => (window as unknown as { __sd?: { map?: SourceMap } }).__sd?.map?.loaded() === true,
  );
  const tel = telemetry();
  ground.pushLive("telemetry", "telemetry", tel);
  ground.pushLive(
    "commands",
    "command.updated",
    command({
      skill: "goto",
      status: "running",
      params: { lat: HOME.lat + 0.001, lon: HOME.lon, alt_m: 40 },
    }),
  );
  await expect
    .poll(async () => {
      const s = await sources(page, [
        "drone",
        "charter-area",
        "pilot-range",
        "home",
        "sectors",
        "target",
        "track",
      ]);
      return Object.fromEntries(Object.entries(s).map(([k, v]) => [k, v.length]));
    })
    .toEqual({
      drone: 1,
      "charter-area": 1,
      "pilot-range": 2,
      home: 1,
      sectors: 2,
      target: 1,
      track: 1,
    });
  const drone = await feats(page, "drone");
  const range = await feats(page, "pilot-range");
  const track = await feats(page, "track");
  const [lon, lat] = (drone[0]?.geometry.coordinates ?? [0, 0]) as [number, number];
  expect(Math.abs(lon - tel.lon)).toBeLessThan(1e-6);
  expect(Math.abs(lat - tel.lat)).toBeLessThan(1e-6);
  const poly = range.find((f) => f.geometry.type === "Polygon");
  const ring = ((poly?.geometry.coordinates ?? []) as number[][][])[0] ?? [];
  const r = Math.hypot(
    ((ring[0]?.[0] ?? 0) - HOME.lon) * 111_320 * Math.cos((HOME.lat * Math.PI) / 180),
    ((ring[0]?.[1] ?? 0) - HOME.lat) * 111_320,
  );
  expect(Math.abs(r - 500) / 500).toBeLessThan(0.01);
  expect(((track[0]?.geometry.coordinates ?? []) as unknown[]).length).toBeGreaterThanOrEqual(2);
  expect(ground.calls(/^GET \/api\/v1\/missions\/M1\/track/).length).toBeGreaterThan(0);

  // Without telemetry for more than 3 s the drone is drawn as stale.
  await expect
    .poll(
      async () =>
        page.evaluate(() => {
          const map = (
            window as unknown as {
              __sd: {
                map: {
                  getSource(id: string): {
                    serialize(): { data: { features: { properties: { stale: boolean } }[] } };
                  };
                };
              };
            }
          ).__sd.map;
          return map.getSource("drone").serialize().data.features[0]?.properties.stale;
        }),
      { timeout: 6_000 },
    )
    .toBe(true);
  void approval;
});

test("draw a rectangle and set a pin: one POST each, shown on the map", async ({
  page,
  ground,
}) => {
  ground.setConfig({ map: MAP });
  await ground.open();
  const webgl2 = await page.evaluate(
    () => document.createElement("canvas").getContext("webgl2") !== null,
  );
  test.skip(!webgl2, "WebGL2 saknas i denna webbläsare");
  if (isMobile(page)) await page.getByTestId("tab-map").click();
  await ground.liveConnected();
  await ground.resynced();
  await page.waitForFunction(
    () => (window as unknown as { __sd?: { map?: SourceMap } }).__sd?.map?.loaded() === true,
  );
  await zoomIn(page);
  await page.getByTestId("draw-rectangle").click();
  await page.waitForTimeout(200); // the drawing bar takes its place under the map
  const drawBox = await page.getByTestId("map").boundingBox();
  if (!drawBox) throw new Error("no map box");
  // Real presses on a desktop pane; small phone panes get the clicks MapLibre would report.
  if (isMobile(page)) {
    await mapClick(page, -30, 60);
    await mapClick(page, 30, 0);
  } else {
    await press(page, drawBox.x + drawBox.width * 0.55, drawBox.y + drawBox.height * 0.5);
    await press(page, drawBox.x + drawBox.width * 0.7, drawBox.y + drawBox.height * 0.75);
  }
  await page.getByTestId("draw-done").click();
  await expect.poll(() => ground.posts("/api/v1/missions/M1/annotations").length).toBe(1);
  const area = ground.posts("/api/v1/missions/M1/annotations")[0]?.body as {
    kind: string;
    label: string;
    geometry: { type: string; coordinates: number[][][] };
  };
  expect(area.kind).toBe("area");
  expect(area.label).toBe("ar-1");
  expect(area.geometry.type).toBe("Polygon");
  expect(area.geometry.coordinates[0]).toHaveLength(5);
  await expect(page.getByTestId("toasts")).toContainText("ar-1");
  if (isMobile(page)) {
    await expect(page.getByTestId("view-chat")).toBeVisible(); // a phone goes to the chat
    await page.getByTestId("tab-map").click();
  }
  await page.waitForFunction(
    () => (window as unknown as { __sd?: { map?: SourceMap } }).__sd?.map?.loaded() === true,
  );
  await expect.poll(async () => (await feats(page, "areas")).length).toBe(1);
  await page.getByTestId("draw-pin").click();
  await mapClick(page, 20, 20);
  await expect.poll(() => ground.posts("/api/v1/missions/M1/annotations").length).toBe(2);
  const pin = ground.posts("/api/v1/missions/M1/annotations")[1]?.body as {
    kind: string;
    label: string;
    geometry: { type: string };
  };
  expect(pin).toMatchObject({ kind: "pin", label: "P1", geometry: { type: "Point" } });
  await expect.poll(async () => (await feats(page, "pins")).length).toBe(1);
  expect(ground.calls(/\/(areas|pins)$/)).toEqual([]);
});

test("a self-crossing polygon is refused in the panel", async ({ page, ground }) => {
  ground.setConfig({ map: MAP });
  await ground.open();
  const webgl2 = await page.evaluate(
    () => document.createElement("canvas").getContext("webgl2") !== null,
  );
  test.skip(!webgl2, "WebGL2 saknas i denna webbläsare");
  if (isMobile(page)) await page.getByTestId("tab-map").click();
  await ground.resynced();
  await page.waitForFunction(
    () => (window as unknown as { __sd?: { map?: SourceMap } }).__sd?.map?.loaded() === true,
  );
  await zoomIn(page);
  await page.getByTestId("draw-polygon").click();
  for (const [e, n] of [
    [0, 0],
    [50, 50],
    [50, 0],
    [0, 50],
  ] as const)
    await mapClick(page, e, n);
  await page.getByTestId("draw-done").click();
  await expect(page.getByTestId("draw-bar")).toContainText("Området korsar sig självt");
  expect(ground.posts("/api/v1/missions/M1/annotations")).toEqual([]);
});
