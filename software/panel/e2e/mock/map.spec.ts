// The offline map (B1-11) with every other host blocked by the fake ground station: the
// synthetic test pack is drawn at zoom 12, 15 and 18 (16-18 overzoomed from 15), with no
// tile or glyph errors and the OpenStreetMap attribution visible.
import type { MapConfig } from "@searchdrone/schemas";
import { HOME } from "../fixtures";
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
    if (m.type() === "error") errors.push(m.text());
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

test("without a map pack the map view says so and requests no tiles", async ({ page, ground }) => {
  await ground.open();
  if (isMobile(page)) await page.getByTestId("tab-map").click();
  await expect(page.getByText("Ingen offlinekarta laddad")).toBeVisible();
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
