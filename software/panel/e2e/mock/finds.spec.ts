// Finds on the map: a red block found by the colour detector and a car from the
// simulator's labels arrive live. The block's card opens by itself with its class, how
// sure the detector is, its thumbnail and that it is not confirmed; the car is marked
// "facit". The safety bar stays visible over it all.
import type { Find, MapConfig } from "@searchdrone/schemas";
import { HOME } from "../fixtures";
import { expect, test } from "../helpers/fakeGround";
import { expectSafetyVisible, isMobile } from "../helpers/safety";

const MAP: MapConfig = {
  pack_url: "/tiles/fixture.pmtiles",
  center: HOME,
  zoom: 17,
  maxzoom: 15,
  attribution: "© OpenStreetMap contributors, Protomaps",
  assets_url: "/tiles/assets",
};

/** A place `north` and `east` metres from home. */
function near(north: number, east: number): { lat: number; lon: number } {
  const perDegLat = 111_320;
  const perDegLon = perDegLat * Math.cos((HOME.lat * Math.PI) / 180);
  return { lat: HOME.lat + north / perDegLat, lon: HOME.lon + east / perDegLon };
}

const BLOCK: Find = {
  find_id: "0192f4a1-7b90-7a22-b1c3-77aa00bb11cc",
  label: "red_block",
  name_sv: "rött block",
  kind: "target",
  source: "color",
  ...near(33, 18),
  sigma95_m: 4.6,
  score: 0.71,
  count: 3,
  thumb_detection_id: "0192f4a1-7b90-7a22-b1c3-77aa00bb11dd",
  synthetic: true,
};
const CAR: Find = {
  ...BLOCK,
  find_id: "0192f4a1-7b90-7a22-b1c3-77aa00bb11ee",
  label: "car",
  name_sv: "bil",
  kind: "context",
  source: "oracle",
  ...near(11, -18),
  score: 1,
  count: 1,
  thumb_detection_id: null,
};

interface TestMap {
  loaded(): boolean;
  jumpTo(o: { center: [number, number]; zoom: number }): void;
  once(type: "idle"): Promise<unknown>;
  queryRenderedFeatures(o: { layers: string[] }): { properties: Record<string, unknown> }[];
}

test("a find shows on the map with its class, score and thumbnail", async ({ page, ground }) => {
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
  await ground.liveConnected();
  await ground.resynced();

  ground.pushLive("detections", "find.new", CAR);
  ground.pushLive("detections", "find.new", BLOCK);

  const card = page.getByTestId("find-card");
  await expect(card).toBeVisible();
  await expect(card).toContainText("rött block");
  await expect(card).toContainText("Säkerhet 71");
  await expect(card).toContainText("Sett 3 gånger");
  await expect(card).toContainText("Färgdetektor");
  await expect(card).toContainText("Inte bekräftat");
  const thumb = page.getByTestId("find-thumb");
  await expect
    .poll(() => thumb.evaluate((img) => (img as HTMLImageElement).naturalWidth))
    .toBeGreaterThan(0);
  expect(ground.calls(/\/detections\/[^/]+\/crop$/).length).toBeGreaterThan(0);

  const labels = await page.evaluate(
    async ({ center }) => {
      const map = (window as unknown as { __sd: { map: TestMap } }).__sd.map;
      const idle = map.once("idle");
      map.jumpTo({ center: [center.lon, center.lat], zoom: 17 });
      await idle;
      return map
        .queryRenderedFeatures({ layers: ["finds-point"] })
        .map((f) => String(f.properties.label).replace(/\s/g, " ")) // sv-SE: "71 %" uses a no-break space
        .sort();
    },
    { center: HOME },
  );
  expect(labels).toEqual(["bil (facit)", "rött block 71 % ×3"]);
  await expectSafetyVisible(page);
  await page.screenshot({ path: test.info().outputPath("find-on-map.png") });

  await card.getByRole("button", { name: "Stäng" }).click();
  await expect(card).toBeHidden();
});
