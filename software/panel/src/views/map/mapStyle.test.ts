import { describe, expect, it } from "vitest";
import { attributionHtml, buildStyle, hiddenUnderPhoto, IMAGERY_ID, SOURCE_ID } from "./mapStyle";

const cfg = {
  pack_url: "/tiles/demo.pmtiles",
  center: { lat: -35.363261, lon: 149.16523 },
  zoom: 15,
  maxzoom: 15,
  attribution: "© OpenStreetMap contributors, Protomaps",
  assets_url: "/tiles/assets",
};
const ORIGIN = "http://127.0.0.1:8080";

describe("offline map style", () => {
  it("has a vector source with maxzoom 15, so zoom 16-18 are overzoomed", () => {
    const style = buildStyle(cfg, "dark", ORIGIN, "photo");
    const source = style.sources[SOURCE_ID];
    expect(source?.type).toBe("vector");
    expect(source && "maxzoom" in source ? source.maxzoom : undefined).toBe(15);
  });

  it("follows the pack's own maxzoom and defaults to 15", () => {
    const zoom = (c: Parameters<typeof buildStyle>[0]) => {
      const s = buildStyle(c, "dark", ORIGIN, "photo").sources[SOURCE_ID];
      return s && "maxzoom" in s ? s.maxzoom : undefined;
    };
    expect(zoom({ ...cfg, maxzoom: 13 })).toBe(13);
    const { maxzoom: _, ...noMax } = cfg;
    expect(zoom(noMax)).toBe(15);
  });

  it("loads everything from the panel's own origin", () => {
    const style = buildStyle(cfg, "light", ORIGIN, "photo");
    const source = style.sources[SOURCE_ID];
    const tiles = source && "tiles" in source ? source.tiles : [];
    expect(tiles).toEqual(["pmtiles://http://127.0.0.1:8080/tiles/demo.pmtiles/{z}/{x}/{y}"]);
    expect(style.glyphs).toBe("http://127.0.0.1:8080/tiles/assets/fonts/{fontstack}/{range}.pbf");
    expect(style.sprite).toBe("http://127.0.0.1:8080/tiles/assets/sprites/v4/light");
    const all = JSON.stringify(style);
    expect(all).not.toMatch(/https?:\/\/(?!127\.0\.0\.1:8080)/);
  });

  it("uses the Protomaps layers for the chosen theme", () => {
    const dark = buildStyle(cfg, "dark", ORIGIN, "photo");
    const light = buildStyle(cfg, "light", ORIGIN, "photo");
    expect(dark.layers.length).toBeGreaterThan(20);
    expect(dark.layers.every((l) => l.type === "background" || "source" in l)).toBe(true);
    expect(JSON.stringify(dark.layers)).not.toBe(JSON.stringify(light.layers));
    expect(dark.layers.some((l) => l.id === "earth")).toBe(true);
  });

  it("keeps the attribution plain text", () => {
    expect(attributionHtml({ ...cfg, attribution: '<img src=x onerror="a()">& ©' })).toBe(
      "&lt;img src=x onerror=&quot;a()&quot;&gt;&amp; ©",
    );
    const { attribution: _, ...noText } = cfg;
    expect(attributionHtml(noText)).toContain("OpenStreetMap");
  });

  const withPhoto = {
    ...cfg,
    imagery: {
      pack_url: "/tiles/imagery.pmtiles",
      maxzoom: 14,
      attribution: "Imagery <b>&</b> co",
    },
  };
  const visible = (style: ReturnType<typeof buildStyle>, id: string) =>
    style.layers.find((l) => l.id === id)?.layout?.visibility ?? "visible";

  it("draws the photo pack right above the background, under every map layer", () => {
    const style = buildStyle(withPhoto, "dark", ORIGIN, "photo");
    expect(style.layers[0]?.id).toBe("background");
    expect(style.layers[1]).toMatchObject({ id: IMAGERY_ID, type: "raster", source: IMAGERY_ID });
    const source = style.sources[IMAGERY_ID];
    expect(source).toMatchObject({
      type: "raster",
      tiles: ["pmtiles://http://127.0.0.1:8080/tiles/imagery.pmtiles/{z}/{x}/{y}"],
      maxzoom: 14,
      attribution: "Imagery &lt;b&gt;&amp;&lt;/b&gt; co",
    });
  });

  it("on the photo, land fills give way while water, roads and names stay", () => {
    const style = buildStyle(withPhoto, "dark", ORIGIN, "photo");
    expect(visible(style, IMAGERY_ID)).toBe("visible");
    for (const id of ["earth", "landcover", "landuse_park", "buildings"])
      expect(visible(style, id)).toBe("none");
    for (const id of ["water", "water_stream", "roads_minor", "roads_other", "places_locality"])
      expect(visible(style, id)).toBe("visible");
    const water = style.layers.find((l) => l.id === "water");
    expect(water && "paint" in water ? water.paint : {}).toMatchObject({ "fill-opacity": 0.3 });
  });

  it("leaves the panel's own fills alone (swept area, sectors, drawn areas)", () => {
    for (const source of ["swept", "sectors", "areas", "charter-preview"])
      expect(hiddenUnderPhoto({ id: `${source}-fill`, type: "fill", source })).toBe(false);
    expect(hiddenUnderPhoto({ id: "earth", type: "fill", source: SOURCE_ID })).toBe(true);
  });

  it("the map choice hides the photo and shows the drawn map", () => {
    const style = buildStyle(withPhoto, "dark", ORIGIN, "map");
    expect(visible(style, IMAGERY_ID)).toBe("none");
    expect(visible(style, "earth")).toBe("visible");
  });

  it("without a photo pack the drawn map shows, whatever was chosen", () => {
    const style = buildStyle(cfg, "dark", ORIGIN, "photo");
    expect(style.sources[IMAGERY_ID]).toBeUndefined();
    expect(visible(style, "earth")).toBe("visible");
  });
});
