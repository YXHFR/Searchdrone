import { describe, expect, it } from "vitest";
import { buildStyle, escapeAttribution, PACK_MAXZOOM, SOURCE_ID } from "./mapStyle";

const cfg = {
  pack_url: "/tiles/demo.pmtiles",
  center: { lat: -35.363261, lon: 149.16523 },
  zoom: 15,
  attribution: "© OpenStreetMap contributors, Protomaps",
  assets_url: "/tiles/assets",
};
const ORIGIN = "http://127.0.0.1:8080";

describe("offline map style", () => {
  it("has a vector source with maxzoom 15, so zoom 16-18 are overzoomed", () => {
    const style = buildStyle(cfg, "dark", ORIGIN);
    const source = style.sources[SOURCE_ID];
    expect(source?.type).toBe("vector");
    expect(PACK_MAXZOOM).toBe(15);
    expect(source && "maxzoom" in source ? source.maxzoom : undefined).toBe(15);
  });

  it("loads everything from the panel's own origin", () => {
    const style = buildStyle(cfg, "light", ORIGIN);
    const source = style.sources[SOURCE_ID];
    const tiles = source && "tiles" in source ? source.tiles : [];
    expect(tiles).toEqual(["pmtiles://http://127.0.0.1:8080/tiles/demo.pmtiles/{z}/{x}/{y}"]);
    expect(style.glyphs).toBe("http://127.0.0.1:8080/tiles/assets/fonts/{fontstack}/{range}.pbf");
    expect(style.sprite).toBe("http://127.0.0.1:8080/tiles/assets/sprites/v4/light");
    const all = JSON.stringify(style);
    expect(all).not.toMatch(/https?:\/\/(?!127\.0\.0\.1:8080)/);
  });

  it("uses the Protomaps layers for the chosen theme", () => {
    const dark = buildStyle(cfg, "dark", ORIGIN);
    const light = buildStyle(cfg, "light", ORIGIN);
    expect(dark.layers.length).toBeGreaterThan(20);
    expect(dark.layers.every((l) => l.type === "background" || "source" in l)).toBe(true);
    expect(JSON.stringify(dark.layers)).not.toBe(JSON.stringify(light.layers));
    expect(dark.layers.some((l) => l.id === "earth")).toBe(true);
  });

  it("keeps the attribution plain text", () => {
    expect(escapeAttribution('<img src=x onerror="a()">& ©')).toBe(
      "&lt;img src=x onerror=&quot;a()&quot;&gt;&amp; ©",
    );
  });
});
