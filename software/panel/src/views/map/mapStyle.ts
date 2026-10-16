// The offline map style, built in code from @protomaps/basemaps. Everything it loads comes
// from sd-ground on the panel's own origin: the PMTiles pack and the glyphs and sprites
// under /tiles/assets/. Nothing reaches the internet.
import { layers, namedFlavor } from "@protomaps/basemaps";
import type { MapConfig } from "@searchdrone/schemas";
import type { StyleSpecification } from "maplibre-gl";

export const SOURCE_ID = "protomaps";
// The packs are cut at zoom 15 (`just maps-fetch`); MapLibre overzooms 16-18 from there.
export const PACK_MAXZOOM = 15;

export type MapFlavor = "dark" | "light";
// The server's defaults for the two optional MapConfig fields.
export const DEFAULT_ASSETS_URL = "/tiles/assets";
export const DEFAULT_ATTRIBUTION = "© OpenStreetMap contributors, Protomaps";

/** Absolute same-origin URL (MapLibre's workers need absolute URLs). */
export function absolute(origin: string, path: string): string {
  return new URL(path, origin).href;
}

export function buildStyle(cfg: MapConfig, flavor: MapFlavor, origin: string): StyleSpecification {
  const pack = absolute(origin, cfg.pack_url);
  const assets = absolute(origin, (cfg.assets_url ?? DEFAULT_ASSETS_URL).replace(/\/+$/, ""));
  return {
    version: 8,
    glyphs: `${assets}/fonts/{fontstack}/{range}.pbf`,
    sprite: `${assets}/sprites/v4/${flavor}`,
    sources: {
      [SOURCE_ID]: {
        type: "vector",
        // Tile URLs (not a TileJSON url) so the zoom range is fixed here, not read from the pack.
        tiles: [`pmtiles://${pack}/{z}/{x}/{y}`],
        minzoom: 0,
        maxzoom: PACK_MAXZOOM,
      },
    },
    layers: layers(SOURCE_ID, namedFlavor(flavor), { lang: "sv" }),
  };
}

/** The attribution as plain text: MapLibre renders it as HTML, so it is escaped first. */
export function escapeAttribution(text: string): string {
  return text.replace(
    /[&<>"']/g,
    (c) => ({ "&": "&amp;", "<": "&lt;", ">": "&gt;", '"': "&quot;", "'": "&#39;" })[c] ?? c,
  );
}
