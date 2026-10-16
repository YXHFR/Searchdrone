// The offline map style, built in code from @protomaps/basemaps. Everything it loads comes
// from sd-ground on the panel's own origin: the PMTiles pack and the glyphs and sprites
// under /tiles/assets/. Nothing reaches the internet.
import { layers, namedFlavor } from "@protomaps/basemaps";
import type { MapConfig } from "@searchdrone/schemas";
import type { StyleSpecification } from "maplibre-gl";

export const SOURCE_ID = "protomaps";
// The packs are cut at zoom 15 by default (`just maps-fetch`); sd-ground reads the real value
// from the pack header (MapConfig.maxzoom) and MapLibre overzooms above it, up to 18.
const DEFAULT_PACK_MAXZOOM = 15;
// sd-ground always sends these (defaults in sd_schemas.api.MapConfig); the fallbacks only
// satisfy the optional fields of the generated type.
const DEFAULT_ASSETS_URL = "/tiles/assets";
const DEFAULT_ATTRIBUTION = "© OpenStreetMap contributors, Protomaps";

export type MapFlavor = "dark" | "light";

/** Absolute same-origin URL (MapLibre's workers need absolute URLs). */
function absolute(origin: string, path: string): string {
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
        maxzoom: cfg.maxzoom ?? DEFAULT_PACK_MAXZOOM,
      },
    },
    layers: layers(SOURCE_ID, namedFlavor(flavor), { lang: "sv" }),
  };
}

/** The attribution as plain text: MapLibre renders it as HTML, so it is escaped first. */
export function attributionHtml(cfg: MapConfig): string {
  return (cfg.attribution ?? DEFAULT_ATTRIBUTION).replace(
    /[&<>"']/g,
    (c) => ({ "&": "&amp;", "<": "&lt;", ">": "&gt;", '"': "&quot;", "'": "&#39;" })[c] ?? c,
  );
}
