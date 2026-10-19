// The offline map style, built in code from @protomaps/basemaps. Everything it loads comes
// from sd-ground on the panel's own origin: the PMTiles packs and the glyphs and sprites
// under /tiles/assets/. Nothing reaches the internet.
import { layers, namedFlavor } from "@protomaps/basemaps";
import type { MapConfig } from "@searchdrone/schemas";
import type { LayerSpecification, StyleSpecification } from "maplibre-gl";

export const SOURCE_ID = "protomaps";
export const IMAGERY_ID = "imagery";
// The packs are cut at zoom 15 by default (`just maps-fetch`); sd-ground reads the real value
// from the pack header (MapConfig.maxzoom) and MapLibre overzooms above it, up to 18.
const DEFAULT_PACK_MAXZOOM = 15;
// sd-ground always sends these (defaults in sd_schemas.api.MapConfig); the fallbacks only
// satisfy the optional fields of the generated type.
const DEFAULT_ASSETS_URL = "/tiles/assets";
const DEFAULT_ATTRIBUTION = "© OpenStreetMap contributors, Protomaps";

export type MapFlavor = "dark" | "light";
/** What lies under the roads, paths and names: the aerial photo or the drawn map. */
export type Basemap = "photo" | "map";

/** The map's land, landuse and building fills give way to the photo; water, roads and names
 * stay, and so does everything the panel draws on top (its own sources). */
export function hiddenUnderPhoto(layer: { id: string; type: string; source?: unknown }): boolean {
  return layer.source === SOURCE_ID && layer.type === "fill" && layer.id !== "water";
}
/** Water stays as a tint on the photo, so the shore in the photo shows through. */
export const PHOTO_WATER_OPACITY = 0.3;

/** Absolute same-origin URL (MapLibre's workers need absolute URLs). */
function absolute(origin: string, path: string): string {
  return new URL(path, origin).href;
}

export function buildStyle(
  cfg: MapConfig,
  flavor: MapFlavor,
  origin: string,
  basemap: Basemap,
): StyleSpecification {
  const pack = absolute(origin, cfg.pack_url);
  const assets = absolute(origin, (cfg.assets_url ?? DEFAULT_ASSETS_URL).replace(/\/+$/, ""));
  const imagery = cfg.imagery ?? null;
  const photo = imagery !== null && basemap === "photo";
  const vector = layers(SOURCE_ID, namedFlavor(flavor), { lang: "sv" }).map((l) => {
    if (!photo) return l;
    if (hiddenUnderPhoto(l)) return { ...l, layout: { ...l.layout, visibility: "none" } };
    if (l.id === "water")
      return { ...l, paint: { ...l.paint, "fill-opacity": PHOTO_WATER_OPACITY } };
    return l;
  }) as LayerSpecification[];
  const style: StyleSpecification = {
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
    layers: vector,
  };
  if (imagery) {
    style.sources[IMAGERY_ID] = {
      type: "raster",
      tiles: [`pmtiles://${absolute(origin, imagery.pack_url)}/{z}/{x}/{y}`],
      tileSize: 256,
      minzoom: 0,
      maxzoom: imagery.maxzoom,
      // MapLibre shows a source's attribution only while one of its layers is visible.
      attribution: escapeHtml(imagery.attribution),
    };
    // Right above the background, so every map layer that stays is drawn on the photo.
    style.layers.splice(1, 0, {
      id: IMAGERY_ID,
      type: "raster",
      source: IMAGERY_ID,
      layout: { visibility: photo ? "visible" : "none" },
    });
  }
  return style;
}

function escapeHtml(text: string): string {
  return text.replace(
    /[&<>"']/g,
    (c) => ({ "&": "&amp;", "<": "&lt;", ">": "&gt;", '"': "&quot;", "'": "&#39;" })[c] ?? c,
  );
}

/** The attribution as plain text: MapLibre renders it as HTML, so it is escaped first. */
export function attributionHtml(cfg: MapConfig): string {
  return escapeHtml(cfg.attribution ?? DEFAULT_ATTRIBUTION);
}
