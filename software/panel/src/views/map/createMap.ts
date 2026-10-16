import "maplibre-gl/dist/maplibre-gl.css";
import type { MapConfig } from "@searchdrone/schemas";
import * as maplibregl from "maplibre-gl";
// MapLibre finds its worker next to its own module by default, which a bundle breaks. Vite
// bundles the worker as its own same-origin file (CSP worker-src 'self').
import workerUrl from "maplibre-gl/dist/maplibre-gl-worker.mjs?worker&url";
import { Protocol } from "pmtiles";
import { attributionHtml, buildStyle, type MapFlavor } from "./mapStyle";

// The worker URL and the pmtiles:// protocol are set once, here in the lazily loaded map
// chunk, so that neither MapLibre nor PMTiles ends up in the main bundle.
let protocolAdded = false;

function setUpMapLibre(): void {
  if (protocolAdded) return;
  maplibregl.setWorkerUrl(workerUrl);
  maplibregl.addProtocol("pmtiles", new Protocol().tile);
  protocolAdded = true;
}

/**
 * Never add MapLibre's FullscreenControl or cooperativeGestures here: their overlays use
 * z-index 99999 / the browser's top layer and would cover STOPP and FLYG HEM. The panel's own
 * full-screen view (layout) keeps the safety bar on top.
 */
export function createMap(
  container: HTMLElement,
  cfg: MapConfig,
  flavor: MapFlavor,
  compact: boolean,
): maplibregl.Map {
  setUpMapLibre();
  return new maplibregl.Map({
    container,
    center: [cfg.center.lon, cfg.center.lat],
    zoom: cfg.zoom,
    maxZoom: 18,
    style: buildStyle(cfg, flavor, window.location.origin),
    attributionControl: {
      compact,
      customAttribution: attributionHtml(cfg),
    },
  });
}
