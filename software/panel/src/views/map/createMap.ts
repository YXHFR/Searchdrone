import "maplibre-gl/dist/maplibre-gl.css";
import type { MapConfig } from "@searchdrone/schemas";
import * as maplibregl from "maplibre-gl";
// MapLibre finds its worker next to its own module by default, which a bundle breaks. Vite
// bundles the worker as its own same-origin file (CSP worker-src 'self').
import workerUrl from "maplibre-gl/dist/maplibre-gl-worker.mjs?worker&url";
import { Protocol } from "pmtiles";
import { addOverlay, setSweptLook } from "./layers";
import {
  attributionHtml,
  type Basemap,
  buildStyle,
  hiddenUnderPhoto,
  IMAGERY_ID,
  type MapFlavor,
  PHOTO_WATER_OPACITY,
} from "./mapStyle";
import { addPathLayers } from "./pathLayers";

export { setData, setVisible } from "./layers";
export { setPathData, setPathsVisible } from "./pathLayers";

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
  cfg: MapConfig | null,
  flavor: MapFlavor,
  basemap: Basemap,
  compact: boolean,
  start: { lon: number; lat: number } | null,
  onReady: (map: maplibregl.Map) => void,
): maplibregl.Map {
  setUpMapLibre();
  // Without a map pack: a plain background; every overlay layer still works.
  const style: maplibregl.StyleSpecification = cfg
    ? buildStyle(cfg, flavor, window.location.origin, basemap)
    : {
        version: 8,
        sources: {},
        layers: [
          {
            id: "background",
            type: "background",
            paint: { "background-color": flavor === "dark" ? "#1d262c" : "#e8eef2" },
          },
        ],
      };
  const center: [number, number] = cfg
    ? [cfg.center.lon, cfg.center.lat]
    : start
      ? [start.lon, start.lat]
      : [0, 0];
  const map = new maplibregl.Map({
    container,
    center,
    zoom: cfg ? cfg.zoom : start ? 16 : 2,
    maxZoom: 18,
    style,
    attributionControl: cfg ? { compact, customAttribution: attributionHtml(cfg) } : false,
  });
  map.addControl(new maplibregl.ScaleControl({ unit: "metric" }), "bottom-left");
  map.addControl(
    new maplibregl.NavigationControl({ showZoom: false, visualizePitch: false }),
    "top-right",
  );
  map.on("load", () => {
    addOverlay(map, cfg !== null, flavor);
    addPathLayers(map, cfg !== null);
    onReady(map);
  });
  return map;
}

/** Photo or drawn map under the roads, paths and names; no-op without a photo pack. */
export function setBasemap(map: maplibregl.Map, basemap: Basemap, flavor: MapFlavor): void {
  if (!map.getLayer(IMAGERY_ID)) return;
  const photo = basemap === "photo";
  setSweptLook(map, flavor, photo);
  map.setLayoutProperty(IMAGERY_ID, "visibility", photo ? "visible" : "none");
  for (const layer of map.getStyle().layers)
    if (hiddenUnderPhoto(layer))
      map.setLayoutProperty(layer.id, "visibility", photo ? "none" : "visible");
  if (map.getLayer("water"))
    map.setPaintProperty("water", "fill-opacity", photo ? PHOTO_WATER_OPACITY : 1);
}
