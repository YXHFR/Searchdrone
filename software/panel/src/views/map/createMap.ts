import "maplibre-gl/dist/maplibre-gl.css";
import type { MapConfig } from "@searchdrone/schemas";
import * as maplibregl from "maplibre-gl";

export function createMap(container: HTMLElement, cfg: MapConfig): maplibregl.Map {
  return new maplibregl.Map({
    container,
    center: [cfg.center.lon, cfg.center.lat],
    zoom: cfg.zoom,
    attributionControl: false,
    style: {
      version: 8,
      sources: {},
      layers: [{ id: "background", type: "background", paint: { "background-color": "#16241b" } }],
    },
  });
}
