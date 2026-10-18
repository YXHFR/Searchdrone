// The panel's own map layers, on top of the base map. Loaded only in the lazy map chunk.
import type * as maplibregl from "maplibre-gl";
import { EMPTY, type FC, SOURCE_IDS, type SourceId } from "./mapData";

type GeoData = Parameters<maplibregl.GeoJSONSource["setData"]>[0];

const C = {
  acc: "#4fb3ff",
  ok: "#3ecf8e",
  warn: "#ffb547",
  bad: "#ff5a5a",
  grey: "#8a9aa6",
  white: "#ffffff",
  home: "#ffb547",
};

/** Layers that the "Lager" menu can switch off (the drone and the charter always stay). */
export const TOGGLEABLE: SourceId[] = [
  "charter-preview",
  "sectors",
  "pilot-range",
  "home",
  "target",
  "track",
  "areas",
  "pins",
];

/** Adds every source (empty) and its layers. `labels` only when the style has glyphs. */
export function addOverlay(map: maplibregl.Map, labels: boolean): void {
  for (const id of SOURCE_IDS) {
    if (!map.getSource(id))
      map.addSource(id, { type: "geojson", data: EMPTY as unknown as GeoData });
  }
  const add = (layer: maplibregl.LayerSpecification): void => {
    if (!map.getLayer(layer.id)) map.addLayer(layer);
  };
  const text = (
    id: string,
    source: SourceId,
    field: string,
    extra: Partial<maplibregl.SymbolLayerSpecification> = {},
  ) => {
    if (!labels) return;
    add({
      id,
      type: "symbol",
      source,
      layout: {
        "text-field": ["get", field],
        "text-size": 12,
        "text-font": ["Noto Sans Regular"],
        "text-offset": [0, 1.2],
        ...extra.layout,
      },
      paint: {
        "text-color": C.white,
        "text-halo-color": "#000000",
        "text-halo-width": 1.5,
        ...extra.paint,
      },
    } as maplibregl.LayerSpecification);
  };

  add({
    id: "charter-area-fill",
    type: "fill",
    source: "charter-area",
    paint: { "fill-color": C.acc, "fill-opacity": 0.1 },
  });
  add({
    id: "charter-area-line",
    type: "line",
    source: "charter-area",
    paint: { "line-color": C.acc, "line-width": 2 },
  });
  add({
    id: "charter-preview-line",
    type: "line",
    source: "charter-preview",
    paint: { "line-color": C.warn, "line-width": 2, "line-dasharray": [2, 2] },
  });
  add({
    id: "sectors-fill",
    type: "fill",
    source: "sectors",
    paint: {
      "fill-color": ["match", ["get", "status"], "done", C.ok, "active", C.warn, C.grey],
      "fill-opacity": 0.18,
    },
  });
  add({
    id: "sectors-line",
    type: "line",
    source: "sectors",
    paint: { "line-color": C.grey, "line-width": 1 },
  });
  text("sectors-label", "sectors", "label", { layout: { "text-offset": [0, 0] } });
  add({
    id: "pilot-range-line",
    type: "line",
    source: "pilot-range",
    filter: ["==", ["get", "kind"], "range"],
    paint: { "line-color": C.grey, "line-width": 1, "line-dasharray": [4, 3] },
  });
  add({
    id: "pilot-range-point",
    type: "circle",
    source: "pilot-range",
    filter: ["==", ["get", "kind"], "pilot"],
    paint: {
      "circle-radius": 5,
      "circle-color": C.white,
      "circle-stroke-color": "#000000",
      "circle-stroke-width": 1,
    },
  });
  add({
    id: "home-point",
    type: "circle",
    source: "home",
    paint: {
      "circle-radius": 7,
      "circle-color": C.home,
      "circle-stroke-color": "#000000",
      "circle-stroke-width": 2,
    },
  });
  add({
    id: "areas-fill",
    type: "fill",
    source: "areas",
    paint: { "fill-color": C.white, "fill-opacity": 0.08 },
  });
  add({
    id: "areas-line",
    type: "line",
    source: "areas",
    paint: { "line-color": C.white, "line-width": 2, "line-dasharray": [1, 1] },
  });
  text("areas-label", "areas", "label", { layout: { "text-offset": [0, 0] } });
  add({
    id: "target-fill",
    type: "fill",
    source: "target",
    filter: ["==", ["geometry-type"], "Polygon"],
    paint: { "fill-color": C.warn, "fill-opacity": 0.12 },
  });
  add({
    id: "target-point",
    type: "circle",
    source: "target",
    filter: ["==", ["geometry-type"], "Point"],
    paint: {
      "circle-radius": 8,
      "circle-color": "rgba(0,0,0,0)",
      "circle-stroke-color": C.warn,
      "circle-stroke-width": 3,
    },
  });
  add({
    id: "track-line",
    type: "line",
    source: "track",
    paint: { "line-color": C.acc, "line-width": 2, "line-opacity": 0.8 },
  });
  add({
    id: "pins-point",
    type: "circle",
    source: "pins",
    paint: {
      "circle-radius": 6,
      "circle-color": C.bad,
      "circle-stroke-color": C.white,
      "circle-stroke-width": 2,
    },
  });
  text("pins-label", "pins", "label");
  add({
    id: "draw-line",
    type: "line",
    source: "draw",
    paint: { "line-color": C.warn, "line-width": 2 },
  });
  add({
    id: "draw-fill",
    type: "fill",
    source: "draw",
    filter: ["==", ["geometry-type"], "Polygon"],
    paint: { "fill-color": C.warn, "fill-opacity": 0.15 },
  });
  add({
    id: "draw-vertex",
    type: "circle",
    source: "draw",
    filter: ["==", ["get", "kind"], "vertex"],
    paint: { "circle-radius": 4, "circle-color": C.warn },
  });
  add({
    id: "drone-point",
    type: "circle",
    source: "drone",
    paint: {
      "circle-radius": 9,
      "circle-color": ["case", ["get", "stale"], C.grey, C.ok],
      "circle-stroke-color": C.white,
      "circle-stroke-width": ["case", ["get", "stale"], 1, 3],
      "circle-opacity": ["case", ["get", "stale"], 0.5, 1],
    },
  });
  text("drone-label", "drone", "alt", {
    layout: {
      "text-field": [
        "case",
        ["get", "stale"],
        ["concat", "senast för ", ["to-string", ["get", "age_s"]], " s"],
        ["concat", ["to-string", ["get", "alt"]], " m"],
      ],
    },
  });
}

export function setData(map: maplibregl.Map, id: SourceId, data: FC): void {
  const src = map.getSource(id) as maplibregl.GeoJSONSource | undefined;
  src?.setData(data as unknown as GeoData);
}

export function setVisible(map: maplibregl.Map, id: SourceId, on: boolean): void {
  for (const layer of map.getStyle().layers ?? []) {
    if ("source" in layer && layer.source === id)
      map.setLayoutProperty(layer.id, "visibility", on ? "visible" : "none");
  }
}
