// The map layers for drawn paths and the path being flown, in their own sources so they stay
// apart from the other overlay layers. Loaded only in the lazy map chunk.
import type * as maplibregl from "maplibre-gl";
import { EMPTY, type FC } from "./mapData";
import { PATH_SOURCE_IDS, type PathSourceId } from "./pathData";

type GeoData = Parameters<maplibregl.GeoJSONSource["setData"]>[0];

const PLAN = "#ffb547";
const FLOWN = "#3ecf8e";
const DRAWN = "#ffffff";

/** Adds the path sources (empty) and their layers, under the drone. */
export function addPathLayers(map: maplibregl.Map, labels: boolean): void {
  for (const id of PATH_SOURCE_IDS) {
    if (!map.getSource(id))
      map.addSource(id, { type: "geojson", data: EMPTY as unknown as GeoData });
  }
  const below = map.getLayer("drone-point") ? "drone-point" : undefined;
  const add = (layer: maplibregl.LayerSpecification): void => {
    if (!map.getLayer(layer.id)) map.addLayer(layer, below);
  };
  const isLine: maplibregl.FilterSpecification = ["==", ["geometry-type"], "LineString"];
  add({
    id: "paths-line",
    type: "line",
    source: "paths",
    filter: isLine,
    layout: { "line-cap": "round", "line-join": "round" },
    paint: { "line-color": DRAWN, "line-width": 2, "line-dasharray": [2, 2] },
  });
  add({
    id: "paths-start",
    type: "circle",
    source: "paths",
    filter: ["==", ["get", "kind"], "start"],
    paint: { "circle-radius": 4, "circle-color": DRAWN },
  });
  add({
    id: "path-plan-line",
    type: "line",
    source: "path-plan",
    filter: isLine,
    layout: { "line-cap": "round", "line-join": "round" },
    paint: { "line-color": PLAN, "line-width": 3, "line-dasharray": [2, 1.5] },
  });
  add({
    id: "path-plan-point",
    type: "circle",
    source: "path-plan",
    filter: ["==", ["get", "kind"], "point"],
    paint: {
      "circle-radius": 5,
      "circle-color": "#000000",
      "circle-stroke-color": PLAN,
      "circle-stroke-width": 2,
    },
  });
  add({
    id: "path-flown-line",
    type: "line",
    source: "path-flown",
    layout: { "line-cap": "round", "line-join": "round" },
    paint: { "line-color": FLOWN, "line-width": 4 },
  });
  if (!labels) return;
  const text = (id: string, source: PathSourceId, filter: maplibregl.FilterSpecification): void =>
    add({
      id,
      type: "symbol",
      source,
      filter,
      layout: {
        "text-field": ["coalesce", ["get", "n"], ["get", "label"]],
        "text-size": 12,
        "text-font": ["Noto Sans Regular"],
        "text-offset": [0, 1.2],
      },
      paint: { "text-color": "#ffffff", "text-halo-color": "#000000", "text-halo-width": 1.5 },
    });
  text("paths-label", "paths", ["==", ["get", "kind"], "start"]);
  text("path-plan-label", "path-plan", ["==", ["get", "kind"], "point"]);
}

export function setPathData(map: maplibregl.Map, id: PathSourceId, data: FC): void {
  const src = map.getSource(id) as maplibregl.GeoJSONSource | undefined;
  src?.setData(data as unknown as GeoData);
}

/** The "Lager" switch for drawn paths and the path being flown. */
export function setPathsVisible(map: maplibregl.Map, on: boolean): void {
  for (const layer of map.getStyle().layers ?? []) {
    if ("source" in layer && (PATH_SOURCE_IDS as readonly string[]).includes(String(layer.source)))
      map.setLayoutProperty(layer.id, "visibility", on ? "visible" : "none");
  }
}
