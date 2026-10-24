// The probability map on the map: the heat layer (where the person may be) and the sought
// layer (how well each cell was seen), as image sources the ground draws one pixel per
// cell. The map only ranks where to look: nothing here can start a flight.
import { useEffect, useState } from "react";
import { api } from "../../api/http";
import { paths } from "../../api/paths";
import type { ProbabilityMapView } from "../../api/types";
import { useMissionStore } from "../../state/missionStore";

type MapLike = import("maplibre-gl").Map;
type MapMod = typeof import("./createMap");

export type ProbLayer = "poa" | "pod";
export const PROB_LAYERS: readonly ProbLayer[] = ["poa", "pod"];
const LAYER_ID: Record<ProbLayer, string> = { poa: "prob-poa", pod: "prob-pod" };
const OPACITY: Record<ProbLayer, number> = { poa: 0.75, pod: 0.6 };
/** The panel's own layers start here: the heat lies under the sectors and the drawings. */
const BELOW = "swept-fill";

/** What the legend and the notes need; the arrays stay with the ground's image. */
export interface ProbMeta {
  version: number;
  row: number;
  frozen: boolean;
  notes: NonNullable<ProbabilityMapView["notes"]>;
  grid: ProbabilityMapView["grid"];
}

/** The image's corners for a MapLibre image source: NW, NE, SE, SW. */
export function corners(grid: ProbabilityMapView["grid"]): [number, number][] {
  return [
    [grid.west, grid.north],
    [grid.east, grid.north],
    [grid.east, grid.south],
    [grid.west, grid.south],
  ];
}

/** "12 %": the chance the person is outside the map, in whole percent. */
export function rowPercent(row: number): string {
  const pct = Math.round(Math.min(Math.max(row, 0), 1) * 100);
  return `${new Intl.NumberFormat("sv-SE").format(pct)} %`;
}

function addOrUpdate(
  map: MapLike,
  layer: ProbLayer,
  missionId: string,
  meta: ProbMeta,
  visible: boolean,
): void {
  const id = LAYER_ID[layer];
  const url = `${paths.probabilityPng(missionId, layer)}&v=${meta.version}`;
  const coordinates = corners(meta.grid) as [
    [number, number],
    [number, number],
    [number, number],
    [number, number],
  ];
  const src = map.getSource(id) as import("maplibre-gl").ImageSource | undefined;
  if (src) src.updateImage({ url, coordinates });
  else map.addSource(id, { type: "image", url, coordinates });
  if (!map.getLayer(id))
    map.addLayer(
      {
        id,
        type: "raster",
        source: id,
        paint: {
          "raster-opacity": OPACITY[layer],
          "raster-resampling": "nearest",
          "raster-fade-duration": 0,
        },
      },
      map.getLayer(BELOW) ? BELOW : undefined,
    );
  map.setLayoutProperty(id, "visibility", visible ? "visible" : "none");
}

/** Keeps the probability layers in step with the newest map version of the mission. */
export function useProbabilityLayers(
  ready: boolean,
  handle: { current: { map: MapLike; mod: MapMod } | null },
  missionId: string | null,
  shown: Record<ProbLayer, boolean>,
): ProbMeta | null {
  const rev = useMissionStore((s) => s.mapRev);
  const [meta, setMeta] = useState<ProbMeta | null>(null);

  // A new mission starts without layers; a new version keeps the old ones until it arrives.
  // biome-ignore lint/correctness/useExhaustiveDependencies: missionId is the reset signal
  useEffect(() => {
    setMeta(null);
  }, [missionId]);

  // biome-ignore lint/correctness/useExhaustiveDependencies: rev is the refetch signal
  useEffect(() => {
    if (!missionId) return;
    let stale = false;
    // null: no map yet (no approved profile). An error leaves the layers as they were.
    api<ProbabilityMapView | null>(paths.probabilityMap(missionId))
      .then((v) => {
        if (stale) return;
        if (v === null) {
          setMeta(null);
          return;
        }
        const { version, row, grid } = v;
        setMeta({ version, row, grid, frozen: v.frozen ?? false, notes: v.notes ?? [] });
      })
      .catch(() => undefined);
    return () => {
      stale = true;
    };
  }, [missionId, rev]);

  useEffect(() => {
    const h = handle.current;
    if (!ready || !h || !meta || !missionId) return;
    for (const layer of PROB_LAYERS) addOrUpdate(h.map, layer, missionId, meta, shown[layer]);
  }, [ready, handle, meta, missionId, shown]);

  useEffect(() => {
    const h = handle.current;
    if (!ready || !h || meta) return;
    for (const layer of PROB_LAYERS) {
      const id = LAYER_ID[layer];
      if (h.map.getLayer(id)) h.map.setLayoutProperty(id, "visibility", "none");
    }
  }, [ready, handle, meta]);

  return meta;
}
