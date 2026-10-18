import { Crosshair, House, Layers, MapPin, Maximize2, Pentagon, Square } from "lucide-react";
import { useEffect, useRef, useState } from "react";
import { useTranslation } from "react-i18next";
import { api, HttpError } from "../../api/http";
import { paths } from "../../api/paths";
import type { Annotation } from "../../api/types";
import { areaM2, selfIntersects } from "../../lib/geo";
import { exposeMap } from "../../lib/testHooks";
import { useLayoutStore } from "../../state/layoutStore";
import { useLiveStore } from "../../state/liveStore";
import { activeCharter, useMissionStore } from "../../state/missionStore";
import { useIsPilot, useSessionStore } from "../../state/sessionStore";
import { useUiStore } from "../../state/uiStore";
import type { ViewProps } from "../registry";
import { FindCard } from "./FindCard";
import {
  areasData,
  bounds,
  charterAreaData,
  charterPreviewData,
  drawData,
  droneData,
  type FC,
  findsData,
  homeData,
  nextLabel,
  pilotRangeData,
  pinsData,
  rectangle,
  type SourceId,
  sectorsData,
  targetData,
  trackData,
} from "./mapData";

export function hasWebGL2(): boolean {
  try {
    return document.createElement("canvas").getContext("webgl2") !== null;
  } catch {
    return false;
  }
}

function DroneText() {
  const { t } = useTranslation();
  const tel = useLiveStore((s) => s.telemetry);
  if (!tel) return <p className="muted">{t("views.map.noPosition")}</p>;
  const fmt = new Intl.NumberFormat("sv-SE", { maximumFractionDigits: 6 });
  return (
    <p className="muted">
      {t("views.map.position", { lat: fmt.format(tel.lat), lon: fmt.format(tel.lon) })}
    </p>
  );
}

/** Narrow or touch screens get the compact (folded) attribution. */
function compactAttribution(): boolean {
  return typeof matchMedia === "function" && matchMedia("(max-width: 600px)").matches;
}

type MapLike = import("maplibre-gl").Map;
type MapMod = typeof import("./createMap");
type DrawMode = "none" | "polygon" | "rectangle" | "pin";
async function noPin(_p: [number, number]): Promise<void> {}

const LAYERS_KEY = "sd.panel.map.layers.v1";
const TOGGLES: SourceId[] = [
  "charter-preview",
  "sectors",
  "pilot-range",
  "home",
  "target",
  "track",
  "areas",
  "pins",
  "finds",
];
const TRACK_EVERY_MS = 500;

function loadHidden(): SourceId[] {
  try {
    const raw = JSON.parse(localStorage.getItem(LAYERS_KEY) ?? "[]") as unknown;
    return Array.isArray(raw)
      ? (raw.filter((x) => TOGGLES.includes(x as SourceId)) as SourceId[])
      : [];
  } catch {
    return [];
  }
}

/** Everything the overlay shows, computed from the stores. */
function overlayData(): Partial<Record<SourceId, FC>> {
  const live = useLiveStore.getState();
  const m = useMissionStore.getState();
  const active = activeCharter(m, live.mission?.active_charter_id ?? null);
  const approvals = Object.values(m.approvals);
  const pendingCharter = approvals
    .filter((a) => a.kind === "charter" && a.status === "pending")
    .map((a) => m.charters[a.subject_id])
    .find((c) => c !== undefined);
  const pendingCommands = approvals
    .filter((a) => a.kind === "command" && a.status === "pending")
    .map((a) => m.commands[a.subject_id])
    .filter((c) => c !== undefined);
  const anns = Object.values(m.annotations);
  return {
    "charter-area": charterAreaData(active),
    "charter-preview": charterPreviewData(approvals, m.charters),
    sectors: sectorsData(m.sectors),
    "pilot-range": pilotRangeData(active ?? pendingCharter ?? null),
    home: homeData(active ?? pendingCharter ?? null),
    target: targetData(
      m.activeCommandId ? (m.commands[m.activeCommandId] ?? null) : null,
      pendingCommands,
    ),
    areas: areasData(anns),
    pins: pinsData(anns),
    finds: findsData(Object.values(m.finds)),
  };
}

/**
 * The map: the offline pack from sd-ground (PMTiles) or a plain background, with the drone,
 * its track, the charter, sectors, home, pilot range, target, drawn areas and pins on top.
 */
export default function MapView({ isCompact }: ViewProps) {
  const { t } = useTranslation();
  const mapCfg = useSessionStore((s) => s.config?.map ?? null);
  const theme = useUiStore((s) => s.theme);
  const isPilot = useIsPilot();
  const missionId = useLiveStore((s) => s.mission?.id ?? null);
  const box = useRef<HTMLDivElement>(null);
  const handle = useRef<{ map: MapLike; mod: MapMod } | null>(null);
  const follow = useRef(true);
  const [webgl] = useState(hasWebGL2);
  const [failed, setFailed] = useState(false);
  const [ready, setReady] = useState(false);
  const [mode, setMode] = useState<DrawMode>("none");
  const [points, setPoints] = useState<[number, number][]>([]);
  const [hidden, setHidden] = useState<SourceId[]>(loadHidden);
  const [layersOpen, setLayersOpen] = useState(false);
  const [cursor, setCursor] = useState<string | null>(null);
  const [drawError, setDrawError] = useState<string | null>(null);
  const [findId, setFindId] = useState<string | null>(null);
  const find = useMissionStore((s) => (findId ? (s.finds[findId] ?? null) : null));
  const modeRef = useRef(mode);
  modeRef.current = mode;
  const savePinRef = useRef(noPin);

  useEffect(() => {
    if (!webgl || !box.current) return;
    let disposed = false;
    let map: MapLike | null = null;
    setFailed(false);
    setReady(false);
    const tel = useLiveStore.getState().telemetry;
    const start = tel ? { lon: tel.lon, lat: tel.lat } : null;
    // A map that cannot start (chunk, worker or WebGL context) fails here alone: the view
    // shows the text fallback and the rest of the panel, STOPP included, is untouched.
    import("./createMap")
      .then((mod) => {
        if (disposed || !box.current) return;
        map = mod.createMap(box.current, mapCfg, theme, compactAttribution(), start, (m) => {
          if (disposed) return;
          handle.current = { map: m, mod };
          setReady(true);
        });
        map.on("dragstart", (e) => {
          if ((e as { originalEvent?: unknown }).originalEvent) follow.current = false;
        });
        map.on("mousemove", (e) =>
          setCursor(`${e.lngLat.lat.toFixed(5)}, ${e.lngLat.lng.toFixed(5)}`),
        );
        exposeMap(map);
      })
      .catch(() => {
        if (!disposed) setFailed(true);
      });
    return () => {
      disposed = true;
      handle.current = null;
      map?.remove();
    };
  }, [mapCfg, webgl, theme]);

  // Overlay data: on every store change, the track at most twice a second.
  useEffect(() => {
    if (!ready) return;
    const h = handle.current;
    if (!h) return;
    const push = (): void => {
      const data = overlayData();
      for (const [id, fc] of Object.entries(data)) h.mod.setData(h.map, id as SourceId, fc);
    };
    const drone = (): void => {
      const live = useLiveStore.getState();
      const age = live.lastTelemetryAt === null ? null : Date.now() - live.lastTelemetryAt;
      h.mod.setData(h.map, "drone", droneData(live.telemetry, age));
      const tel = live.telemetry;
      if (follow.current && tel && Number.isFinite(tel.lat) && modeRef.current === "none")
        h.map.easeTo({ center: [tel.lon, tel.lat], duration: 0 });
    };
    let trackDue = 0;
    const track = (): void => {
      const now = Date.now();
      if (now < trackDue) return;
      trackDue = now + TRACK_EVERY_MS;
      requestAnimationFrame(() =>
        h.mod.setData(h.map, "track", trackData(useMissionStore.getState().track)),
      );
    };
    push();
    drone();
    track();
    const unsubs = [
      useMissionStore.subscribe((s, p) => {
        if (s.track !== p.track) track();
        if (
          s.charters !== p.charters ||
          s.approvals !== p.approvals ||
          s.commands !== p.commands ||
          s.activeCommandId !== p.activeCommandId ||
          s.sectors !== p.sectors ||
          s.annotations !== p.annotations ||
          s.finds !== p.finds
        )
          push();
      }),
      useLiveStore.subscribe((s, p) => {
        if (s.telemetry !== p.telemetry) drone();
        if (s.mission !== p.mission) push();
      }),
    ];
    const tick = setInterval(drone, 1_000); // the stale mark appears without new telemetry
    return () => {
      clearInterval(tick);
      for (const u of unsubs) u();
    };
  }, [ready]);

  // A new target find opens its card; a click on a find's marker opens that one.
  useEffect(() => {
    const h = handle.current;
    if (!ready || !h) return;
    const unsub = useMissionStore.subscribe((s, p) => {
      if (s.finds === p.finds) return;
      const fresh = Object.values(s.finds).find((f) => f.kind === "target" && !p.finds[f.find_id]);
      if (fresh) setFindId(fresh.find_id);
    });
    const onFind = (e: { features?: { properties?: Record<string, unknown> }[] }): void => {
      const id = e.features?.[0]?.properties?.id;
      if (modeRef.current === "none" && typeof id === "string") setFindId(id);
    };
    h.map.on("click", "finds-point", onFind);
    return () => {
      unsub();
      h.map.off("click", "finds-point", onFind);
    };
  }, [ready]);

  // Layer switches.
  useEffect(() => {
    const h = handle.current;
    if (!ready || !h) return;
    for (const id of TOGGLES) h.mod.setVisible(h.map, id, !hidden.includes(id));
    try {
      localStorage.setItem(LAYERS_KEY, JSON.stringify(hidden));
    } catch {
      // not remembered
    }
  }, [hidden, ready]);

  // The drawing in progress.
  useEffect(() => {
    const h = handle.current;
    if (!ready || !h) return;
    const closed = mode === "rectangle" ? points.length === 4 : mode === "polygon";
    h.mod.setData(h.map, "draw", drawData(points, closed));
  }, [points, mode, ready]);

  // Clicks while drawing or placing a pin: one listener for the map's life, so a click
  // right after choosing a tool is never lost.
  useEffect(() => {
    const h = handle.current;
    if (!ready || !h) return;
    const onClick = (e: { lngLat: { lng: number; lat: number } }): void => {
      const p: [number, number] = [e.lngLat.lng, e.lngLat.lat];
      const m = modeRef.current;
      if (m === "pin") {
        modeRef.current = "none";
        setMode("none");
        void savePinRef.current(p);
      } else if (m === "rectangle") {
        setPoints((prev) => (prev.length === 1 && prev[0] ? rectangle(prev[0], p) : [p]));
      } else if (m === "polygon") {
        setPoints((prev) => [...prev, p]);
      }
    };
    h.map.on("click", onClick);
    return () => {
      h.map.off("click", onClick);
    };
  }, [ready]);

  useEffect(() => {
    const h = handle.current;
    if (!ready || !h) return;
    h.map.getCanvas().style.cursor = mode === "none" ? "" : "crosshair";
    // Quick clicks while drawing are corners, not a double-click zoom.
    if (mode === "none") h.map.doubleClickZoom.enable();
    else h.map.doubleClickZoom.disable();
  }, [mode, ready]);

  async function postAnnotation(
    kind: "pin" | "area",
    geometry: unknown,
  ): Promise<Annotation | null> {
    if (!missionId) return null;
    const attempt = (anns: Annotation[]): Promise<Annotation> => {
      const body = { kind, label: nextLabel(anns, kind), geometry };
      return api<Annotation>(paths.missionAnnotations(missionId), { method: "POST", body });
    };
    try {
      return await attempt(Object.values(useMissionStore.getState().annotations));
    } catch (e) {
      // A label clash (another client was faster): read the list again and try once more.
      if (e instanceof HttpError && (e.status === 409 || e.status === 422)) {
        const fresh = await api<Annotation[]>(paths.missionAnnotations(missionId));
        useMissionStore.getState().setAnnotations(fresh);
        return await attempt(fresh);
      }
      throw e;
    }
  }

  async function savePin(p: [number, number]): Promise<void> {
    try {
      const a = await postAnnotation("pin", { type: "Point", coordinates: p });
      if (a) useMissionStore.getState().upsertAnnotation(a);
    } catch (e) {
      useUiStore.getState().toast(e instanceof HttpError ? e.message : t("draw.failed"));
    }
  }

  savePinRef.current = savePin;

  function startDraw(m: "polygon" | "rectangle"): void {
    if (mode === m) {
      cancelDraw();
      return;
    }
    setPoints([]);
    modeRef.current = m;
    setMode(m);
  }

  async function finishArea(): Promise<void> {
    if (points.length < 3) return;
    const ring = [...points, points[0] as [number, number]];
    if (selfIntersects(ring)) {
      setDrawError(t("draw.selfIntersects"));
      return;
    }
    const polygon = { type: "Polygon", coordinates: [ring] };
    const m2 = areaM2(polygon);
    const cap = useSessionStore.getState().config?.limits_absolute.max_area_km2;
    if (cap !== undefined && m2 / 1e6 > cap) {
      setDrawError(t("draw.tooBig", { cap }));
      return;
    }
    try {
      const a = await postAnnotation("area", polygon);
      if (!a) return;
      useMissionStore.getState().upsertAnnotation(a);
      useUiStore.getState().toast(
        t("draw.saved", {
          label: a.label,
          ha: new Intl.NumberFormat("sv-SE", { maximumFractionDigits: 1 }).format(m2 / 10_000),
        }),
      );
      cancelDraw();
      if (useLayoutStore.getState().vp === "mobile") useLayoutStore.getState().showOnly("chat");
    } catch (e) {
      setDrawError(e instanceof HttpError ? e.message : t("draw.failed"));
    }
  }

  function cancelDraw(): void {
    setMode("none");
    setPoints([]);
    setDrawError(null);
  }

  function centre(to: "drone" | "home" | "charter"): void {
    const h = handle.current;
    if (!h) return;
    const live = useLiveStore.getState();
    const charter = activeCharter(
      useMissionStore.getState(),
      live.mission?.active_charter_id ?? null,
    );
    if (to === "drone") {
      follow.current = true;
      if (live.telemetry) h.map.easeTo({ center: [live.telemetry.lon, live.telemetry.lat] });
    } else if (to === "home" && charter) {
      follow.current = false;
      h.map.easeTo({ center: [charter.body.home.lon, charter.body.home.lat] });
    } else if (to === "charter" && charter) {
      const b = bounds(charter.body.area);
      follow.current = false;
      if (b) h.map.fitBounds(b, { padding: 40, duration: 300 });
    }
  }

  if (!webgl || failed) {
    return (
      <div className="view-body view-empty">
        <p>{t(webgl ? "views.map.failed" : "views.map.noWebgl")}</p>
        <DroneText />
      </div>
    );
  }
  const canDraw = isPilot && missionId !== null;
  return (
    <div className="map-wrap">
      <div ref={box} className="map-box" data-hotkeys="map" data-testid="map" />
      {!mapCfg ? <p className="map-note">{t("views.map.noPack")}</p> : null}
      <div className="map-tools" role="toolbar" aria-label={t("views.map.tools")}>
        <button
          type="button"
          className="icon-btn"
          aria-label={t("views.map.centreDrone")}
          onClick={() => centre("drone")}
        >
          <Crosshair aria-hidden="true" size={18} />
        </button>
        <button
          type="button"
          className="icon-btn"
          aria-label={t("views.map.centreHome")}
          onClick={() => centre("home")}
        >
          <House aria-hidden="true" size={18} />
        </button>
        <button
          type="button"
          className="icon-btn"
          aria-label={t("views.map.showCharter")}
          onClick={() => centre("charter")}
        >
          <Maximize2 aria-hidden="true" size={18} />
        </button>
        <button
          type="button"
          className="icon-btn"
          aria-label={t("views.map.layers")}
          aria-expanded={layersOpen}
          onClick={() => setLayersOpen(!layersOpen)}
        >
          <Layers aria-hidden="true" size={18} />
        </button>
        {canDraw ? (
          <>
            <button
              type="button"
              className="icon-btn"
              data-testid="draw-polygon"
              aria-pressed={mode === "polygon"}
              aria-label={t("draw.polygon")}
              onClick={() => startDraw("polygon")}
            >
              <Pentagon aria-hidden="true" size={18} />
            </button>
            <button
              type="button"
              className="icon-btn"
              data-testid="draw-rectangle"
              aria-pressed={mode === "rectangle"}
              aria-label={t("draw.rectangle")}
              onClick={() => startDraw("rectangle")}
            >
              <Square aria-hidden="true" size={18} />
            </button>
            <button
              type="button"
              className="icon-btn"
              data-testid="draw-pin"
              aria-pressed={mode === "pin"}
              aria-label={t("draw.pin")}
              onClick={() => {
                modeRef.current = mode === "pin" ? "none" : "pin";
                setMode(modeRef.current);
              }}
            >
              <MapPin aria-hidden="true" size={18} />
            </button>
          </>
        ) : null}
      </div>
      {layersOpen ? (
        <fieldset className="map-layers" data-testid="map-layers">
          <legend>{t("views.map.layers")}</legend>
          {TOGGLES.map((id) => (
            <label key={id}>
              <input
                type="checkbox"
                checked={!hidden.includes(id)}
                onChange={(e) =>
                  setHidden(e.target.checked ? hidden.filter((x) => x !== id) : [...hidden, id])
                }
              />
              {t(`views.map.layer.${id}`)}
            </label>
          ))}
        </fieldset>
      ) : null}
      {mode !== "none" ? (
        <div className="map-draw" role="status" data-testid="draw-bar">
          <span>
            {mode === "pin"
              ? t("draw.pinHint")
              : mode === "rectangle"
                ? t("draw.rectangleHint")
                : t("draw.polygonHint", { n: points.length })}
          </span>
          {mode !== "pin" ? (
            <button
              type="button"
              className="btn btn-small btn-primary"
              data-testid="draw-done"
              disabled={points.length < 3}
              onClick={() => void finishArea()}
            >
              {t("draw.done")}
            </button>
          ) : null}
          <button type="button" className="btn btn-small btn-outline" onClick={cancelDraw}>
            {t("limitsForm.cancel")}
          </button>
          {drawError ? <span className="error">{drawError}</span> : null}
        </div>
      ) : null}
      {find && missionId ? (
        <FindCard find={find} missionId={missionId} onClose={() => setFindId(null)} />
      ) : null}
      {cursor && !isCompact ? <p className="map-cursor">{cursor}</p> : null}
    </div>
  );
}
