import {
  Crosshair,
  House,
  Layers,
  MapPin,
  Maximize2,
  Pencil,
  Pentagon,
  Route,
  Square,
} from "lucide-react";
import { useEffect, useRef, useState } from "react";
import { useTranslation } from "react-i18next";
import { api, HttpError } from "../../api/http";
import { paths } from "../../api/paths";
import type { Annotation } from "../../api/types";
import { Menu } from "../../header/Menu";
import { areaM2, selfIntersects } from "../../lib/geo";
import { exposeMap } from "../../lib/testHooks";
import { useLiveStore } from "../../state/liveStore";
import { activeCharter, useMissionStore } from "../../state/missionStore";
import { useIsPilot, useSessionStore } from "../../state/sessionStore";
import { useShellStore } from "../../state/shellStore";
import { useUiStore } from "../../state/uiStore";
import { useTheme } from "../../theme/theme";
import type { ViewProps } from "../registry";
import {
  areasData,
  bounds,
  charterAreaData,
  charterPreviewData,
  drawData,
  droneData,
  type FC,
  homeData,
  nextLabel,
  pilotRangeData,
  pinsData,
  rectangle,
  type SourceId,
  sectorsData,
  sweptData,
  targetData,
  trackData,
} from "./mapData";
import { formatLength, lengthM, nextPathLabel, pathProblem } from "./pathData";
import { usePathOverlay } from "./pathOverlay";

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
type Basemap = import("./mapStyle").Basemap;
type DrawMode = "none" | "polygon" | "rectangle" | "path" | "pin";
type ToggleId = SourceId | "paths";
async function noPin(_p: [number, number]): Promise<void> {}

const LAYERS_KEY = "sd.panel.map.layers.v1";
const TOGGLES: ToggleId[] = [
  "charter-preview",
  "sectors",
  "swept",
  "pilot-range",
  "home",
  "target",
  "track",
  "areas",
  "pins",
  "paths",
];
const TRACK_EVERY_MS = 500;
const BASEMAP_KEY = "sd.panel.map.basemap.v1";

function loadBasemap(): Basemap {
  try {
    return localStorage.getItem(BASEMAP_KEY) === "map" ? "map" : "photo";
  } catch {
    return "photo";
  }
}

function loadHidden(): ToggleId[] {
  try {
    const raw = JSON.parse(localStorage.getItem(LAYERS_KEY) ?? "[]") as unknown;
    return Array.isArray(raw)
      ? (raw.filter((x) => TOGGLES.includes(x as ToggleId)) as ToggleId[])
      : [];
  } catch {
    return [];
  }
}

/** The charter in force, or else the one waiting for approval: its home and area frame the map. */
function framingCharter() {
  const m = useMissionStore.getState();
  const active = activeCharter(m, useLiveStore.getState().mission?.active_charter_id ?? null);
  if (active) return active;
  return (
    Object.values(m.approvals)
      .filter((a) => a.kind === "charter" && a.status === "pending")
      .map((a) => m.charters[a.subject_id])
      .find((c) => c !== undefined) ?? null
  );
}

/** Everything the overlay shows, computed from the stores. */
function overlayData(): Partial<Record<SourceId, FC>> {
  const m = useMissionStore.getState();
  const active = activeCharter(m, useLiveStore.getState().mission?.active_charter_id ?? null);
  const framing = framingCharter();
  const approvals = Object.values(m.approvals);
  const pendingCommands = approvals
    .filter((a) => a.kind === "command" && a.status === "pending")
    .map((a) => m.commands[a.subject_id])
    .filter((c) => c !== undefined);
  const anns = Object.values(m.annotations);
  return {
    "charter-area": charterAreaData(active),
    "charter-preview": charterPreviewData(approvals, m.charters),
    sectors: sectorsData(m.sectors),
    swept: sweptData(m.sectors),
    "pilot-range": pilotRangeData(framing),
    home: homeData(framing),
    target: targetData(
      m.activeCommandId ? (m.commands[m.activeCommandId] ?? null) : null,
      pendingCommands,
    ),
    areas: areasData(anns),
    pins: pinsData(anns),
  };
}

/**
 * The map: the offline pack from sd-ground (PMTiles) or a plain background, with the drone,
 * its track, the charter, sectors, home, pilot range, target, drawn areas and pins on top.
 */
export default function MapView({ isCompact }: ViewProps) {
  const { t } = useTranslation();
  const mapCfg = useSessionStore((s) => s.config?.map ?? null);
  const theme = useTheme();
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
  const [hidden, setHidden] = useState<ToggleId[]>(loadHidden);
  const [basemap, setBasemap] = useState<Basemap>(loadBasemap);
  const basemapRef = useRef(basemap);
  basemapRef.current = basemap;
  const [cursor, setCursor] = useState<string | null>(null);
  const [drawError, setDrawError] = useState<string | null>(null);
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
        map = mod.createMap(
          box.current,
          mapCfg,
          theme,
          basemapRef.current,
          compactAttribution(),
          start,
          (m) => {
            if (disposed) return;
            handle.current = { map: m, mod };
            setReady(true);
          },
        );
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
          s.annotations !== p.annotations
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

  // Layer switches.
  useEffect(() => {
    const h = handle.current;
    if (!ready || !h) return;
    for (const id of TOGGLES) if (id !== "paths") h.mod.setVisible(h.map, id, !hidden.includes(id));
    try {
      localStorage.setItem(LAYERS_KEY, JSON.stringify(hidden));
    } catch {
      // not remembered
    }
  }, [hidden, ready]);

  useEffect(() => {
    const h = handle.current;
    if (!ready || !h) return;
    h.mod.setBasemap(h.map, basemap, theme);
    try {
      localStorage.setItem(BASEMAP_KEY, basemap);
    } catch {
      // not remembered
    }
  }, [basemap, ready, theme]);

  const shownPath = usePathOverlay(ready, handle, !hidden.includes("paths"));

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
      } else if (m === "polygon" || m === "path") {
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
    kind: "pin" | "area" | "path",
    geometry: unknown,
  ): Promise<Annotation | null> {
    if (!missionId) return null;
    const attempt = (anns: Annotation[]): Promise<Annotation> => {
      const label = kind === "path" ? nextPathLabel(anns) : nextLabel(anns, kind);
      const body = { kind, label, geometry };
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

  function startDraw(m: "polygon" | "rectangle" | "path"): void {
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
      useShellStore.getState().showTab("chat");
    } catch (e) {
      setDrawError(e instanceof HttpError ? e.message : t("draw.failed"));
    }
  }

  async function finishPath(): Promise<void> {
    const live = useLiveStore.getState();
    const charter = activeCharter(
      useMissionStore.getState(),
      live.mission?.active_charter_id ?? null,
    );
    const problem = pathProblem(points, charter?.body.area ?? null);
    if (problem) {
      const { key, ...values } = problem;
      setDrawError(t(key, values));
      return;
    }
    try {
      const a = await postAnnotation("path", { type: "LineString", coordinates: points });
      if (!a) return;
      useMissionStore.getState().upsertAnnotation(a);
      useUiStore.getState().toast(
        t("draw.pathSaved", {
          label: a.label,
          n: points.length,
          length: formatLength(lengthM(points)),
        }),
      );
      cancelDraw();
      useShellStore.getState().showTab("chat");
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
    const charter = framingCharter();
    if (to === "drone") {
      follow.current = true;
      if (live.telemetry) h.map.easeTo({ center: [live.telemetry.lon, live.telemetry.lat] });
    } else if (to === "home" && charter) {
      follow.current = false;
      h.map.easeTo({ center: [charter.body.home.lon, charter.body.home.lat] });
    } else if (to === "charter" && charter) {
      const b = bounds(charter.body.area);
      const { lon, lat } = charter.body.home;
      follow.current = false;
      if (b)
        h.map.fitBounds(
          [
            [Math.min(b[0][0], lon), Math.min(b[0][1], lat)],
            [Math.max(b[1][0], lon), Math.max(b[1][1], lat)],
          ],
          { padding: { top: 56, bottom: 48, left: 40, right: 40 }, duration: 300 },
        );
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
      <div className="map-stage">
        <div ref={box} className="map-box" data-hotkeys="map" data-testid="map" />
        {!mapCfg ? (
          <p className="map-note">{t("views.map.noPack")}</p>
        ) : !mapCfg.imagery && basemap === "photo" ? (
          <p className="map-note" data-testid="no-imagery">
            {t("views.map.noImagery")}
          </p>
        ) : null}
        <div className="map-tools" role="toolbar" aria-label={t("views.map.tools")}>
          <button type="button" className="tool" onClick={() => centre("drone")}>
            <Crosshair aria-hidden="true" size={15} />
            {t("views.map.centreDrone")}
          </button>
          <button type="button" className="tool" onClick={() => centre("home")}>
            <House aria-hidden="true" size={15} />
            {t("views.map.centreHome")}
          </button>
          <button type="button" className="tool" onClick={() => centre("charter")}>
            <Maximize2 aria-hidden="true" size={15} />
            {t("views.map.showCharter")}
          </button>
          <Menu
            tool
            label={t("views.map.layers")}
            icon={<Layers aria-hidden="true" size={15} />}
            testId="map-layers-menu"
          >
            {() => (
              <>
                <fieldset className="map-layers map-basemap" data-testid="map-basemap">
                  <legend>{t("views.map.basemap")}</legend>
                  {(["photo", "map"] as const).map((b) => (
                    <label key={b}>
                      <input
                        type="radio"
                        name="basemap"
                        data-testid={`basemap-${b}`}
                        checked={basemap === b}
                        onChange={() => setBasemap(b)}
                      />
                      {t(`views.map.basemapChoice.${b}`)}
                    </label>
                  ))}
                </fieldset>
                <fieldset className="map-layers" data-testid="map-layers">
                  <legend>{t("views.map.overlays")}</legend>
                  {TOGGLES.map((id) => (
                    <label key={id}>
                      <input
                        type="checkbox"
                        checked={!hidden.includes(id)}
                        onChange={(e) =>
                          setHidden(
                            e.target.checked ? hidden.filter((x) => x !== id) : [...hidden, id],
                          )
                        }
                      />
                      {t(`views.map.layer.${id}`)}
                    </label>
                  ))}
                </fieldset>
              </>
            )}
          </Menu>
          {canDraw ? (
            <Menu
              tool
              label={t("draw.menu")}
              icon={<Pencil aria-hidden="true" size={15} />}
              testId="draw-menu"
              active={mode !== "none"}
            >
              {(close) => (
                <>
                  <button
                    type="button"
                    className="menu-item"
                    data-testid="draw-polygon"
                    aria-pressed={mode === "polygon"}
                    onClick={() => {
                      close();
                      startDraw("polygon");
                    }}
                  >
                    <Pentagon aria-hidden="true" size={15} />
                    {t("draw.polygon")}
                  </button>
                  <button
                    type="button"
                    className="menu-item"
                    data-testid="draw-rectangle"
                    aria-pressed={mode === "rectangle"}
                    onClick={() => {
                      close();
                      startDraw("rectangle");
                    }}
                  >
                    <Square aria-hidden="true" size={15} />
                    {t("draw.rectangle")}
                  </button>
                  <button
                    type="button"
                    className="menu-item"
                    data-testid="draw-path"
                    aria-pressed={mode === "path"}
                    onClick={() => {
                      close();
                      startDraw("path");
                    }}
                  >
                    <Route aria-hidden="true" size={15} />
                    {t("draw.path")}
                  </button>
                  <button
                    type="button"
                    className="menu-item"
                    data-testid="draw-pin"
                    aria-pressed={mode === "pin"}
                    onClick={() => {
                      close();
                      modeRef.current = mode === "pin" ? "none" : "pin";
                      setMode(modeRef.current);
                    }}
                  >
                    <MapPin aria-hidden="true" size={15} />
                    {t("draw.pin")}
                  </button>
                </>
              )}
            </Menu>
          ) : null}
        </div>
        {shownPath ? (
          <p className="map-path-progress" role="status" data-testid="path-progress">
            {t(
              shownPath.running || shownPath.pct >= 100 ? "draw.pathProgress" : "draw.pathPlanned",
              {
                label: String((shownPath.command.params as { label?: unknown }).label ?? ""),
                pct: Math.round(shownPath.pct),
              },
            )}
          </p>
        ) : null}
        {cursor && !isCompact ? <p className="map-cursor">{cursor}</p> : null}
      </div>
      {mode !== "none" ? (
        <div className="map-draw" role="status" data-testid="draw-bar">
          <span>
            {mode === "pin"
              ? t("draw.pinHint")
              : mode === "rectangle"
                ? t("draw.rectangleHint")
                : mode === "path"
                  ? t("draw.pathHint", {
                      n: points.length,
                      length: formatLength(lengthM(points)),
                    })
                  : t("draw.polygonHint", { n: points.length })}
          </span>
          {mode !== "pin" ? (
            <button
              type="button"
              className="btn btn-small btn-primary"
              data-testid="draw-done"
              disabled={points.length < (mode === "path" ? 2 : 3)}
              onClick={() => void (mode === "path" ? finishPath() : finishArea())}
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
    </div>
  );
}
