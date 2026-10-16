import { useEffect, useRef, useState } from "react";
import { useTranslation } from "react-i18next";
import { exposeMap } from "../../lib/testHooks";
import { useLiveStore } from "../../state/liveStore";
import { useSessionStore } from "../../state/sessionStore";
import { useUiStore } from "../../state/uiStore";
import type { ViewProps } from "../registry";

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

/**
 * The map frame: the offline map pack from sd-ground (PMTiles) with the Protomaps style in
 * the panel's theme. Without a pack or without WebGL2 it shows the text fallbacks instead.
 */
export default function MapView(_: ViewProps) {
  const { t } = useTranslation();
  const mapCfg = useSessionStore((s) => s.config?.map ?? null);
  const theme = useUiStore((s) => s.theme);
  const box = useRef<HTMLDivElement>(null);
  const [webgl] = useState(hasWebGL2);
  const [failed, setFailed] = useState(false);

  useEffect(() => {
    if (!mapCfg || !webgl || !box.current) return;
    let disposed = false;
    let map: { remove(): void } | null = null;
    setFailed(false);
    // A map that cannot start (chunk, worker or WebGL context) fails here alone: the view
    // shows the text fallback and the rest of the panel, STOPP included, is untouched.
    import("./createMap")
      .then(({ createMap }) => {
        if (disposed || !box.current) return;
        map = createMap(box.current, mapCfg, theme, compactAttribution());
        exposeMap(map);
      })
      .catch(() => {
        if (!disposed) setFailed(true);
      });
    return () => {
      disposed = true;
      map?.remove();
    };
  }, [mapCfg, webgl, theme]);

  if (!mapCfg) {
    return (
      <div className="view-body view-empty">
        <p>{t("views.map.noPack")}</p>
        <DroneText />
      </div>
    );
  }
  if (!webgl || failed) {
    return (
      <div className="view-body view-empty">
        <p>{t(webgl ? "views.map.failed" : "views.map.noWebgl")}</p>
        <DroneText />
      </div>
    );
  }
  return <div ref={box} className="map-box" data-hotkeys="map" />;
}
