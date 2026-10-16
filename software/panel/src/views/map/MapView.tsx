import { useEffect, useRef, useState } from "react";
import { useTranslation } from "react-i18next";
import { exposeMap } from "../../lib/testHooks";
import { useLiveStore } from "../../state/liveStore";
import { useSessionStore } from "../../state/sessionStore";
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

/**
 * The map frame. The offline map pack and the layers come with the map ticket; here MapLibre
 * starts on a plain background when the server offers a map, and the text fallbacks exist.
 */
export default function MapView(_: ViewProps) {
  const { t } = useTranslation();
  const mapCfg = useSessionStore((s) => s.config?.map ?? null);
  const box = useRef<HTMLDivElement>(null);
  const [webgl] = useState(hasWebGL2);

  useEffect(() => {
    if (!mapCfg || !webgl || !box.current) return;
    let disposed = false;
    let map: { remove(): void } | null = null;
    void import("./createMap").then(({ createMap }) => {
      if (disposed || !box.current) return;
      map = createMap(box.current, mapCfg);
      exposeMap(map);
    });
    return () => {
      disposed = true;
      map?.remove();
    };
  }, [mapCfg, webgl]);

  if (!mapCfg) {
    return (
      <div className="view-body view-empty">
        <p>{t("views.map.noPack")}</p>
        <DroneText />
      </div>
    );
  }
  if (!webgl) {
    return (
      <div className="view-body view-empty">
        <p>{t("views.map.noWebgl")}</p>
        <DroneText />
      </div>
    );
  }
  return <div ref={box} className="map-box" data-hotkeys="map" />;
}
