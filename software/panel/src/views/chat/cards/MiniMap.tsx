import { useEffect, useRef, useState } from "react";
import { useTranslation } from "react-i18next";
import { fit, type Ring } from "./miniMap";

type LL = { lat: number; lon: number };

export interface MiniMapProps {
  home: LL | null;
  /** The area approved today, drawn plain. */
  current: Ring | null;
  /** The proposal, drawn dashed in the accent colour. */
  proposed: Ring | null;
  proposedSectors: Ring[];
  /** Area already swept by the camera. */
  swept: Ring[];
  drone: LL | null;
  rangeM: number | null;
}

const H = 132;
const PAD = 18;

/** A static drawing for a card: where the proposal is, how big, and how far from home. */
export function MiniMap(p: MiniMapProps) {
  const { t } = useTranslation();
  const box = useRef<HTMLDivElement>(null);
  const [w, setW] = useState(320);
  useEffect(() => {
    const el = box.current;
    if (!el || typeof ResizeObserver === "undefined") return;
    const ro = new ResizeObserver(() => setW(Math.max(200, Math.round(el.clientWidth))));
    ro.observe(el);
    return () => ro.disconnect();
  }, []);
  const pts: number[][] = [
    ...(p.current ?? []),
    ...(p.proposed ?? []),
    ...p.proposedSectors.flat(),
    ...(p.home ? [[p.home.lon, p.home.lat]] : []),
    ...(p.drone ? [[p.drone.lon, p.drone.lat]] : []),
  ];
  const f = fit(pts, w, H, PAD);
  const d = (r: Ring): string =>
    f
      ? `M${r.map((q) => `${f.x(q[0] ?? 0).toFixed(1)} ${f.y(q[1] ?? 0).toFixed(1)}`).join("L")}Z`
      : "";
  return (
    <figure className="minimap" ref={box}>
      {f ? (
        <svg viewBox={`0 0 ${w} ${H}`} role="img" aria-label={t("minimap.label")}>
          <defs>
            <pattern id="mm-grid" width="24" height="24" patternUnits="userSpaceOnUse">
              <path d="M24 0H0V24" className="mm-grid" />
            </pattern>
          </defs>
          <rect width={w} height={H} className="mm-bg" />
          <rect width={w} height={H} fill="url(#mm-grid)" />
          {p.home && p.rangeM ? (
            <circle
              cx={f.x(p.home.lon)}
              cy={f.y(p.home.lat)}
              r={p.rangeM / f.mpp}
              className="mm-range"
            />
          ) : null}
          {p.current ? <path d={d(p.current)} className="mm-current" /> : null}
          {p.swept.map((r, i) => (
            // biome-ignore lint/suspicious/noArrayIndexKey: a static drawing
            <path key={i} d={d(r)} className="mm-swept" />
          ))}
          {p.proposed ? <path d={d(p.proposed)} className="mm-proposed" /> : null}
          {p.proposedSectors.map((r, i) => (
            // biome-ignore lint/suspicious/noArrayIndexKey: a static drawing
            <path key={i} d={d(r)} className="mm-proposed-sector" />
          ))}
          {p.drone ? (
            <circle cx={f.x(p.drone.lon)} cy={f.y(p.drone.lat)} r="4" className="mm-drone" />
          ) : null}
          {p.home ? (
            <g transform={`translate(${f.x(p.home.lon)} ${f.y(p.home.lat)})`} className="mm-home">
              <rect x="-6" y="-6" width="12" height="12" rx="2" />
              <text y="3.5" textAnchor="middle">
                H
              </text>
            </g>
          ) : null}
          <g transform={`translate(${w - 10 - f.scaleM / f.mpp} ${H - 10})`} className="mm-scale">
            <path d={`M0 -4V0H${f.scaleM / f.mpp}V-4`} />
            <text x={f.scaleM / f.mpp / 2} y="-6" textAnchor="middle">
              {f.scaleM >= 1000 ? `${f.scaleM / 1000} km` : `${f.scaleM} m`}
            </text>
          </g>
          <text x="10" y="16" className="mm-north">
            N ↑
          </text>
        </svg>
      ) : null}
      <figcaption className="mm-key">
        <span className="k k-prop" />
        {t("minimap.proposed")}
        <span className="k k-swept" />
        {t("minimap.swept")}
        {p.rangeM ? (
          <>
            <span className="k k-range" />
            {t("minimap.range")}
          </>
        ) : null}
      </figcaption>
    </figure>
  );
}
