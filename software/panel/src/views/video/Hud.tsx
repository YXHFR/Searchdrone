import { useTranslation } from "react-i18next";
import { distance, heading as headingLabel, isStale } from "../../header/instruments";
import { useNow } from "../../lib/useNow";
import { useLiveStore } from "../../state/liveStore";
import { horizon, tapeTicks } from "./hud";

const TAPE_SPAN = 35;
const TAPE_PX = 5;
const PITCH_PX = 4;
const LADDER = [-20, -10, 10, 20];
const one = new Intl.NumberFormat("sv-SE", { minimumFractionDigits: 1, maximumFractionDigits: 1 });
const clock = new Intl.DateTimeFormat("sv-SE", {
  hour: "2-digit",
  minute: "2-digit",
  second: "2-digit",
});

function HeadingTape({ heading }: { heading: number }) {
  const w = TAPE_SPAN * 2 * TAPE_PX;
  return (
    <div className="hud-tape">
      <svg viewBox={`${-w / 2} 0 ${w} 30`} width={w} height={30} aria-hidden="true">
        {tapeTicks(heading, TAPE_SPAN).map((t) => (
          <g key={`${t.deg}-${t.offset}`} transform={`translate(${t.offset * TAPE_PX} 0)`}>
            <line y1={t.label ? 18 : t.major ? 21 : 24} y2={30} />
            {t.label ? (
              <text y={13} textAnchor="middle">
                {t.label}
              </text>
            ) : null}
          </g>
        ))}
      </svg>
      <b className="hud-heading num">{headingLabel(heading)}</b>
    </div>
  );
}

/** Artificial horizon with a pitch ladder, turned by the drone's own roll. */
function Horizon({ roll, pitch }: { roll: number; pitch: number }) {
  return (
    <svg className="hud-horizon" viewBox="-120 -90 240 180" aria-hidden="true">
      <defs>
        <clipPath id="hud-clip">
          <rect x="-120" y="-90" width="240" height="180" />
        </clipPath>
      </defs>
      <g clipPath="url(#hud-clip)">
        <g transform={`rotate(${-roll}) translate(0 ${pitch * PITCH_PX})`}>
          <line x1="-400" x2="-28" y1="0" y2="0" className="hud-hz" />
          <line x1="28" x2="400" y1="0" y2="0" className="hud-hz" />
          {LADDER.map((p) => (
            <g
              key={p}
              transform={`translate(0 ${-p * PITCH_PX})`}
              className={p < 0 ? "hud-below" : undefined}
            >
              <line x1="-34" x2="-14" />
              <line x1="14" x2="34" />
              <text x="40" y="3.5">
                {Math.abs(p)}
              </text>
            </g>
          ))}
        </g>
      </g>
      <path d="M-22 0H-8L0 7L8 0H22" className="hud-craft" />
    </svg>
  );
}

/** Cockpit HUD over the large video. Every value is the drone's own; old data turns grey. */
export function Hud() {
  const { t } = useTranslation();
  const now = useNow(250);
  const tel = useLiveStore((s) => s.telemetry);
  const at = useLiveStore((s) => s.lastTelemetryAt);
  const age = at === null ? null : now - at;
  const stale = isStale(age);
  const hz = horizon(tel, age);
  return (
    <div className="hud" data-testid="hud" data-stale={stale}>
      {tel ? <HeadingTape heading={tel.heading_deg} /> : null}
      <div className="hud-centre">
        {hz ? (
          <Horizon roll={hz.roll} pitch={hz.pitch} />
        ) : (
          <span className="hud-note" data-testid="hud-no-attitude">
            {t("hud.noAttitude")}
          </span>
        )}
      </div>
      <div className="hud-box hud-left">
        <span>{t("strip.speed")}</span>
        <b className="num">{tel ? one.format(tel.groundspeed_ms) : "–"}</b>
        <small>m/s</small>
      </div>
      <div className="hud-box hud-right">
        <span>{t("strip.alt")}</span>
        <b className="num">{tel ? Math.round(tel.alt_rel_home_m) : "–"}</b>
        <small>m</small>
      </div>
      <div className="hud-corner hud-tr num">
        {clock.format(now)}
        {stale && tel && age !== null ? (
          <span className="hud-age" data-testid="hud-age">
            {t("hud.age", { s: Math.floor(age / 1000) })}
          </span>
        ) : null}
      </div>
      <div className="hud-corner hud-br num">
        {t("strip.battery")} {tel?.battery_pct != null ? `${Math.round(tel.battery_pct)} %` : "–"}
        {tel?.dist_to_pilot_m != null
          ? ` · ${t("strip.fromYou")} ${distance(tel.dist_to_pilot_m)}`
          : ""}
      </div>
    </div>
  );
}
