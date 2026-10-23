import { useTranslation } from "react-i18next";
import { STALE_MS } from "../header/pills";
import { useNow } from "../lib/useNow";
import { useCandidateStore } from "../state/candidateStore";
import { useLiveStore } from "../state/liveStore";
import { useMissionStore } from "../state/missionStore";
import { useShellStore } from "../state/shellStore";
import { findNumbers, groupQueue } from "../views/candidates/queue";
import { coverage, doneCount, sectorRows } from "./mission";

const hm = new Intl.DateTimeFormat("sv-SE", { hour: "2-digit", minute: "2-digit" });
const ha = new Intl.NumberFormat("sv-SE", { maximumFractionDigits: 1, minimumFractionDigits: 1 });

function elapsed(now: number, iso: string): string {
  const m = Math.max(0, Math.floor((now - Date.parse(iso)) / 60_000));
  return m < 60 ? `${m} min` : `${Math.floor(m / 60)} h ${m % 60} min`;
}

/** What the mission is, how far it has come and what is left: the left column's top. */
export function MissionPanel() {
  const { t } = useTranslation();
  const now = useNow(5_000);
  const mission = useLiveStore((s) => s.mission);
  const away = useLiveStore((s) => s.telemetry?.dist_to_pilot_m ?? null);
  const at = useLiveStore((s) => s.lastTelemetryAt);
  const sectors = useMissionStore((s) => s.sectors);
  const rows = sectorRows(sectors);
  const cov = coverage(sectors);
  const stale = at === null || now - at > STALE_MS;
  const started = mission?.activated_at ?? null;
  return (
    <section className="lc-sec" data-testid="mission-panel">
      <header className="lc-head">
        <h2 id="lc-mission">{t("mission.title")}</h2>
        {mission ? (
          <span className="lc-state" data-status={mission.status}>
            {t(`mission.state.${mission.status}`, { defaultValue: mission.status })}
          </span>
        ) : null}
      </header>
      {mission && started ? (
        <>
          <dl className="kv">
            <div>
              <dt>{t("mission.started")}</dt>
              <dd className="num">
                {hm.format(new Date(started))} · {elapsed(now, started)}
              </dd>
            </div>
            <div>
              <dt>{t("mission.mode")}</dt>
              <dd>
                {mission.autonomy_mode
                  ? t(
                      mission.autonomy_mode === "WITHIN_CHARTER"
                        ? "mission.modeB"
                        : "mission.modeA",
                    )
                  : "–"}
              </dd>
            </div>
          </dl>
          <div className="readouts" data-stale={stale}>
            <div>
              <span className="lbl">{t("strip.timeLeft")}</span>
              <b className="num">–</b>
              <small>{t("mission.timeLeftNone")}</small>
            </div>
            <div>
              <span className="lbl">{t("mission.toYou")}</span>
              <b className="num">{away === null ? "–" : `${Math.round(away)} m`}</b>
              <small>{t("mission.toYouHint")}</small>
            </div>
          </div>
        </>
      ) : (
        <p className="lc-empty">{t(mission ? "mission.notStarted" : "mission.none")}</p>
      )}
      <header className="lc-sub">
        <span className="lbl">{t("mission.sectors")}</span>
        {rows.length > 0 ? (
          <span className="muted">
            {t("mission.sectorsDone", { n: doneCount(rows), of: rows.length })}
          </span>
        ) : null}
      </header>
      {rows.length === 0 ? (
        <p className="lc-empty">{t("mission.noSectors")}</p>
      ) : (
        <ol className="sectors" data-testid="sectors">
          {rows.map((r) => (
            <li key={r.id} data-status={r.status} data-testid={`sector-${r.id}`}>
              <span className="sec-name">{r.name}</span>
              <span className="sec-bar" aria-hidden="true">
                <i style={{ width: `${r.pct}%` }} />
              </span>
              <span className="sec-pct num">{r.pct} %</span>
              <span className="sec-st">{t(`mission.sector.${r.status}`)}</span>
            </li>
          ))}
        </ol>
      )}
      {cov ? (
        <p className="lc-foot" data-testid="coverage">
          {t("mission.sweptLead")} <b className="num">{ha.format(cov.sweptHa)} ha</b>{" "}
          {t("mission.sweptOf", { total: ha.format(cov.totalHa) })}
        </p>
      ) : null}
    </section>
  );
}

/** Finds arrive with the candidate queue; until then the column says so plainly. */
export function FindsPanel() {
  const { t } = useTranslation();
  const cards = useCandidateStore((s) => s.cards);
  const all = Object.values(cards);
  const q = groupQueue(all);
  const finds = findNumbers(all);
  return (
    <section className="lc-sec" data-testid="finds-panel">
      <header className="lc-head">
        <h2 id="lc-finds">{t("finds.title")}</h2>
      </header>
      {q.confirmed.length === 0 ? <p className="lc-empty">{t("finds.none")}</p> : null}
      {q.confirmed.map((c) => (
        <p key={c.candidate_id} className="lc-find" data-testid="finds-confirmed">
          {t("finds.confirmed", { n: finds.get(c.candidate_id), seq: c.seq })}
        </p>
      ))}
      <p className="muted">{t("finds.toReview", { n: q.review.length })}</p>
      <button
        type="button"
        className="btn btn-quiet btn-block"
        data-testid="finds-open"
        onClick={() => useShellStore.getState().setMode("review")}
      >
        {t("finds.open")}
      </button>
    </section>
  );
}
