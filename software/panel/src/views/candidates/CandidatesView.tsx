// Granska: the candidate queue in three tabs, or one candidate open for a decision.
// J/K and the arrow keys move between cards, Enter opens one. No key confirms anything.
import { type KeyboardEvent, useEffect, useMemo, useRef, useState } from "react";
import { useTranslation } from "react-i18next";
import { useNow } from "../../lib/useNow";
import { useCandidateStore } from "../../state/candidateStore";
import { useLiveStore } from "../../state/liveStore";
import type { ViewProps } from "../registry";
import { CandidateCard } from "./CandidateCard";
import { BACKLOG_AT, findNumbers, groupQueue, type Tab } from "./queue";
import { ReviewCandidate } from "./ReviewCandidate";

const TABS: Tab[] = ["review", "confirmed", "rejected"];
const BLINK_MS = 1_200;

export default function CandidatesView(_: ViewProps) {
  const { t } = useTranslation();
  const missionId = useLiveStore((s) => s.mission?.id ?? null);
  const cards = useCandidateStore((s) => s.cards);
  const fresh = useCandidateStore((s) => s.fresh);
  const openId = useCandidateStore((s) => s.openId);
  const now = useNow(30_000);
  const [tab, setTab] = useState<Tab>("review");
  const [lowOpen, setLowOpen] = useState(false);
  const [selected, setSelected] = useState(0);
  const boxRef = useRef<HTMLDivElement>(null);
  const all = useMemo(() => Object.values(cards), [cards]);
  const q = useMemo(() => groupQueue(all), [all]);
  const finds = useMemo(() => findNumbers(all), [all]);
  const shown = tab === "review" ? [...q.review, ...(lowOpen ? q.low : [])] : q[tab];

  // A new card blinks once, then it is an ordinary card.
  useEffect(() => {
    const ids = Object.keys(fresh);
    if (ids.length === 0) return;
    const timer = setTimeout(() => {
      for (const id of ids) useCandidateStore.getState().seen(id);
    }, BLINK_MS);
    return () => clearTimeout(timer);
  }, [fresh]);

  if (!missionId) {
    return (
      <div className="view-body view-empty">
        <p className="muted">{t("candidates.empty.review")}</p>
      </div>
    );
  }
  const open = openId ? cards[openId] : undefined;
  if (open) return <ReviewCandidate card={open} findNo={finds.get(open.candidate_id)} />;

  const move = (d: number): void => {
    if (shown.length === 0) return;
    const next = Math.min(shown.length - 1, Math.max(0, selected + d));
    setSelected(next);
    boxRef.current?.querySelectorAll<HTMLButtonElement>(".cand-open")[next]?.focus();
  };
  const onKey = (e: KeyboardEvent): void => {
    if (e.altKey || e.ctrlKey || e.metaKey) return;
    if (e.key === "j" || e.key === "J" || e.key === "ArrowDown") {
      e.preventDefault();
      move(1);
    } else if (e.key === "k" || e.key === "K" || e.key === "ArrowUp") {
      e.preventDefault();
      move(-1);
    }
  };

  return (
    // biome-ignore lint/a11y/noStaticElementInteractions: the list's own J/K keys; each card is a button
    <div className="view-body cand-view" ref={boxRef} onKeyDown={onKey}>
      <h2 className="sr-only">{t("candidates.label")}</h2>
      <p className="muted cand-only">{t("views.candidates.onlyYou")}</p>
      {q.review.length > BACKLOG_AT ? (
        <p className="warn" data-testid="cand-backlog">
          {t("candidates.backlog")}
        </p>
      ) : null}
      <div className="seg cand-tabs" role="tablist" aria-label={t("candidates.label")}>
        {TABS.map((id) => (
          <button
            key={id}
            type="button"
            role="tab"
            aria-selected={tab === id}
            data-testid={`cand-tab-${id}`}
            onClick={() => {
              setTab(id);
              setSelected(0);
            }}
          >
            {t(`candidates.tabs.${id}`)} ({id === "review" ? q.review.length : q[id].length})
          </button>
        ))}
      </div>
      {tab === "review" && q.review.length + q.low.length > 0 ? (
        <p className="muted cand-keys">{t("candidates.keys")}</p>
      ) : null}
      {shown.length === 0 && !(tab === "review" && q.low.length > 0) ? (
        <p className="muted" data-testid="cand-empty">
          {t(`candidates.empty.${tab}`)}
        </p>
      ) : null}
      <ul className="cand-list" data-testid="cand-list">
        {(tab === "review" ? q.review : shown).map((c, i) => (
          <CandidateCard
            key={c.candidate_id}
            card={c}
            missionId={missionId}
            now={now}
            selected={i === selected}
            fresh={Boolean(fresh[c.candidate_id])}
            findNo={finds.get(c.candidate_id)}
            onSelect={() => setSelected(i)}
            onOpen={() => useCandidateStore.getState().open(c.candidate_id)}
          />
        ))}
      </ul>
      {tab === "review" && q.low.length > 0 ? (
        <section className="cand-low">
          <button
            type="button"
            className="btn btn-quiet btn-block"
            aria-expanded={lowOpen}
            data-testid="cand-low-toggle"
            onClick={() => setLowOpen((v) => !v)}
          >
            {t("candidates.low", { n: q.low.length })}
          </button>
          {lowOpen ? (
            <ul className="cand-list" data-testid="cand-low-list">
              {q.low.map((c, i) => (
                <CandidateCard
                  key={c.candidate_id}
                  card={c}
                  missionId={missionId}
                  now={now}
                  selected={q.review.length + i === selected}
                  fresh={Boolean(fresh[c.candidate_id])}
                  onSelect={() => setSelected(q.review.length + i)}
                  onOpen={() => useCandidateStore.getState().open(c.candidate_id)}
                />
              ))}
            </ul>
          ) : null}
        </section>
      ) : null}
    </div>
  );
}
