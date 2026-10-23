// One card in the queue: K-number, thumbnail, priority, what the AI thought in words, the
// fields, which engine assessed it, how often it was seen, its age and what is near it.
import { useTranslation } from "react-i18next";
import { paths } from "../../api/paths";
import type { CandidateCard as Card } from "../../api/types";
import { ageMinutes, fieldLabel } from "./queue";

interface Props {
  card: Card;
  missionId: string;
  now: number;
  selected: boolean;
  fresh: boolean;
  findNo?: number | undefined;
  onOpen(): void;
  onSelect(): void;
}

export function CandidateCard({
  card,
  missionId,
  now,
  selected,
  fresh,
  findNo,
  onOpen,
  onSelect,
}: Props) {
  const { t } = useTranslation();
  const age = ageMinutes(card, now);
  const flagged = (card.flags ?? []).length > 0;
  const profile = card.assessed_profile_version;
  return (
    <li
      className="cand-card"
      data-testid="candidate-card"
      data-seq={card.seq}
      data-selected={selected}
      data-fresh={fresh}
    >
      <button
        type="button"
        className="cand-open"
        aria-current={selected}
        onClick={onOpen}
        onFocus={onSelect}
      >
        {card.thumb_detection_id ? (
          <img
            className="cand-thumb"
            src={paths.detectionCrop(missionId, card.thumb_detection_id)}
            alt=""
            width={72}
            height={72}
            loading="lazy"
          />
        ) : (
          <span className="cand-thumb cand-nothumb">{t("candidates.noThumb")}</span>
        )}
        <span className="cand-main">
          <span className="cand-head">
            <strong className="cand-k">
              {findNo ? `F${findNo} · ` : ""}K{card.seq}
            </strong>
            <span className={`cand-prio cand-prio-${card.priority_class}`}>
              {t(`candidates.prio.${card.priority_class}`)}
            </span>
            {card.engine ? (
              <span className="tag" data-testid="engine-badge">
                {t(`candidates.engine.${card.engine}`)}
              </span>
            ) : null}
          </span>
          {card.person ? (
            <span className="cand-line">{t(`candidates.person.${card.person}`)}</span>
          ) : null}
          {Object.keys(card.fields ?? {}).length > 0 ? (
            <span className="cand-chips">
              {Object.entries(card.fields ?? {}).map(([key, v]) => {
                const f = fieldLabel(key);
                return (
                  <span key={key} className={`chip chip-${v}`}>
                    {t(`candidates.field.${f.key}`, { n: f.n })}: {t(`candidates.fieldValue.${v}`)}
                  </span>
                );
              })}
            </span>
          ) : null}
          {flagged ? (
            <span className="cand-line warn" data-testid="card-flag">
              {t("candidates.flagNote")}
            </span>
          ) : null}
          <span className="cand-line muted">
            {t("candidates.seen", { n: card.n_seen, m: card.n_in_view })} ·{" "}
            {age === 0 ? t("candidates.ageNow") : t("candidates.age", { n: age })}
            {card.near_seq && card.near_state
              ? ` · ${t(`candidates.near.${card.near_state}`, { seq: card.near_seq })}`
              : ""}
          </span>
          {profile ? (
            <span className="cand-line muted">
              {card.current_profile_version && card.current_profile_version !== profile
                ? t("candidates.profileOld", { v: profile, now: card.current_profile_version })
                : t("candidates.profile", { v: profile })}
            </span>
          ) : null}
        </span>
      </button>
    </li>
  );
}
