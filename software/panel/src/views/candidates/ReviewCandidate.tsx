// One candidate open for a decision. The pictures come first, oldest first; the AI's
// assessment stays folded and Bekräfta stays grey until this session has opened a picture in
// large size (the server checks the same thing). Confirming is a hold, then a question.
import { ArrowLeft, Copy, Download } from "lucide-react";
import { useCallback, useEffect, useState } from "react";
import { useTranslation } from "react-i18next";
import { api, HttpError } from "../../api/http";
import { paths } from "../../api/paths";
import type {
  CandidateCard,
  CandidateDecision,
  CandidateDetail,
  CandidateView,
  StoredAssessment,
} from "../../api/types";
import { decimalText, dmsText, swerefText } from "../../lib/coords";
import { useCandidateStore } from "../../state/candidateStore";
import { useIsPilot, useSessionStore } from "../../state/sessionStore";
import { Dialog } from "../../ui/Dialog";
import { HoldButton } from "../../ui/HoldButton";
import { crops, fieldLabel, isOpen } from "./queue";

const pct = new Intl.NumberFormat("sv-SE", { style: "percent", maximumFractionDigits: 0 });
const hhmm = new Intl.DateTimeFormat("sv-SE", { hour: "2-digit", minute: "2-digit" });
const NOTE_MAX = 300;

function latest(list: StoredAssessment[]): StoredAssessment | null {
  return [...list].sort((a, b) => Date.parse(b.created_at) - Date.parse(a.created_at))[0] ?? null;
}

export function ReviewCandidate({
  card,
  findNo,
}: {
  card: CandidateCard;
  findNo?: number | undefined;
}) {
  const { t } = useTranslation();
  const isPilot = useIsPilot();
  const viewed = useCandidateStore((s) => Boolean(s.viewed[card.candidate_id]));
  const [detail, setDetail] = useState<CandidateDetail | null>(null);
  const [failed, setFailed] = useState(false);
  const [large, setLarge] = useState<{ viewId: string; time: string } | null>(null);
  const [aiOpen, setAiOpen] = useState(false);
  const [note, setNote] = useState("");
  const [asking, setAsking] = useState<"confirm" | "undo" | null>(null);
  const [busy, setBusy] = useState(false);
  const [msg, setMsg] = useState<string | null>(null);
  const back = (): void => useCandidateStore.getState().open(null);

  const load = useCallback(async (): Promise<void> => {
    try {
      const d = await api<CandidateDetail>(paths.candidate(card.candidate_id));
      setDetail(d);
      setFailed(false);
      useCandidateStore.getState().upsert(d.card);
    } catch {
      setFailed(true);
    }
  }, [card.candidate_id]);

  // Again whenever the candidate changes underneath (a new picture, an assessment).
  // biome-ignore lint/correctness/useExhaustiveDependencies: card.rev is the trigger
  useEffect(() => {
    void load();
  }, [load, card.rev]);

  const showLarge = async (detectionId: string, time: string): Promise<void> => {
    try {
      const v = await api<CandidateView>(paths.detectionView(detectionId), { method: "POST" });
      useCandidateStore.getState().markViewed(card.candidate_id);
      setLarge({ viewId: v.view_id, time });
    } catch (e) {
      setMsg(t("candidates.review.error", { msg: e instanceof Error ? e.message : "" }));
    }
  };

  const decide = async (decision: CandidateDecision["decision"]): Promise<void> => {
    setBusy(true);
    setMsg(null);
    try {
      const body: CandidateDecision = { decision, note: note.trim(), subject_rev: card.rev };
      const d = await api<CandidateDetail>(paths.candidateDecision(card.candidate_id), {
        method: "POST",
        body,
      });
      useCandidateStore.getState().upsert(d.card);
      setDetail(d);
      setNote("");
      setAsking(null);
      if (decision === "reject" || decision === "unsure") back();
    } catch (e) {
      setAsking(null);
      if (e instanceof HttpError && e.code === "SUPERSEDED") {
        setMsg(t("candidates.review.superseded"));
        await load();
      } else {
        setMsg(t("candidates.review.error", { msg: e instanceof Error ? e.message : "" }));
      }
    } finally {
      setBusy(false);
    }
  };

  const pictures = detail ? crops(detail.observations) : [];
  const ai = detail ? latest(detail.assessments) : null;
  const open = isOpen(card);
  // The server takes confirm and reject only once the search service has made it ready.
  const ready = card.state !== "new";

  return (
    <div className="view-body cand-review" data-testid="cand-review" data-seq={card.seq}>
      <header className="cand-review-head">
        <button type="button" className="btn btn-quiet" data-testid="cand-back" onClick={back}>
          <ArrowLeft aria-hidden="true" size={15} /> {t("candidates.review.back")}
        </button>
        <h2>
          {findNo ? `F${findNo} · ` : ""}K{card.seq}
        </h2>
        <span className="tag">{t(`candidates.state.${card.state}`)}</span>
      </header>
      <p className="muted">{t("views.candidates.onlyYou")}</p>
      {failed ? <p className="warn">{t("candidates.review.failed")}</p> : null}
      {!detail && !failed ? <p className="muted">{t("candidates.review.loading")}</p> : null}

      {card.state === "confirmed" ? (
        <Found card={card} detail={detail} findNo={findNo} onUndo={() => setAsking("undo")} />
      ) : null}

      <section aria-labelledby="cand-pics">
        <h3 id="cand-pics">{t("candidates.review.pictures")}</h3>
        {detail && pictures.length === 0 ? (
          <p className="muted">{t("candidates.review.noPictures")}</p>
        ) : null}
        <ul className="cand-pics">
          {pictures.map((o) => {
            const time = hhmm.format(new Date(o.captured_utc));
            return (
              <li key={o.detection_id}>
                <img
                  src={paths.detectionCrop(detail?.mission_id ?? "", o.detection_id)}
                  alt=""
                  width={128}
                  height={128}
                />
                <span className="muted">{time}</span>
                <button
                  type="button"
                  className="btn btn-small"
                  data-testid="cand-larger"
                  onClick={() => void showLarge(o.detection_id, time)}
                >
                  {t("candidates.review.showLarger")}
                </button>
              </li>
            );
          })}
        </ul>
        <p className="muted">
          {t("candidates.review.where", { sigma: Math.round(card.sigma95_m) })}
        </p>
      </section>

      <section className="cand-ai" aria-labelledby="cand-ai-toggle">
        <button
          id="cand-ai-toggle"
          type="button"
          className="btn btn-quiet"
          aria-expanded={viewed && aiOpen}
          disabled={!viewed}
          data-testid="cand-ai-toggle"
          onClick={() => setAiOpen((v) => !v)}
        >
          {aiOpen && viewed ? t("candidates.review.hideAi") : t("candidates.review.showAi")}
        </button>
        {!viewed ? <p className="muted">{t("candidates.review.aiLocked")}</p> : null}
        {viewed && aiOpen ? <Assessment ai={ai} /> : null}
      </section>

      {open ? (
        <section className="cand-decide" aria-label={t("candidates.label")}>
          <label className="cand-note">
            {t("candidates.review.note")}
            <textarea
              value={note}
              maxLength={NOTE_MAX}
              rows={2}
              data-testid="cand-note"
              onChange={(e) => setNote(e.target.value)}
            />
          </label>
          <div className="cand-buttons">
            <HoldButton
              onDone={() => setAsking("confirm")}
              disabled={!viewed || busy || !ready}
              hint={
                !ready
                  ? t("candidates.review.notReady")
                  : viewed
                    ? t("candidates.review.confirmHint")
                    : t("candidates.review.confirmLocked")
              }
              testId="cand-confirm"
            >
              {t("candidates.review.confirm")}
            </HoldButton>
            <button
              type="button"
              className="btn cand-reject"
              disabled={busy || !ready}
              data-testid="cand-reject"
              onClick={() => void decide("reject")}
            >
              {t("candidates.review.reject")}
            </button>
            <button
              type="button"
              className="btn btn-quiet"
              disabled
              title={t("candidates.review.inspectLater")}
              data-testid="cand-inspect"
            >
              {t("candidates.review.inspect")}
            </button>
            <button
              type="button"
              className="btn btn-quiet"
              disabled={busy}
              data-testid="cand-unsure"
              onClick={() => void decide("unsure")}
            >
              {t("candidates.review.unsure")}
            </button>
          </div>
        </section>
      ) : null}
      {card.state === "rejected" && isPilot ? (
        <button
          type="button"
          className="btn btn-quiet"
          disabled={busy}
          data-testid="cand-reopen"
          onClick={() => void decide("reopen")}
        >
          {t("candidates.found.reopen")}
        </button>
      ) : null}
      {msg ? (
        <p className="warn" role="status" data-testid="cand-msg">
          {msg}
        </p>
      ) : null}

      {large ? (
        <Dialog
          title={t("candidates.review.large", { time: large.time })}
          onClose={() => setLarge(null)}
          testId="cand-large"
        >
          <img className="cand-large-img" src={paths.view(large.viewId)} alt="" />
          <button type="button" className="btn" onClick={() => setLarge(null)}>
            {t("candidates.review.closeLarge")}
          </button>
        </Dialog>
      ) : null}
      {asking === "confirm" ? (
        <Dialog
          title={t("candidates.dialog.title", { seq: card.seq })}
          onClose={() => setAsking(null)}
          testId="cand-confirm-dialog"
        >
          <p>{t("candidates.dialog.body")}</p>
          <div className="dialog-actions">
            <button type="button" className="btn" onClick={() => setAsking(null)}>
              {t("candidates.dialog.cancel")}
            </button>
            <button
              type="button"
              className="btn btn-primary"
              disabled={busy}
              data-testid="cand-confirm-yes"
              onClick={() => void decide("confirm")}
            >
              {t("candidates.dialog.yes")}
            </button>
          </div>
        </Dialog>
      ) : null}
      {asking === "undo" ? (
        <Dialog
          title={t("candidates.found.undoConfirm")}
          onClose={() => setAsking(null)}
          testId="cand-undo-dialog"
        >
          <label className="cand-note">
            {t("candidates.found.undoNote")}
            <textarea
              value={note}
              maxLength={NOTE_MAX}
              rows={2}
              data-testid="cand-undo-note"
              onChange={(e) => setNote(e.target.value)}
            />
          </label>
          <div className="dialog-actions">
            <button type="button" className="btn" onClick={() => setAsking(null)}>
              {t("candidates.found.cancel")}
            </button>
            <button
              type="button"
              className="btn btn-primary"
              disabled={busy || note.trim() === ""}
              data-testid="cand-undo-yes"
              onClick={() => void decide("reopen")}
            >
              {t("candidates.found.undoConfirm")}
            </button>
          </div>
        </Dialog>
      ) : null}
    </div>
  );
}

/** What the AI said, its claims marked as unverified. Percentages only here. */
function Assessment({ ai }: { ai: StoredAssessment | null }) {
  const { t } = useTranslation();
  if (!ai) return <p className="muted">{t("candidates.review.aiNone")}</p>;
  return (
    <div className="cand-ai-body" data-testid="cand-ai-body">
      <p>
        <span className="tag">{t(`candidates.engine.${ai.engine}`)}</span>{" "}
        {t("candidates.review.aiPerson", { p: pct.format(ai.person_p) })} ·{" "}
        {t("candidates.review.aiOverall", { p: pct.format(ai.overall) })} ·{" "}
        {t("candidates.profile", { v: ai.profile_version })}
      </p>
      <p>{ai.explanation_sv}</p>
      {Object.keys(ai.fields ?? {}).length > 0 ? (
        <ul className="cand-chips">
          {Object.entries(ai.fields ?? {}).map(([key, v]) => {
            const f = fieldLabel(key);
            return (
              <li key={key} className={`chip chip-${v}`}>
                {t(`candidates.field.${f.key}`, { n: f.n })}: {t(`candidates.fieldValue.${v}`)}
              </li>
            );
          })}
        </ul>
      ) : null}
      {(ai.flags ?? []).map((f) => (
        <p key={f} className="warn" data-testid="cand-ai-flag">
          {t(`candidates.flag.${f}`)}: {t("candidates.flagNote")}
        </p>
      ))}
    </div>
  );
}

/** The confirmed find: where it is in three forms, to copy or download, and who said so. */
function Found({
  card,
  detail,
  findNo,
  onUndo,
}: {
  card: CandidateCard;
  detail: CandidateDetail | null;
  findNo?: number | undefined;
  onUndo(): void;
}) {
  const { t } = useTranslation();
  const me = useSessionStore((s) => s.session?.issuer_ref ?? null);
  const isPilot = useIsPilot();
  const [copied, setCopied] = useState<boolean | null>(null);
  const rows = [
    { key: "decimal", text: decimalText(card.lat, card.lon) },
    { key: "dms", text: dmsText(card.lat, card.lon) },
    { key: "sweref", text: swerefText(card.lat, card.lon) ?? t("candidates.found.notSweden") },
  ];
  const confirm = [...(detail?.decisions ?? [])]
    .filter((d) => d.decision === "confirm")
    .sort((a, b) => Date.parse(b.created_at) - Date.parse(a.created_at))[0];
  const copy = async (): Promise<void> => {
    const text = rows.map((r) => `${t(`candidates.found.${r.key}`)}: ${r.text}`).join("\n");
    try {
      await navigator.clipboard.writeText(`K${card.seq}\n${text}`);
      setCopied(true);
    } catch {
      setCopied(false);
    }
  };
  return (
    <section className="cand-found" data-testid="cand-found" aria-labelledby="cand-found-title">
      <h3 id="cand-found-title">
        {t("candidates.found.title", { n: findNo ?? 1, seq: card.seq })}
      </h3>
      <dl className="cand-coords">
        {rows.map((r) => (
          <div key={r.key}>
            <dt>{t(`candidates.found.${r.key}`)}</dt>
            <dd data-testid={`cand-coord-${r.key}`}>{r.text}</dd>
          </div>
        ))}
      </dl>
      {confirm ? (
        <p className="muted" data-testid="cand-confirmed-by">
          {confirm.operator_ref === me
            ? t("candidates.found.byYou", { time: hhmm.format(new Date(confirm.created_at)) })
            : t("candidates.found.by", {
                who: confirm.operator_ref.replace(/^operator:/, ""),
                time: hhmm.format(new Date(confirm.created_at)),
              })}
        </p>
      ) : null}
      <div className="cand-buttons">
        <button type="button" className="btn" data-testid="cand-copy" onClick={() => void copy()}>
          <Copy aria-hidden="true" size={15} /> {t("candidates.found.copy")}
        </button>
        <a
          className="btn"
          href={paths.candidateGpx(card.candidate_id)}
          download={`K${card.seq}.gpx`}
          data-testid="cand-gpx"
        >
          <Download aria-hidden="true" size={15} /> {t("candidates.found.gpx")}
        </a>
        {isPilot ? (
          <button type="button" className="btn btn-quiet" data-testid="cand-undo" onClick={onUndo}>
            {t("candidates.found.undo")}
          </button>
        ) : null}
      </div>
      {copied !== null ? (
        <p className="muted" role="status">
          {copied ? t("candidates.found.copied") : t("candidates.found.copyFailed")}
        </p>
      ) : null}
    </section>
  );
}
