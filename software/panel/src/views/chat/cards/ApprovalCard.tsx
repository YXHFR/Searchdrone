import { useEffect, useId, useRef, useState } from "react";
import { useTranslation } from "react-i18next";
import { defaultAutonomy, type PanelApproval } from "../../../api/approvals";
import { api, HttpError } from "../../../api/http";
import { paths } from "../../../api/paths";
import type { Approval, AutonomyMode } from "../../../api/types";
import { mmss, useNow } from "../../../lib/useNow";
import { ClearanceControl } from "../../../safety/ClearanceControl";
import { useMissionStore } from "../../../state/missionStore";
import { useIsPilot } from "../../../state/sessionStore";
import { useUiStore } from "../../../state/uiStore";
import { CharterBody, ProblemList, useCharterProblems } from "./CharterBody";
import { LimitsButton } from "./LimitsForm";
import { SkillBody } from "./SkillBody";

type Outcome = "superseded" | "expired" | null;

function useDecide(a: PanelApproval) {
  const { t } = useTranslation();
  const [busy, setBusy] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const [outcome, setOutcome] = useState<Outcome>(null);
  async function decide(decision: "approve" | "reject", extra: Record<string, unknown>) {
    if (outcome || busy) return; // no more requests from this card after a 409
    setBusy(true);
    setError(null);
    try {
      // Never `confirmations`: the clearance is its own click on its own endpoint.
      const updated = await api<Approval>(paths.approvalDecision(a.approval_id), {
        method: "POST",
        body: { decision, subject_version: a.subject_version, ...extra },
      });
      useMissionStore.getState().upsertApproval(updated);
    } catch (e) {
      const code = e instanceof HttpError ? e.code : undefined;
      if (code === "SUPERSEDED") setOutcome("superseded");
      else if (code === "EXPIRED") setOutcome("expired");
      else if (code === "ALREADY_DECIDED") {
        setOutcome("superseded");
        await refreshPending(a);
      } else setError(e instanceof HttpError ? e.message : t("cards.failed"));
    } finally {
      setBusy(false);
    }
  }
  return { busy, error, outcome, decide };
}

async function refreshPending(a: PanelApproval): Promise<void> {
  const missionId = useMissionStore.getState().missionId;
  if (!missionId) return;
  try {
    const list = await api<Approval[]>(paths.missionApprovals(missionId, "pending"));
    const m = useMissionStore.getState();
    m.setPendingApprovals(list);
    // Ours is no longer pending: mark it so the buttons go away until approval.updated says how.
    if (!list.some((x) => x.id === a.approval_id)) {
      const mine = m.approvals[a.approval_id];
      if (mine?.status === "pending") m.upsertApproval({ ...mine, status: "superseded" });
    }
  } catch {
    // approval.updated brings the status anyway
  }
}

function DecidedLine({ a, outcome }: { a: PanelApproval; outcome: Outcome }) {
  const { t } = useTranslation();
  const time = a.decided_at
    ? new Intl.DateTimeFormat("sv-SE", { hour: "2-digit", minute: "2-digit" }).format(
        new Date(a.decided_at),
      )
    : "";
  const status = outcome ?? a.status;
  const text =
    status === "approved"
      ? t("cards.approvedBy", { name: (a.decided_by ?? "").split(":").pop(), time })
      : status === "rejected"
        ? t("cards.rejected")
        : status === "expired"
          ? t("cards.expired")
          : status === "superseded"
            ? t("cards.superseded")
            : status === "gate_rejected"
              ? t("cards.gateRejected", {
                  reason: (a.gate_reason_codes ?? [])
                    .map((c) => t(`reasons.${c}`, { defaultValue: c }))
                    .join(", "),
                })
              : null;
  return text ? (
    <p className="card-decided" data-testid="card-decided">
      {text}
    </p>
  ) : null;
}

/** Shell shared by every approval card: title, who and why, body, Godkänn / Avvisa. */
export function ApprovalCard({ approval: a }: { approval: PanelApproval }) {
  const { t } = useTranslation();
  const titleId = useId();
  const ref = useRef<HTMLFieldSetElement>(null);
  const isPilot = useIsPilot();
  const focus = useUiStore((s) => s.focusApproval);
  const clearanceActive = useMissionStore((s) => s.clearance.status === "active");
  const { busy, error, outcome, decide } = useDecide(a);
  const [mode, setMode] = useState<AutonomyMode>(() => defaultAutonomy(a));
  const [rejecting, setRejecting] = useState(false);
  const [reason, setReason] = useState("");
  const now = useNow(1_000);
  const isCharter = a.kind === "charter" || a.kind === "charter_change";
  const problems = useCharterProblems(isCharter ? a.charter?.body : undefined);
  const needsClearance =
    a.kind === "skill" && a.required_confirmations.includes("clear_around_drone");
  const pending = a.status === "pending" && outcome === null;

  useEffect(() => {
    if (focus === a.approval_id) {
      ref.current?.scrollIntoView?.({ block: "center" });
      ref.current?.focus();
      useUiStore.getState().showApproval(null);
    }
  }, [focus, a.approval_id]);

  const left = a.expires_at ? Date.parse(a.expires_at) - now : null;
  const title =
    a.kind === "skill"
      ? t("cards.skill.title", {
          skill: t(`skills.${a.command?.skill ?? ""}`, { defaultValue: a.command?.skill ?? "" }),
        })
      : t(`cards.${a.kind}.title`, { v: a.subject_version });
  const approveDisabled =
    !isPilot ||
    busy ||
    a.subject_loading ||
    problems.length > 0 ||
    (needsClearance && !clearanceActive);
  return (
    <fieldset
      ref={ref}
      className="approval"
      aria-labelledby={titleId}
      tabIndex={-1}
      data-testid={`approval-${a.approval_id}`}
      data-kind={a.kind}
    >
      {pending ? <p className="approval-need">{t("cards.needsApproval")}</p> : null}
      <h3 id={titleId}>{title}</h3>
      <p className="muted">
        {t(a.requested_by.startsWith("brain") ? "cards.byAi" : "cards.byOperator")} ·{" "}
        {new Intl.DateTimeFormat("sv-SE", { hour: "2-digit", minute: "2-digit" }).format(
          new Date(a.created_at),
        )}
        {pending && left !== null ? ` · ${t("cards.expiresIn", { left: mmss(left) })}` : ""}
      </p>
      <p>{a.summary_sv}</p>
      {a.rationale ? <p className="muted">{a.rationale}</p> : null}
      {a.subject_loading ? (
        <p className="muted">{t("common.loading")}</p>
      ) : isCharter ? (
        <CharterBody approval={a} />
      ) : (
        <SkillBody approval={a} />
      )}
      {pending && isCharter ? <ProblemList problems={problems} /> : null}
      {pending && isCharter ? (
        <fieldset className="mode-choice">
          <legend>{t("cards.mode.legend")}</legend>
          {(["APPROVE_EACH", "WITHIN_CHARTER"] as const).map((m) => (
            <label key={m}>
              <input
                type="radio"
                name={`mode-${a.approval_id}`}
                value={m}
                checked={mode === m}
                disabled={!isPilot}
                onChange={() => setMode(m)}
              />
              {t(m === "APPROVE_EACH" ? "cards.mode.a" : "cards.mode.b")}
            </label>
          ))}
        </fieldset>
      ) : null}
      {pending && needsClearance ? <ClearanceControl force /> : null}
      {pending ? (
        <div className="approval-actions">
          <button
            type="button"
            className="btn btn-primary"
            data-testid="approve"
            disabled={approveDisabled}
            onClick={() => void decide("approve", isCharter ? { autonomy_mode: mode } : {})}
          >
            {t("cards.approve")}
          </button>
          <button
            type="button"
            className="btn btn-outline"
            data-testid="reject"
            disabled={!isPilot || busy}
            onClick={() =>
              rejecting ? void decide("reject", reason ? { reason } : {}) : setRejecting(true)
            }
          >
            {t(rejecting ? "cards.rejectSend" : "cards.reject")}
          </button>
          {isCharter && isPilot ? (
            <button
              type="button"
              className="btn btn-outline"
              data-testid="edit-charter"
              onClick={() =>
                useUiStore.getState().setDraft({ text: t("cards.editPrefix"), replyTo: null }, true)
              }
            >
              {t("cards.edit")}
            </button>
          ) : null}
          {isCharter && isPilot ? <LimitsButton /> : null}
          {rejecting ? (
            <label className="reject-reason">
              {t("cards.why")}
              <input
                className="input"
                value={reason}
                maxLength={500}
                onChange={(e) => setReason(e.target.value)}
              />
            </label>
          ) : null}
        </div>
      ) : (
        <DecidedLine a={a} outcome={outcome} />
      )}
      {error ? <p className="error">{error}</p> : null}
    </fieldset>
  );
}
