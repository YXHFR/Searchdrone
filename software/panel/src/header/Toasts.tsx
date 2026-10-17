import { X } from "lucide-react";
import { useTranslation } from "react-i18next";
import { toPanelApproval } from "../api/approvals";
import { useLayoutStore } from "../state/layoutStore";
import { useMissionStore } from "../state/missionStore";
import { useUiStore } from "../state/uiStore";

/** Short notices above the safety bar. A new approval card gets "Visa", which brings the chat up. */
export function Toasts() {
  const { t } = useTranslation();
  const toasts = useUiStore((s) => s.toasts);
  const approvals = useMissionStore((s) => s.approvals);
  const charters = useMissionStore((s) => s.charters);
  const commands = useMissionStore((s) => s.commands);
  const announce = useUiStore((s) => s.approvalAnnounce);
  const titleOf = (id: string): string => {
    const a = approvals[id];
    const card = a ? toPanelApproval(a, { charters, commands }) : null;
    if (!card) return "";
    return card.kind === "skill"
      ? t("cards.skill.title", {
          skill: t(`skills.${card.command?.skill ?? ""}`, {
            defaultValue: card.command?.skill ?? "",
          }),
        })
      : t(`cards.${card.kind}.title`, { v: card.subject_version });
  };
  return (
    <>
      <div className="sr-only" aria-live="assertive">
        {announce ? t("cards.announce", { title: titleOf(announce) }) : ""}
      </div>
      <ul className="toasts" data-testid="toasts">
        {toasts.map((x) => (
          <li key={x.id} className="toast" role="status">
            <span>
              {x.approvalId ? t("cards.toast", { title: titleOf(x.approvalId) }) : x.text}
            </span>
            {x.approvalId ? (
              <button
                type="button"
                className="btn btn-small btn-primary"
                data-testid="toast-show"
                onClick={() => {
                  useLayoutStore.getState().showView("chat");
                  useUiStore.getState().showApproval(x.approvalId ?? null);
                  useUiStore.getState().dismissToast(x.id);
                }}
              >
                {t("cards.show")}
              </button>
            ) : null}
            <button
              type="button"
              className="icon-btn"
              aria-label={t("toast.close")}
              onClick={() => useUiStore.getState().dismissToast(x.id)}
            >
              <X aria-hidden="true" size={16} />
            </button>
          </li>
        ))}
      </ul>
    </>
  );
}
