import { Gamepad2 } from "lucide-react";
import { useTranslation } from "react-i18next";
import { useBrainState } from "../state/brain";
import { useLiveStore } from "../state/liveStore";
import { activeCharter, useMissionStore } from "../state/missionStore";
import { useIsPilot } from "../state/sessionStore";
import { useShellStore } from "../state/shellStore";
import { OrderButtons } from "../views/chat/OrderButtons";

/** The AI is gone: by its own alert, or the ground reports the process down. */
export function useAiGone(): boolean {
  const brain = useBrainState();
  const alert = useLiveStore((s) => s.aiDisconnected);
  return alert || brain === "down";
}

/**
 * The one place that says the AI is gone: what the drone does now (the active charter's
 * on_ai_silent), what still works, the order buttons and manual control. No countdown.
 */
export function CommandPanel() {
  const { t } = useTranslation();
  const isPilot = useIsPilot();
  const activeId = useLiveStore((s) => s.mission?.active_charter_id ?? null);
  const charter = useMissionStore((s) => activeCharter(s, activeId));
  const body = charter?.body;
  return (
    <section
      className="cmd-panel"
      role="status"
      aria-labelledby="cmd-title"
      data-testid="ai-down-banner"
    >
      <header className="cmd-head">
        <span className="tag tag-warn">{t("command.flag")}</span>
        <strong id="cmd-title">{t("aiDown.lead")}</strong>
      </header>
      <dl className="kv">
        {body ? (
          <div>
            <dt>{t("command.droneNow")}</dt>
            <dd data-testid="cmd-behaviour">
              {t(`aiDown.${body.on_ai_silent ?? "continue_plan"}`, {
                s: body.ai_silent_timeout_s ?? "–",
              })}
            </dd>
          </div>
        ) : null}
        <div>
          <dt>{t("command.works")}</dt>
          <dd>{t("aiDown.safety")}</dd>
        </div>
      </dl>
      <OrderButtons />
      {isPilot ? (
        <button
          type="button"
          className="btn btn-quiet btn-block"
          data-testid="cmd-manual"
          onClick={() => useShellStore.getState().setOverlay("manual")}
        >
          <Gamepad2 aria-hidden="true" size={15} /> {t("menu.manual")}
        </button>
      ) : null}
    </section>
  );
}
