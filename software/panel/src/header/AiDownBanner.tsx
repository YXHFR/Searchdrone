import { useTranslation } from "react-i18next";
import { useLiveStore } from "../state/liveStore";
import { activeCharter, useMissionStore } from "../state/missionStore";

/**
 * Under the top bar in every view while the AI is gone (alert BRAIN_DISCONNECTED). Says what
 * the drone does by the active charter's on_ai_silent. No countdown: the panel starts nothing.
 */
export function AiDownBanner() {
  const { t } = useTranslation();
  const down = useLiveStore((s) => s.aiDisconnected);
  const activeId = useLiveStore((s) => s.mission?.active_charter_id ?? null);
  const charter = useMissionStore((s) => activeCharter(s, activeId));
  if (!down) return null;
  const body = charter?.body;
  const behaviour = body
    ? t(`aiDown.${body.on_ai_silent ?? "continue_plan"}`, { s: body.ai_silent_timeout_s ?? "–" })
    : "";
  return (
    <div className="banner banner-warn" role="status" data-testid="ai-down-banner">
      {[t("aiDown.lead"), behaviour, t("aiDown.safety")].filter(Boolean).join(" ")}
    </div>
  );
}
