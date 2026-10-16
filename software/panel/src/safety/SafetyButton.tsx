import { House, OctagonX, X } from "lucide-react";
import { useTranslation } from "react-i18next";
import { useVehicleAction } from "./useVehicleAction";
import type { ActionText } from "./vehicleAction";

export function actionText(t: (k: string) => string, text: ActionText | null): string | null {
  if (!text) return null;
  return text.key === "detail" ? text.detail : t(text.key);
}

interface Props {
  action: "stop" | "return_home";
  vehicleId: string | null;
  /** Shown instead of the button action; the button stays visible. */
  disabledReason?: string | null;
}

export function SafetyButton({ action, vehicleId, disabledReason = null }: Props) {
  const { t } = useTranslation();
  const a = useVehicleAction(vehicleId, action);
  const isStop = action === "stop";
  const Icon = isStop ? OctagonX : House;
  const status = actionText(t, a.text);
  const sub = disabledReason ?? status ?? t(isStop ? "safety.stop.sub" : "safety.home.sub");
  return (
    <div className="safety-slot" data-state={a.state}>
      <button
        type="button"
        className={`safety-btn ${isStop ? "safety-stop" : "safety-home"}`}
        data-testid={isStop ? "safety-stop" : "safety-home"}
        data-state={a.state}
        disabled={disabledReason !== null}
        onClick={() => a.trigger()}
      >
        <Icon aria-hidden="true" size={26} />
        <span className="safety-text">
          <span className="safety-label">
            {t(isStop ? "safety.stop.label" : "safety.home.label")}
          </span>
          <span className="safety-sub">{sub}</span>
          {a.onboardReceived && a.state !== "acked" ? (
            <span className="safety-sub2">{t("safety.state.onboardReceived")}</span>
          ) : null}
        </span>
      </button>
      {a.state === "unconfirmed" || a.state === "failed" ? (
        <button
          type="button"
          className="safety-dismiss"
          aria-label={t("safety.dismiss")}
          onClick={() => a.dismiss()}
        >
          <X aria-hidden="true" size={16} />
        </button>
      ) : null}
      <span className="sr-only" aria-live="assertive">
        {status ?? ""}
      </span>
    </div>
  );
}
