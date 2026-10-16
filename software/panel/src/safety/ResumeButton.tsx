import { Play } from "lucide-react";
import { useEffect, useState } from "react";
import { useTranslation } from "react-i18next";
import { useLiveStore } from "../state/liveStore";
import { useIsPilot } from "../state/sessionStore";
import { resumeView } from "./resume";
import { actionText } from "./SafetyButton";
import { useVehicleAction } from "./useVehicleAction";

const ARM_DELAY_MS = 2_000;

/** "Fortsätt": the only way to release a hold. Never in the safety bar. */
export function ResumeButton({ vehicleId }: { vehicleId: string }) {
  const { t } = useTranslation();
  const a = useVehicleAction(vehicleId, "resume");
  const [armed, setArmed] = useState(false);
  useEffect(() => {
    const id = setTimeout(() => setArmed(true), ARM_DELAY_MS);
    return () => clearTimeout(id);
  }, []);
  const status = actionText(t, a.text);
  return (
    <span className="resume">
      <button
        type="button"
        className="btn btn-outline resume-btn"
        data-testid="resume"
        disabled={!armed}
        onClick={() => a.trigger()}
      >
        <Play aria-hidden="true" size={16} />
        {t("resume.button")}
      </button>
      {status ? <span className="muted"> {status}</span> : null}
    </span>
  );
}

/** System line shown in chat and in the vehicle card when Resume applies. */
export function ResumeRow({ vehicleId }: { vehicleId: string }) {
  const { t } = useTranslation();
  const vehicle = useLiveStore((s) => s.vehicle);
  const isPilot = useIsPilot();
  const view = resumeView(vehicle);
  if (view === "none") return null;
  if (view === "home_active") return <p className="system-line">{t("resume.homeActive")}</p>;
  if (view === "onboard_silent") return <p className="system-line">{t("resume.onboardSilent")}</p>;
  return (
    <div className="system-line" data-testid="resume-row">
      <p>{t("resume.held")}</p>
      {isPilot ? <ResumeButton vehicleId={vehicleId} /> : null}
    </div>
  );
}
