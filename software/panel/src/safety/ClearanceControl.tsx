import { ShieldCheck } from "lucide-react";
import { useEffect, useId, useState } from "react";
import { useTranslation } from "react-i18next";
import { api, HttpError } from "../api/http";
import { paths } from "../api/paths";
import type { Clearance } from "../api/types";
import { mmss, useNow } from "../lib/useNow";
import { useLiveStore } from "../state/liveStore";
import { activeCharter, useMissionStore } from "../state/missionStore";
import { useIsPilot, useSessionStore } from "../state/sessionStore";
import { clearanceAvailable } from "./clearance";

/** True while the vehicle is on the ground with an installed charter (see clearanceAvailable). */
export function useClearanceAvailable(): boolean {
  const isPilot = useIsPilot();
  const vehicle = useLiveStore((s) => s.vehicle);
  const telemetry = useLiveStore((s) => s.telemetry);
  const activeId = useLiveStore((s) => s.mission?.active_charter_id ?? null);
  const charter = useMissionStore((s) => activeCharter(s, activeId));
  return clearanceAvailable(isPilot, charter, vehicle, telemetry);
}

/**
 * "Fritt runt drönaren": the pilot's own confirmation before take-off. Its own click, its own
 * endpoint; never combined with an approval and never sent as part of one.
 */
export function ClearanceControl({ force = false }: { force?: boolean }) {
  const { t } = useTranslation();
  const id = useId();
  const available = useClearanceAvailable();
  const missionId = useLiveStore((s) => s.mission?.id ?? null);
  const clearance = useMissionStore((s) => s.clearance);
  const sim = useSessionStore((s) => s.config?.vehicle_kind === "sim");
  const [checked, setChecked] = useState(false);
  const [busy, setBusy] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const now = useNow(500);
  const active = clearance.status === "active";
  useEffect(() => {
    if (!active) setChecked(false); // consumed, expired, invalidated or none: start over
  }, [active]);
  if (!available && !force) return null;
  if (!missionId) return null;

  async function confirm(): Promise<void> {
    if (!missionId) return;
    setBusy(true);
    setError(null);
    try {
      const c = await api<Clearance>(paths.missionClearance(missionId), { method: "POST" });
      useMissionStore.getState().setClearance({ status: c.status, valid_until: c.valid_until });
    } catch (e) {
      setError(e instanceof HttpError ? e.message : t("clearance.failed"));
    } finally {
      setBusy(false);
    }
  }

  if (active) {
    const left = clearance.valid_until ? Date.parse(clearance.valid_until) - now : 0;
    return (
      <div className="clearance clearance-on" data-testid="clearance" role="status">
        <ShieldCheck aria-hidden="true" size={18} />
        {t("clearance.active", { left: mmss(left) })}
      </div>
    );
  }
  return (
    <div className="clearance" data-testid="clearance">
      {sim ? <span className="tag">{t("app.simulator")}</span> : null}
      <label htmlFor={id} className="clearance-check">
        <input
          id={id}
          type="checkbox"
          checked={checked}
          onChange={(e) => setChecked(e.target.checked)}
        />
        {t("clearance.check")}
      </label>
      <button
        type="button"
        className="btn btn-primary"
        data-testid="clearance-button"
        disabled={!checked || busy}
        onClick={() => void confirm()}
      >
        <ShieldCheck aria-hidden="true" size={18} />
        {t("clearance.button")}
      </button>
      {error ? <p className="error">{error}</p> : null}
    </div>
  );
}
