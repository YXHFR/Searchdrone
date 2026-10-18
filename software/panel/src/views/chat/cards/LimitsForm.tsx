import { SlidersHorizontal } from "lucide-react";
import { useId, useState } from "react";
import { useTranslation } from "react-i18next";
import { api, HttpError } from "../../../api/http";
import { paths } from "../../../api/paths";
import type { Mission, SafetyLimits } from "../../../api/types";
import { useLiveStore } from "../../../state/liveStore";
import { useIsPilot, useSessionStore } from "../../../state/sessionStore";
import { useUiStore } from "../../../state/uiStore";
import { Dialog } from "../../../ui/Dialog";
import { limitProblems } from "./limits";

const FIELDS = [
  { key: "max_alt_agl_m", unit: "m" },
  { key: "max_speed_ms", unit: "m/s" },
  { key: "max_range_from_pilot_m", unit: "m" },
  { key: "battery_rtl_pct", unit: "%" },
] as const;

/**
 * "Ändra gränser": the mission's limits, by the pilot's own click only. Up to the outer cap
 * the server reports (limits_absolute); the battery may not go below its floor.
 */
export function LimitsForm({ onClose }: { onClose: () => void }) {
  const { t } = useTranslation();
  const baseId = useId();
  const mission = useLiveStore((s) => s.mission);
  const abs = useSessionStore((s) => s.config?.limits_absolute ?? {});
  const [values, setValues] = useState<SafetyLimits>(() => ({ ...(mission?.limits ?? {}) }));
  const [error, setError] = useState<string | null>(null);
  const [busy, setBusy] = useState(false);
  const problems = limitProblems(values, abs);
  const bad = new Map(problems.map((p) => [p.field, p]));
  const invalid = FIELDS.some((f) => !Number.isFinite(values[f.key]));

  async function save(): Promise<void> {
    if (!mission || problems.length > 0 || invalid) return;
    setBusy(true);
    setError(null);
    try {
      const updated = await api<Mission>(paths.missionLimits(mission.id), {
        method: "PUT",
        body: { ...mission.limits, ...values },
      });
      useLiveStore.getState().patch({ mission: updated });
      useUiStore.getState().toast(t("limitsForm.saved"));
      onClose();
    } catch (e) {
      if (e instanceof HttpError && e.code === "LIMITS_BELOW_ACTIVE_CHARTER") {
        setError(`${t("limitsForm.belowCharter")} ${e.body?.detail_sv ?? ""}`.trim());
      } else {
        setError(e instanceof HttpError ? e.message : t("limitsForm.failed"));
      }
    } finally {
      setBusy(false);
    }
  }

  return (
    <Dialog title={t("limitsForm.title")} onClose={onClose} testId="limits-form">
      <form
        className="limits-form"
        onSubmit={(e) => {
          e.preventDefault();
          void save();
        }}
      >
        {FIELDS.map((f) => {
          const p = bad.get(f.key);
          const id = `${baseId}-${f.key}`;
          return (
            <div key={f.key} className="limits-row" data-bad={p ? "true" : undefined}>
              <label htmlFor={id}>
                {t(`charter.field.${f.key}`)} ({f.unit})
              </label>
              <input
                id={id}
                className="input"
                type="number"
                inputMode="decimal"
                data-testid={`limit-${f.key}`}
                value={values[f.key] ?? ""}
                onChange={(e) => setValues({ ...values, [f.key]: e.target.valueAsNumber })}
              />
              {p ? (
                <span className="error">
                  {p.key === "limits.aboveCap"
                    ? t("limitsForm.aboveCap", { v: p.cap })
                    : t("limitsForm.belowFloor", { v: "floor" in p ? p.floor : "" })}
                </span>
              ) : (
                <span className="muted">
                  {f.key === "battery_rtl_pct"
                    ? t("limitsForm.floor", { v: abs.min_battery_rtl_pct ?? "–" })
                    : t("limitsForm.cap", { v: abs[f.key] ?? "–" })}
                </span>
              )}
            </div>
          );
        })}
        {error ? <p className="error">{error}</p> : null}
        <div className="dialog-actions">
          <button
            type="submit"
            className="btn btn-primary"
            data-testid="limits-save"
            disabled={busy || problems.length > 0 || invalid}
          >
            {t("limitsForm.save")}
          </button>
          <button type="button" className="btn btn-outline" onClick={onClose}>
            {t("limitsForm.cancel")}
          </button>
        </div>
      </form>
    </Dialog>
  );
}

/** The "Ändra gränser" button (charter card and the limits card). Pilot only. */
export function LimitsButton() {
  const { t } = useTranslation();
  const isPilot = useIsPilot();
  const hasMission = useLiveStore((s) => s.mission !== null);
  const [open, setOpen] = useState(false);
  if (!isPilot || !hasMission) return null;
  return (
    <>
      <button
        type="button"
        className="btn btn-outline"
        data-testid="limits-open"
        onClick={() => setOpen(true)}
      >
        <SlidersHorizontal aria-hidden="true" size={16} />
        {t("limitsForm.open")}
      </button>
      {open ? <LimitsForm onClose={() => setOpen(false)} /> : null}
    </>
  );
}
