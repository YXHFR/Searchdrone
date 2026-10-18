import { useState } from "react";
import { useTranslation } from "react-i18next";
import { api, HttpError } from "../../api/http";
import { paths } from "../../api/paths";
import type { AutonomyMode, Mission } from "../../api/types";
import { ClearanceControl } from "../../safety/ClearanceControl";
import { useLiveStore } from "../../state/liveStore";
import { activeCharter, useMissionStore } from "../../state/missionStore";
import { useIsPilot } from "../../state/sessionStore";
import { useUiStore } from "../../state/uiStore";
import { Dialog } from "../../ui/Dialog";

/** Läge A/B. Only a pilot changes it, only with an approved charter, always via a dialog. */
export function AutonomyCard() {
  const { t } = useTranslation();
  const isPilot = useIsPilot();
  const mission = useLiveStore((s) => s.mission);
  const charter = useMissionStore((s) => activeCharter(s, mission?.active_charter_id ?? null));
  const [ask, setAsk] = useState<AutonomyMode | null>(null);
  const [error, setError] = useState<string | null>(null);
  const mode = mission?.autonomy_mode ?? null;
  const other: AutonomyMode = mode === "WITHIN_CHARTER" ? "APPROVE_EACH" : "WITHIN_CHARTER";

  async function change(to: AutonomyMode): Promise<void> {
    if (!mission) return;
    setError(null);
    try {
      const m = await api<Mission>(paths.missionAutonomy(mission.id), {
        method: "PUT",
        body: { mode: to },
      });
      useLiveStore.getState().patch({ mission: m });
    } catch (e) {
      setError(e instanceof HttpError ? e.message : t("autonomy.failed"));
    } finally {
      setAsk(null);
    }
  }

  return (
    <section className="card" aria-labelledby="st-autonomy" data-testid="autonomy-card">
      <h3 id="st-autonomy">{t("autonomy.title")}</h3>
      <p>
        {mode ? t(mode === "WITHIN_CHARTER" ? "pills.mode.b" : "pills.mode.a") : t("autonomy.none")}
      </p>
      {isPilot ? (
        <>
          <button
            type="button"
            className="btn btn-outline"
            data-testid="autonomy-change"
            disabled={!charter || !mode}
            onClick={() => setAsk(other)}
          >
            {t(other === "WITHIN_CHARTER" ? "autonomy.toB" : "autonomy.toA")}
          </button>
          {!charter ? <p className="muted">{t("autonomy.needCharter")}</p> : null}
        </>
      ) : null}
      {error ? <p className="error">{error}</p> : null}
      {ask && charter ? (
        <Dialog
          title={t("autonomy.confirmTitle")}
          onClose={() => setAsk(null)}
          testId="autonomy-dialog"
        >
          <p>
            {t(ask === "WITHIN_CHARTER" ? "autonomy.confirmB" : "autonomy.confirmA", {
              v: charter.version,
            })}
          </p>
          <div className="dialog-actions">
            <button type="button" className="btn btn-primary" onClick={() => void change(ask)}>
              {t("autonomy.switch")}
            </button>
            <button type="button" className="btn btn-outline" onClick={() => setAsk(null)}>
              {t("limitsForm.cancel")}
            </button>
          </div>
        </Dialog>
      ) : null}
    </section>
  );
}

/** Title, status, the clearance, and "Avsluta uppdrag" (the emergency close). No create button. */
export function MissionCard() {
  const { t } = useTranslation();
  const isPilot = useIsPilot();
  const mission = useLiveStore((s) => s.mission);
  const [editing, setEditing] = useState(false);
  const [title, setTitle] = useState("");
  const [ending, setEnding] = useState(false);
  const [error, setError] = useState<string | null>(null);
  if (!mission) {
    return (
      <section className="card" aria-labelledby="st-mission">
        <h3 id="st-mission">{t("missionCard.title")}</h3>
        <p className="muted">{t("app.noMission")}</p>
      </section>
    );
  }

  async function saveTitle(): Promise<void> {
    if (!mission || !title.trim()) return;
    try {
      const m = await api<Mission>(paths.mission(mission.id), {
        method: "PATCH",
        body: { title: title.trim() },
      });
      useLiveStore.getState().patch({ mission: m });
      setEditing(false);
    } catch (e) {
      setError(e instanceof HttpError ? e.message : t("missionCard.failed"));
    }
  }

  async function end(): Promise<void> {
    if (!mission) return;
    setEnding(false);
    setError(null);
    try {
      const m = await api<Mission>(paths.missionEnd(mission.id), {
        method: "POST",
        body: { reason: "Avslutat av piloten i panelen" },
      });
      useLiveStore.getState().patch({ mission: m });
      useUiStore.getState().toast(t("missionCard.ended"));
    } catch (e) {
      setError(
        e instanceof HttpError && e.code === "VEHICLE_AIRBORNE"
          ? t("missionCard.airborne")
          : e instanceof HttpError
            ? e.message
            : t("missionCard.failed"),
      );
    }
  }

  return (
    <section className="card" aria-labelledby="st-mission" data-testid="mission-card">
      <h3 id="st-mission">{t("missionCard.title")}</h3>
      {editing ? (
        <form
          className="title-form"
          onSubmit={(e) => {
            e.preventDefault();
            void saveTitle();
          }}
        >
          <label className="sr-only" htmlFor="mission-title-input">
            {t("missionCard.name")}
          </label>
          <input
            id="mission-title-input"
            className="input"
            value={title}
            maxLength={200}
            onChange={(e) => setTitle(e.target.value)}
          />
          <button type="submit" className="btn btn-small btn-primary">
            {t("limitsForm.save")}
          </button>
        </form>
      ) : (
        <p>
          {mission.title}{" "}
          {isPilot ? (
            <button
              type="button"
              className="btn btn-small btn-outline"
              onClick={() => {
                setTitle(mission.title);
                setEditing(true);
              }}
            >
              {t("missionCard.rename")}
            </button>
          ) : null}
        </p>
      )}
      <p data-testid="mission-status">
        {t(`missionState.${mission.status}`, { defaultValue: mission.status })}
      </p>
      <ClearanceControl />
      {isPilot && mission.status !== "CLOSED" ? (
        <button
          type="button"
          className="btn btn-outline"
          data-testid="mission-end"
          onClick={() => setEnding(true)}
        >
          {t("missionCard.end")}
        </button>
      ) : null}
      {error ? <p className="error">{error}</p> : null}
      {ending ? (
        <Dialog
          title={t("missionCard.endTitle")}
          onClose={() => setEnding(false)}
          testId="end-dialog"
        >
          <p>{t("missionCard.endBody")}</p>
          <div className="dialog-actions">
            <button
              type="button"
              className="btn btn-primary"
              data-testid="end-confirm"
              onClick={() => void end()}
            >
              {t("missionCard.endConfirm")}
            </button>
            <button type="button" className="btn btn-outline" onClick={() => setEnding(false)}>
              {t("limitsForm.cancel")}
            </button>
          </div>
        </Dialog>
      ) : null}
    </section>
  );
}
