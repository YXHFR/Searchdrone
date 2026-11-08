// Inställningar (the pilot's): the overflight permit for a real search. A closer look may fly
// straight over its candidate in SKARPT only with a permit in the specific category; without
// one it looks from the side. Turning it on counts from the next approved charter, turning it
// off at once. The ground keeps this until it restarts; its config file holds the lasting one.
import { useEffect, useState } from "react";
import { useTranslation } from "react-i18next";
import { api, HttpError } from "../../api/http";
import { paths } from "../../api/paths";
import type { Permit } from "../../api/types";
import { useIsPilot } from "../../state/sessionStore";
import type { ViewProps } from "../registry";

const REF_MAX = 64;

export default function SettingsView(_: ViewProps) {
  const { t } = useTranslation();
  const isPilot = useIsPilot();
  const [saved, setSaved] = useState<Permit | null>(null);
  const [ref, setRef] = useState("");
  const [allow, setAllow] = useState(false);
  const [failed, setFailed] = useState(false);
  const [busy, setBusy] = useState(false);
  const [msg, setMsg] = useState<{ ok: boolean; text: string } | null>(null);

  useEffect(() => {
    let live = true;
    api<Permit>(paths.permit())
      .then((p) => {
        if (!live) return;
        setSaved(p);
        setRef(p.specific_category_ref ?? "");
        setAllow(Boolean(p.allow_overflight_live));
      })
      .catch(() => live && setFailed(true));
    return () => {
      live = false;
    };
  }, []);

  const hasRef = ref.trim() !== "";
  const changed =
    saved !== null &&
    (ref.trim() !== (saved.specific_category_ref ?? "") ||
      (allow && hasRef) !== Boolean(saved.allow_overflight_live));

  const save = async (): Promise<void> => {
    setBusy(true);
    setMsg(null);
    try {
      const body: Permit = {
        specific_category_ref: ref.trim(),
        allow_overflight_live: allow && hasRef,
      };
      const p = await api<Permit>(paths.permit(), { method: "PUT", body });
      setSaved(p);
      setRef(p.specific_category_ref ?? "");
      setAllow(Boolean(p.allow_overflight_live));
      setMsg({ ok: true, text: t("settings.saved") });
    } catch (e) {
      setMsg({ ok: false, text: e instanceof HttpError ? e.message : t("settings.failed") });
    } finally {
      setBusy(false);
    }
  };

  return (
    <div className="view-body settings-view" data-testid="settings">
      <section className="card" aria-labelledby="set-permit">
        <h3 id="set-permit">{t("settings.permit.title")}</h3>
        <p className="muted">{t("settings.permit.about")}</p>
        {failed ? <p className="warn">{t("settings.loadFailed")}</p> : null}
        {!isPilot ? <p className="muted">{t("settings.pilotOnly")}</p> : null}
        <label className="settings-field">
          {t("settings.permit.ref")}
          <input
            className="input"
            value={ref}
            maxLength={REF_MAX}
            disabled={!isPilot || saved === null}
            autoComplete="off"
            data-testid="permit-ref"
            onChange={(e) => setRef(e.target.value)}
          />
        </label>
        <label className="settings-check">
          <input
            type="checkbox"
            checked={allow && hasRef}
            disabled={!isPilot || saved === null || !hasRef}
            data-testid="permit-allow"
            onChange={(e) => setAllow(e.target.checked)}
          />
          {t("settings.permit.allow")}
        </label>
        {!hasRef ? <p className="muted">{t("settings.permit.needRef")}</p> : null}
        <p className="muted">{t("settings.permit.when")}</p>
        <button
          type="button"
          className="btn btn-primary"
          disabled={!isPilot || busy || !changed}
          data-testid="permit-save"
          onClick={() => void save()}
        >
          {t("settings.save")}
        </button>
        {msg ? (
          <p className={msg.ok ? "muted" : "warn"} role="status" data-testid="permit-msg">
            {msg.text}
          </p>
        ) : null}
      </section>
    </div>
  );
}
