import { useTranslation } from "react-i18next";

/** No contact with the ground station. The safety bar is still there (it lives outside this page). */
export function EmergencyPage() {
  const { t } = useTranslation();
  return (
    <main className="page-center" id="main" data-testid="emergency">
      <div className="card">
        <h1>{t("emergency.title")}</h1>
        <p>{t("emergency.body")}</p>
        <button type="button" className="btn" onClick={() => window.location.reload()}>
          {t("emergency.reload")}
        </button>
      </div>
    </main>
  );
}
