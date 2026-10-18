import { useEffect } from "react";
import { useTranslation } from "react-i18next";
import { boot, watchUnauthorized } from "../auth/bootstrap";
import { LoginPage } from "../auth/LoginPage";
import { SafetyBar } from "../safety/SafetyBar";
import { useSessionStore } from "../state/sessionStore";
import { useUiStore } from "../state/uiStore";
import { useTheme } from "../theme/theme";
import { EmergencyPage } from "./EmergencyPage";
import { ErrorBoundary } from "./ErrorBoundary";
import { Shell } from "./Shell";

function AuthGate() {
  const { t } = useTranslation();
  const status = useSessionStore((s) => s.status);
  const vehicleId = useSessionStore((s) => s.config?.default_vehicle_id ?? null);
  if (status === "login") return <LoginPage />;
  if (status === "emergency") return <EmergencyPage />;
  if (status === "loading" || !vehicleId) {
    return (
      <main className="page-center" id="main">
        <p className="muted">{t("common.loading")}</p>
      </main>
    );
  }
  return (
    <ErrorBoundary fallback={() => <EmergencyPage />}>
      <Shell vehicleId={vehicleId} />
    </ErrorBoundary>
  );
}

/** DOM order: skip link, safety bar (fixed at the bottom), then everything else. */
export function App() {
  const { t } = useTranslation();
  const theme = useTheme();
  const alert = useUiStore((s) => s.alert);
  useEffect(() => {
    document.documentElement.dataset.theme = theme;
  }, [theme]);
  useEffect(() => {
    const stop = watchUnauthorized();
    void boot();
    return stop;
  }, []);
  return (
    <>
      <a className="skip-link" href="#main">
        {t("app.skip")}
      </a>
      <SafetyBar />
      <AuthGate />
      <div className="sr-only" aria-live="assertive">
        {alert ?? ""}
      </div>
    </>
  );
}
