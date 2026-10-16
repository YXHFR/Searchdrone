import { useEffect, useRef, useState } from "react";
import { useTranslation } from "react-i18next";
import { useLiveStore } from "../state/liveStore";
import { cachedVehicleId, useSessionStore } from "../state/sessionStore";
import { SafetyButton } from "./SafetyButton";
import { useVisualViewportBottom } from "./useVisualViewportBottom";

const NO_TELEMETRY_MS = 3_000;

function useNow(everyMs: number): number {
  const [now, setNow] = useState(() => Date.now());
  useEffect(() => {
    const id = setInterval(() => setNow(Date.now()), everyMs);
    return () => clearInterval(id);
  }, [everyMs]);
  return now;
}

/**
 * STOP and RETURN HOME. Rendered outside the auth gate and every error boundary, so it is
 * there on the login page, the emergency page and while loading. Only these two buttons.
 */
export function SafetyBar() {
  const { t } = useTranslation();
  const ref = useRef<HTMLDivElement>(null);
  const bottom = useVisualViewportBottom();
  const status = useSessionStore((s) => s.status);
  const needsReauth = useSessionStore((s) => s.needsReauth);
  const role = useSessionStore((s) => s.session?.role ?? null);
  const configVehicle = useSessionStore((s) => s.config?.default_vehicle_id ?? null);
  const lastTelemetryAt = useLiveStore((s) => s.lastTelemetryAt);
  const now = useNow(1_000);
  const vehicleId = configVehicle ?? cachedVehicleId();
  const loggedOut = status === "login" || needsReauth;
  const noContact =
    status === "ready" && (lastTelemetryAt === null || now - lastTelemetryAt > NO_TELEMETRY_MS);

  // Everything else ends above the bar: publish its height as --safety-h.
  useEffect(() => {
    const el = ref.current;
    if (!el || typeof ResizeObserver === "undefined") return;
    const ro = new ResizeObserver(() => {
      document.documentElement.style.setProperty("--safety-h", `${el.offsetHeight}px`);
    });
    ro.observe(el);
    return () => ro.disconnect();
  }, []);

  return (
    <section
      ref={ref}
      className="safety-bar"
      aria-label={t("safety.region")}
      data-testid="safety-bar"
      style={{ bottom: `${bottom}px` }}
    >
      {loggedOut ? <p className="safety-note safety-note-warn">{t("safety.notLoggedIn")}</p> : null}
      {noContact ? <p className="safety-note">{t("safety.noVehicleContact")}</p> : null}
      <div className="safety-buttons">
        {/* Everyone may press STOP, observers included. */}
        <SafetyButton action="stop" vehicleId={vehicleId} />
        <SafetyButton
          action="return_home"
          vehicleId={vehicleId}
          disabledReason={role === "observer" ? t("safety.home.pilotOnly") : null}
        />
      </div>
    </section>
  );
}
