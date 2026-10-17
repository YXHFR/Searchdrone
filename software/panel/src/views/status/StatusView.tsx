import { useTranslation } from "react-i18next";
import { vehiclePill } from "../../header/pills";
import { ResumeRow } from "../../safety/ResumeButton";
import { useLiveStore } from "../../state/liveStore";
import { activeCharter, useMissionStore } from "../../state/missionStore";
import { useSessionStore } from "../../state/sessionStore";
import { useUiStore } from "../../state/uiStore";
import { LimitsButton } from "../chat/cards/LimitsForm";
import type { ViewProps } from "../registry";
import { AutonomyCard, MissionCard } from "./MissionCards";

function Row({ label, value }: { label: string; value: string }) {
  return (
    <div className="kv">
      <dt>{label}</dt>
      <dd>{value}</dd>
    </div>
  );
}

const num = (x: number | null | undefined, digits = 0): string =>
  x === null || x === undefined
    ? "–"
    : new Intl.NumberFormat("sv-SE", { maximumFractionDigits: digits }).format(x);

function VehicleCard({ vehicleId }: { vehicleId: string }) {
  const { t } = useTranslation();
  const tel = useLiveStore((s) => s.telemetry);
  const vehicle = useLiveStore((s) => s.vehicle);
  const pill = vehiclePill(vehicle?.snapshot);
  return (
    <section className="card" aria-labelledby="st-vehicle">
      <h3 id="st-vehicle">{t("status.vehicle.title")}</h3>
      <dl>
        <Row label={t("status.vehicle.state")} value={pill.raw ?? t(pill.key)} />
        <Row
          label={t("status.vehicle.armed")}
          value={tel ? t(tel.armed ? "common.yes" : "common.no") : "–"}
        />
        <Row label={t("status.vehicle.mode")} value={tel?.flight_mode ?? "–"} />
        <Row
          label={t("status.vehicle.position")}
          value={tel ? `${num(tel.lat, 6)}, ${num(tel.lon, 6)}` : "–"}
        />
        <Row label={t("status.vehicle.alt")} value={tel ? `${num(tel.alt_rel_home_m)} m` : "–"} />
        <Row
          label={t("status.vehicle.speed")}
          value={tel ? `${num(tel.groundspeed_ms, 1)} m/s` : "–"}
        />
        <Row label={t("status.vehicle.heading")} value={tel ? `${num(tel.heading_deg)}°` : "–"} />
        <Row
          label={t("status.vehicle.gps")}
          value={tel ? t("status.vehicle.gpsValue", { fix: tel.gps_fix, sats: tel.gps_sats }) : "–"}
        />
        <Row label={t("status.vehicle.battery")} value={tel ? `${num(tel.battery_pct)} %` : "–"} />
      </dl>
      <ResumeRow vehicleId={vehicleId} />
    </section>
  );
}

function LinksCard() {
  const { t } = useTranslation();
  const ws = useLiveStore((s) => s.ws);
  const link = useLiveStore((s) => s.vehicle?.link ?? null);
  const brain = useLiveStore((s) => s.system?.parts.brain?.state ?? null);
  return (
    <section className="card" aria-labelledby="st-links">
      <h3 id="st-links">{t("status.links.title")}</h3>
      <dl>
        <Row label={t("status.links.ground")} value={t(`status.ws.${ws}`)} />
        <Row label={t("status.links.mavlink")} value={link?.mavlink ?? "–"} />
        <Row label={t("status.links.sdlink")} value={link?.sdlink ?? "–"} />
        <Row label={t("status.links.ai")} value={brain ?? "–"} />
      </dl>
    </section>
  );
}

function LimitsCard() {
  const { t } = useTranslation();
  const config = useSessionStore((s) => s.config);
  const missionLimits = useLiveStore((s) => s.mission?.limits ?? null);
  const lim = missionLimits ?? config?.limits_default ?? null;
  const abs = config?.limits_absolute ?? null;
  const activeId = useLiveStore((s) => s.mission?.active_charter_id ?? null);
  const charter = useMissionStore((s) => activeCharter(s, activeId))?.body ?? null;
  return (
    <section className="card" aria-labelledby="st-limits">
      <h3 id="st-limits">{t("status.limits.title")}</h3>
      <dl>
        <Row
          label={t("status.limits.alt")}
          value={`${num(lim?.max_alt_agl_m)} m (${t("status.limits.cap")} ${num(abs?.max_alt_agl_m)} m)`}
        />
        <Row
          label={t("status.limits.speed")}
          value={`${num(lim?.max_speed_ms)} m/s (${t("status.limits.cap")} ${num(abs?.max_speed_ms)} m/s)`}
        />
        <Row
          label={t("status.limits.range")}
          value={`${num(lim?.max_range_from_pilot_m)} m (${t("status.limits.cap")} ${num(abs?.max_range_from_pilot_m)} m)`}
        />
        <Row label={t("status.limits.battery")} value={`${num(lim?.battery_rtl_pct)} %`} />
        {charter ? (
          <Row
            label={t("status.limits.charter")}
            value={`${num(charter.alt_max_agl_m)} m · ${num(charter.max_speed_ms)} m/s · ${num(charter.max_range_from_pilot_m)} m · ${num(charter.battery_rtl_pct)} %`}
          />
        ) : null}
      </dl>
      <LimitsButton />
    </section>
  );
}

function EventLog() {
  const { t } = useTranslation();
  const log = useUiStore((s) => s.eventLog);
  const time = new Intl.DateTimeFormat("sv-SE", {
    hour: "2-digit",
    minute: "2-digit",
    second: "2-digit",
  });
  return (
    <section className="card" aria-labelledby="st-log">
      <h3 id="st-log">{t("status.log.title")}</h3>
      {log.length === 0 ? (
        <p className="muted">{t("status.log.empty")}</p>
      ) : (
        <ol className="event-log" data-testid="event-log">
          {log.map((e) => (
            <li key={`${e.t}-${e.message}`} data-level={e.level}>
              <time>{time.format(new Date(e.t))}</time> {e.message}
            </li>
          ))}
        </ol>
      )}
    </section>
  );
}

export default function StatusView({ vehicleId }: ViewProps) {
  return (
    <div className="view-body status-grid">
      <VehicleCard vehicleId={vehicleId} />
      <MissionCard />
      <AutonomyCard />
      <LinksCard />
      <LimitsCard />
      <EventLog />
    </div>
  );
}
